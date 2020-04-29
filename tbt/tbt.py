import copy
import enum
from typing import Union, Callable, Dict, List, Iterable, Tuple, Optional, Any

import numpy as np
import pyIOTA.acnet
import pyIOTA.iota.run2
import pyIOTA.iota.run2 as iota
import scipy as sc
import pandas as pd
from scipy.signal import hilbert, chirp, butter, filtfilt

# special_keys = ['idx', 'kickv', 'kickh', 'state', 'custom']
from pyIOTA.tbt.naff import NAFF

import pyIOTA.acnet.utils as acutils

special_keys = acutils.special_keys.copy()

critical_keys = ['kickv', 'kickh', 'idx']


# def FilteredSignalButter(signal, fs, cutoff):
#     B, A = butter(1, cutoff / (fs / 2), btype='low')
#     filtered_signal = filtfilt(B, A, signal, axis=0)
#     return filtered_signal


class Kick:
    @enum.unique
    class Datatype(enum.Enum):
        """
        Enum containing the most common types of data in a kick. Intended to be used with data retrieval methods.
        """
        RAW = ''
        FFT_FREQ = 'fft_freq_'
        FFT_POWER = 'fft_power_'
        NUX = 'nux'
        NUY = 'nuy'

    def __init__(self,
                 df: pd.DataFrame,
                 kick_id: int = -1,
                 bpm_list: Optional[Iterable] = None,
                 parent_sequence: Optional['KickSequence'] = None,
                 file_name: str = None,
                 trim: Tuple = None):
        for ck in critical_keys:
            if ck not in df.columns:
                raise Exception(f'Missing critical key ({ck})')

        self.df = df
        self.idx = kick_id
        self.set('idx', kick_id)
        self.ks = parent_sequence
        self.file_name = file_name
        self.matrix_cache = {}  # old way of caching BPM matrices
        self.trim = self.default_trim = trim
        self.force_own_trim = False  # if True, all analysis method will use kick trim instead of their own

        if not bpm_list:
            bpm_list = set([k[:-1] for k in pyIOTA.iota.run2.BPMS.ALLA])
            # bpm_list = set([k[:-1] for k in df.columns if k not in special_keys])
            # print(f'BPM list not specified - deducing {len(bpm_list)}: {bpm_list}')
        self.H = [i + "H" for i in bpm_list]
        self.V = [i + "V" for i in bpm_list]
        self.S = [i + "S" for i in bpm_list]
        self.ALL = []
        self.C = []  # Calculated
        self.bpms_update()

        self.bpm_families_active = {'H': self.H, 'V': self.V, 'S': self.S, 'A': self.ALL, 'C': self.C}
        self.bpm_families_all = copy.deepcopy(self.bpm_families_active)

        # Convenience attributes
        self.n_turns = self.get_turns()
        self.kickv = self.get('kickv')
        self.kickh = self.get('kickh')
        self.nux = None  # main tune
        self.nuy = None  # main tune
        self.fft_pwr = self.fft_freq = self.peaks = None

    def __getitem__(self, key):
        return self.get(key)

    def copy(self) -> 'Kick':
        df2 = self.df.copy(deep=True)
        kick = Kick(df=df2, kick_id=self.idx, trim=self.trim, parent_sequence=self.ks)
        kick.H = self.H
        kick.V = self.V
        kick.S = self.S
        kick.bpms_update()
        # dont copy other attributes
        return kick

    # These are wrapper methods

    def set(self, column: str, value: Any):
        """
        Direct column setter
        :param column:
        :param value:
        """
        self.df.iloc[0, self.df.columns.get_loc(column)] = value

    def get(self, column: str):
        """
        Direct column getter
        :param column:
        :return:
        """
        return self.df.iloc[0, self.df.columns.get_loc(column)]

    def col(self, column: str):
        """
        Return a column of underlying dataframe as value or array (different from KickSequence!!!!)
        :param column:
        :return:
        """
        return self.df.iloc[0, self.df.columns.get_loc(column)]

    def state(self, param: str):
        """
        Returns value of specified key in state dictionary
        """
        return self.df.iloc[0, self.df.columns.get_loc('state')][param]

    def get_full_state(self):
        """
        Gets a copy of full kick state
        """
        return self.df.iloc[0, self.df.columns.get_loc('state')].copy()

    def summarize(self):
        print(f'Kick {self.idx}: ({self.get_turns()}) turns at ({self.kickh:.5f})H ({self.kickv:.5f})V')
        self.bpms_summarize_status()

    def set_trim(self, trim: Tuple):
        self.trim = trim

    def reset_trim(self):
        self.trim = self.default_trim

    def suggest_trim(self, min_idx: int, max_idx: int, threshold: float = 0.2, verbose: bool = False):
        """
        Finds longest signal trim within constraints, based on local SNR
        :param min_idx: Starting index
        :param max_idx: Maximum end index
        :param threshold: Minimum signal fraction
        :param verbose:
        :return: Trim tuple
        """
        offsets = {}
        families = ['H', 'V']
        for fam in families:
            for k, v in self.get_bpm_data(family=fam, no_trim=True).items():
                v = v[min_idx:max_idx] - np.mean(v[min_idx:max_idx])
                initial_ampl = np.mean(np.abs(v[:100]))
                offset = 0
                while True:
                    ampl = np.mean(np.abs(v[offset:50 + offset]))
                    if ampl < initial_ampl * threshold:
                        break
                    elif offset + 50 > max_idx - min_idx:
                        break
                    elif offset + 55 > len(v):
                        raise Exception
                    offset += 5
                if verbose: print(f'BPM {k}: Iampl {initial_ampl:.3f}, Fampl {ampl:.3f}, offset {offset}')
                offsets[k] = offset
        if verbose: print(f'Found trims: {min_idx} + {offsets}')
        offset_avg = int(np.round(np.mean(list(offsets.values()))))
        if verbose: print(f'Average offset: {offset_avg}')
        return np.s_[min_idx:min_idx + offset_avg + 50]

    def search_state(self, search_string: str):
        """
        Searches state for keys matching regex expression
        """
        import re
        r = re.compile(search_string)
        state = self.col('state')
        match_keys = list(filter(r.match, state.keys()))
        return {k: state[k] for k in match_keys}

    def get_column_names(self, bpms: List = None, family: str = None, data_type: Datatype = Datatype.RAW):
        """
        Determines final column names
        :param bpms:
        :param family:
        :param data_type:
        :return:
        """
        assert bpms or family
        if not bpms:
            bpms = self.get_bpms(family)
        return [b + data_type.value for b in bpms]

    def __list_to_matrix(self, data_list):
        """
        Converts list of arrays into 2D matrix, stacked vertically, with appropriate error checking
        :param data_list:
        :return:
        """
        if not all(isinstance(v, np.ndarray) for v in data_list) or not all(v.ndim == 1 for v in data_list):
            raise Exception(f'List elements are not all 1D arrays')
        lengths = np.array([len(v) for v in data_list])
        if not np.all(lengths == lengths[0]):
            raise Exception(f'Data lengths are not equal: ({lengths})')
        matrix = np.vstack(data_list)
        return matrix

    def get_bpm_data(self,
                     columns: List[str] = None,
                     bpms: List[str] = None,
                     family: str = 'A',
                     data_type: Datatype = Datatype.RAW,
                     return_type: str = None,
                     use_cache: bool = True,
                     add_to_cache: bool = True,
                     no_trim: bool = False):
        """
        General data retrieval method for data that is per-bpm
        :param no_trim: Force full data, ignoring kick trim
        :param add_to_cache:
        :param use_cache:
        :param family: Family (aka plane) of BPMs, used if no bpms/columns provided
        :param return_type:
        :param data_type:
        :param columns: List of columns, supercedes all other parameters
        :param bpms: List of bpms (supercedes family)
        :param data_type: The type of data to return - by default, it is raw TBT data
        :return:
        """
        # catch shorthand 1 bpm notation
        if not isinstance(columns, list) and columns is not None:
            columns = [columns]
            if return_type is None:
                return_type = 'single'
        return_type = return_type or 'dict'
        assert columns is None or isinstance(columns, list)
        assert bpms is None or isinstance(bpms, list)
        assert columns or bpms or family
        if not columns:
            columns = self.get_column_names(bpms, family, data_type)

        if self.trim and not no_trim:
            data = {c: self.col(c) for c in columns}
            is_array = [isinstance(v, np.ndarray) for v in data.values()]
            if all(is_array):
                data = {k: v[self.trim] for k, v in data.items()}
            elif all([not i for i in is_array]):
                pass  # trim does not apply, return is uniform
            else:
                raise Exception('WARN - data has both arrays and non-arrays, trim is ambiguous')
        else:
            data = {c: self.col(c) for c in columns}

        if return_type == 'dict':
            return data
        elif return_type == 'list':
            return list(data.values())
        elif return_type == 'matrix':
            # This return type copies data and is bad for performance...
            # if not all(isinstance(v, np.ndarray) for k,v in data.items()) or not all(v.ndim == 1 for k,v in data.items()):
            #     raise Exception(f'Columns ({columns}) do not all contain 1D arrays')
            # lengths = [len(v) for k,v in data.items()]
            # if not np.all(lengths == lengths[0]):
            #     raise Exception(f'Data length of columns ({columns}) are not equal: ({lengths})')
            # matrix = np.vstack(list(data.values()))
            return self.__list_to_matrix(list(data.values()))
        elif return_type == 'single':
            return list(data.values())[0]
        else:
            raise Exception(f'Unknown return type ({return_type}')

    def get_bpm_matrix(self, family: str = 'V', remove_sequence_data: bool = False) -> np.ndarray:
        """
        Returns all raw data of active BPMs in a family as 2D array (M BPMs x N turns)
        :param family:
        :param remove_sequence_data: whether to remove first point, which in IOTA is sequence number
        :return:
        """
        if family in ['H', 'V', 'S']:
            if family in self.matrix_cache:
                data = self.matrix_cache[family]
            else:
                bpms = self.get_bpms(family)
                data = np.vstack(self.df.loc[self.df.index[0], bpms])
                self.matrix_cache[family] = data
            if remove_sequence_data:
                return data[:, 1:]
            else:
                return data
        else:
            raise Exception("Invalid family specified")

    def as_dict(self, family: str = 'A', bpmlist: list = None, trim: Tuple = (1, -1)) -> dict:
        datadict = {}
        if bpmlist is None:
            bpms = self.get_bpms(family)
            for b in bpms:
                # datadict[b] = df.loc[idx, b].values[0]
                if trim:
                    datadict[b] = self.col(b)[trim[0]:trim[1]]
                else:
                    datadict[b] = self.col(b)
        else:
            raise Exception(f'BPM list not supported yet')
        return datadict

    def col_fft(self, bpm: str):
        """
        Return FFT of specified BPM
        :param bpm:
        :return:
        """
        return self.col('fft_freq_' + bpm), self.col('fft_pwr_' + bpm)

    def get_tune_data(self, bpm: str):
        return self.fft_freq[bpm], self.fft_pwr[bpm], self.peaks[bpm]

    def get_tunes(self, bpms: List[str]):
        if isinstance(bpms, str):
            return self.get('nu_' + bpms)
        else:
            return (*[self.get('nu_' + b) for b in bpms],)

    # Import/export

    def to_csv(self, columns: List) -> str:
        """
        Exports selected data to a CSV format. All selected columns must contain arrays of equal length.
        :return:
        """
        data = [self.col(c) for c in columns]
        matrix = self.__list_to_matrix(data)
        df_export = pd.DataFrame(columns=columns, data=matrix.T)
        df_export = df_export.rename_axis(index='Turn #')
        return df_export.to_csv()

    # BPM management functions

    def get_bpms(self, family: Union[List, str] = 'A', soft_fail: bool = False) -> List[str]:
        """
        Retrieves all active BPMs in specified families, returning a unique list
        :param family:
        :return:
        """
        bpms = []
        if isinstance(family, str):
            family = [family]
        for fam in family:
            if fam in self.bpm_families_active:
                bpms += self.bpm_families_active[fam]
            else:
                raise Exception
        if len(bpms) == 0:
            if not soft_fail: raise Exception(f'No BPMs found for families: ({family})')
        # Order-preserving unique list conversion
        seen = set()
        bpms = [x for x in bpms if not (x in seen or seen.add(x))]
        return bpms

    def bpms_update(self):
        self.ALL = self.H + self.V + self.S

    def bpms_add(self, bpms: List):
        for b in bpms:
            if b not in self.C:
                self.C.append(b)

    def bpms_disable(self, bpms: List, plane: str = 'A'):
        # Old signature fix
        if isinstance(bpms, str):
            bpms = [bpms]
        for bpm in bpms:
            if plane == 'A':
                n_removed = 0
                for sp in ['H', 'V', 'S']:
                    bpm_list = self.bpm_families_active[sp]
                    if bpm + sp in bpm_list:
                        bpm_list.remove(bpm + sp)
                        n_removed += 1
                if n_removed not in [0, 3]:
                    raise Exception(f'BPM ({bpm}) only got removed from ({n_removed}) lists, not 3 or 0')
                for sp in ['C']:
                    bpm_list = self.bpm_families_active[sp]
                    if bpm + sp in bpm_list:
                        bpm_list.remove(bpm + sp)
                self.bpms_update()
            elif plane == 'C':
                sp = plane
                bpm_list = self.bpm_families_active[sp]
                if bpm + sp in bpm_list:
                    bpm_list.remove(bpm + sp)
                if bpm in bpm_list:
                    bpm_list.remove(bpm)
            else:
                raise Exception(f'Removing BPM ({bpm}) in only 1 physical plane is not supported')
        self.bpms_update()

    def bpms_summarize_status(self):
        for (k, v) in self.bpm_families_active.items():
            v_all = self.bpm_families_all[k]
            delta = set(v_all).difference(set(v))
            print(f'Plane ({k}): ({len(v_all)}) BPMs total, ({len(v)}) active, disabled: ({delta})')

    def bpms_apply_filters(self,
                           plane: str,
                           methods: List[Tuple[Callable, Tuple]],
                           data_type: Datatype = Datatype.RAW,
                           delete_on_fail: bool = False,
                           debug: bool = True):
        """
        Applies the provided methods with the parameter tuples, and determines which BPMs failed the tests
        :param plane:
        :param methods:
        :param data_type:
        :param delete_on_fail:
        :param debug:
        :return:
        """
        r = []
        r_data = []
        data = self.get_bpm_data(family=plane, data_type=data_type)

        for m, kw in methods:
            kw = kw or {}
            (results, results_data) = m(data, **kw)
            r.append(results)
            r_data.append(results_data)

        f_list = []
        f_data = []
        for results, results_data, (m, kw) in zip(r, r_data, methods):
            failures = {}
            fail_results = {}
            for k, v in results.items():
                if not v:
                    if debug: print(f'BPM ({k}) failed ({m.__name__}) - ({results_data[k]})')
                    failures[k] = data[k]
                    fail_results[k] = results_data[k]
            f_list.append(failures)
            f_data.append(fail_results)

        if delete_on_fail:
            deletion_set = set()
            for failures in f_list:
                keys = list(failures.keys())
                deletion_set += keys
            if debug: print(f'Deleting ({len(deletion_set)}) BPMs: ({deletion_set})')
            self.bpms_disable(list(deletion_set))

        return f_list, f_data, r, r_data

    def bpms_apply_filter(self,
                          plane: str,
                          method: Callable,
                          method_kwargs: Dict = None,
                          data_type: Datatype = Datatype.RAW,
                          delete_on_fail: bool = False,
                          silent: bool = True) -> Tuple[Dict, Dict]:
        """
        Applies the provided method onto each bpm in a plane, optionally removing those that failed. DEPRECATED.
        :param silent:
        :param delete_on_fail:
        :param data_type:
        :param method_kwargs:
        :param plane:
        :param method:
        :return:
        """
        data = self.get_bpm_data(family=plane, data_type=data_type)
        method_kwargs = method_kwargs or {}
        failures = {}
        fail_results = {}
        (results, results_data) = method(data, **method_kwargs)
        for k, v in results.items():
            if not v:
                if not silent: print(f'BPM ({k}) failed ({method.__name__}) - ({results_data[k]})')
                failures[k] = v
                fail_results[k] = results_data[k]
        if delete_on_fail:
            for k in failures:
                self.bpms_disable(k)
        return failures, fail_results

    def bpms_filter_outliers(self,
                             data: Dict,
                             method: str = 'mean',
                             sigma: float = 2.0,
                             use_percentile_std: bool = True):
        if method == 'mean':
            means = {k: np.mean(v) for k, v in data.items()}
            means_vals = np.array([v for v in means.values()])
            mean = np.mean(means_vals)
            if use_percentile_std:
                pcs = np.percentile(means_vals, [10, 90])
                std = np.std(means_vals[(means_vals > pcs[0]) & (means_vals < pcs[1])])
            else:
                std = np.std(means_vals)
            outliers = {k: v for k, v in means.items() if v > mean + sigma * std or v < mean - sigma * std}
            results = {k: k not in outliers for k, v in data.items()}
            results_data = {k: {'method': 'mean', 'values': (mean, std * sigma, v)} for k, v in means.items()}
        else:
            raise Exception(f'Method ({method}) is unrecognized')
        return results, results_data

    def bpms_filter_absval(self,
                           data: Dict,
                           method: str = 'abs',
                           threshold: float = 10.0,
                           neg_threshold: float = None):
        if method == 'abs':
            neg_threshold = neg_threshold or -threshold
            vals1 = {k: np.max(v) for k, v in data.items()}
            vals2 = {k: np.min(v) for k, v in data.items()}

            outliers = {k: v for k, v in vals1.items() if v > threshold}
            outliers2 = {k: v for k, v in vals2.items() if v < neg_threshold}
            results = {k: k not in outliers and k not in outliers2 for k, v in data.items()}
            results_data = {k: {'method': 'abs', 'values': (threshold, neg_threshold, v, v2)} for
                            (k, v), (k2, v2) in zip(vals1.items(), vals2.items())}
        else:
            raise Exception(f'Method ({method}) is unrecognized')
        return results, results_data

    def bpms_filter_symmetry(self,
                             data: Dict,
                             method: str = 'mean',
                             threshold: float = 1.2):
        if method == 'mean':
            data_demean = {k: v[1:] - np.mean(v[1:]) for k, v in data.items()}
            vals1 = {k: (np.mean(v[v > 0.0]), -np.mean(v[v < 0.0])) for k, v in data_demean.items()}
            outliers = {k: v for k, v in vals1.items() if v[0] / v[1] > threshold or v[1] / v[0] > threshold}
            results = {k: k not in outliers for k, v in data.items()}
            results_data = {k: {'method': method,
                                'values': (threshold, v[0], v[1], v[0] / v[1], v[1] / v[0])} for (k, v) in
                            vals1.items()}
        elif method == 'pilot':
            pilot_slice = np.s_[1:171]
            data_demean = {k: v[1:] - np.mean(v[pilot_slice]) for k, v in data.items()}
            vals1 = {k: (np.mean(v[v > 0.0]), -np.mean(v[v < 0.0])) for k, v in data_demean.items()}
            outliers = {k: v for k, v in vals1.items() if v[0] / v[1] > threshold or v[1] / v[0] > threshold}
            results = {k: k not in outliers for k, v in data.items()}
            results_data = {k: {'method': method,
                                # 'values': (threshold, v[0], v[1], v[0]/v[1], v[1]/v[0])} for (k, v) in vals1.items()}
                                'values': f'+avg:{v[0]:.3f} vs -avg:{v[1]:.3f} above ratio of {threshold}'}
                            for (k, v) in vals1.items()}
        elif method == 'endpilot':
            pilot_slice = np.s_[500:]
            data_demean = {k: v[172:] - np.mean(v[pilot_slice]) for k, v in data.items()}
            vals1 = {k: (np.mean(v[v > 0.0]), -np.mean(v[v < 0.0])) for k, v in data_demean.items()}
            outliers = {k: v for k, v in vals1.items() if v[0] / v[1] > threshold or v[1] / v[0] > threshold}
            results = {k: k not in outliers for k, v in data.items()}
            results_data = {k: {'method': method,
                                'values': (threshold, v[0], v[1], v[0] / v[1], v[1] / v[0])} for (k, v) in
                            vals1.items()}
        else:
            raise Exception(f'Method ({method}) is unrecognized')
        return results, results_data

    def bpms_filter_signal_ratio(self,
                                 data: Dict,
                                 method: str = 'std',
                                 threshold: int = 2,
                                 splits: int = 4,
                                 relative: bool = True):
        """
        Tests if first region of signal has specified property, optionally relative to the last region
        """
        results = {}
        results_data = {}
        for k, signal in data.items():
            assert len(signal) >= splits
            split_signals = np.array_split(signal, splits)
            if method == 'std' or method == 'rms':
                fun = np.std
            elif method == 'ptp':
                fun = np.ptp
            else:
                raise Exception(f'Method ({method}) is unrecognized')

            if not relative:
                val1 = fun(split_signals[0])
                result = val1 > threshold
                results[k] = result
                results_data[k] = {'test': method, 'values': (threshold, val1)}
            else:
                val1, val2 = fun(split_signals[0]), fun(split_signals[-1])
                result = val1 / val2 > threshold
                results[k] = result
                results_data[k] = {'test': method + '_relative', 'values': (threshold, val1, val2, val1 / val2,)}
        return results, results_data

    # Helpers for state categories

    def get_optics(self):
        """
        Gets all relevant optics settings
        """
        quads = [q + '.SETTING' for q in pyIOTA.iota.QUADS.ALL_CURRENTS]
        squads = [q + '.SETTING' for q in pyIOTA.iota.SKEWQUADS.ALL_CURRENTS]
        return {k: self.state(k) for k in quads + squads}

    def get_correctors(self):
        """
        Gets all available corrector settings
        """
        elements = [q + '.SETTING' for q in pyIOTA.iota.CORRECTORS.ALL_VIRTUAL]
        return {k: self.state(k) for k in elements}

    def get_sextupoles(self):
        """
        Gets sextupoles settings
        """
        elements = [q + '.SETTING' for q in pyIOTA.iota.SEXTUPOLES.ALL_CURRENTS]
        return {k: self.state(k) for k in elements}

    def compute_tunes_naff(self):
        pass

    def get_turns(self) -> int:
        bpm = self.get_bpms()[0]
        return len(self.col(bpm))

    ### Physics starts here

    def calculate_tune(self,
                       naff: NAFF,
                       families: List[str] = None,
                       selector: Callable = None,
                       search_kwargs: Dict[str, int] = None,
                       use_precalculated: bool = True):
        """
        Calculates tune by finding peaks in FFT data, optionally using precomputed data to save time
        :param families: Families to perform calculation on - typically H, V, or C
        :param naff:
        :param selector:
        :param search_kwargs:
        :param use_precalculated:
        :return:
        """
        families = families or ['H', 'V']
        bpms = self.get_bpms(families)
        freq = {}
        pwr = {}
        peaks = {}
        average_tunes = {f: [] for f in families}
        for i, bpm in enumerate(bpms):
            col_fr = 'fft_freq_' + bpm
            col_pwr = 'fft_pwr_' + bpm
            if use_precalculated and col_fr in self.df.columns and col_pwr in self.df.columns:
                top_tune, peak_tunes, peak_idx, peak_props, (pf, pp) = naff.fft_peaks(data=None,
                                                                                      search_peaks=True,
                                                                                      search_kwargs=search_kwargs,
                                                                                      fft_freq=self.df.iloc[0].loc[
                                                                                          col_fr],
                                                                                      fft_power=self.df.iloc[0].loc[
                                                                                          col_pwr])
            else:
                top_tune, peak_tunes, peak_idx, peak_props, (pf, pp) = naff.fft_peaks(data=self.df.iloc[0].loc[bpm],
                                                                                      search_peaks=True,
                                                                                      search_kwargs=search_kwargs,
                                                                                      )
            # a, b = naff.fft(self.df.iloc[0].loc[bpm])
            freq[bpm] = pf
            pwr[bpm] = pp
            peaks[bpm] = (peak_tunes, peak_props)
            if selector:
                nu = selector(self, peaks[bpm], bpm)
                self.df['nu_' + bpm] = nu
            else:
                raise
            average_tunes[bpm[-1]].append(nu)
        # self.fft_freq = freq
        # self.fft_pwr = pwr
        self.peaks = peaks
        if 'H' in families:
            self.nux = np.mean(average_tunes['H'])
            self.df['nux'] = self.nux
            self.df['sig_nux'] = np.std(average_tunes['H'])
        if 'V' in families:
            self.nuy = np.mean(average_tunes['V'])
            self.df['nuy'] = self.nuy
            self.df['sig_nuy'] = np.std(average_tunes['V'])
        return freq, pwr, peaks

    def calculate_fft(self, naff: NAFF, families: List[str] = None):
        """
        Calculates FFT for each bpms and stores in dataframe
        :param families:
        :param naff:
        :return:
        """
        families = families or ['H', 'V', 'C']
        bpms = self.get_bpms(families, soft_fail=True)
        for i, bpm in enumerate(bpms):
            fft_freq, fft_power = naff.fft(self.df.iloc[0].loc[bpm])
            self.df['fft_freq_' + bpm] = [fft_freq]
            self.df['fft_pwr_' + bpm] = [fft_power]

    def calculate_sum_signal(self) -> float:
        """
        Calculate mean of all sum signals and place onto dataframe
        :return:
        """
        avg = 0
        bpms = self.get_bpms('S')
        for bpm in bpms:
            avg += np.mean(self.df.iloc[0].loc[bpm])

        self.df['intensity'] = avg / len(bpms)
        if np.isnan(self.df.iloc[0]['intensity']):
            raise Exception
        return avg / len(bpms)

    def calculate_stats(self) -> dict:
        """
        Calculates mean and variance of BPM signals and places them into the dataframe
        :return:
        """
        stats = {}
        bpms = self.get_bpms('A')

        averages = np.zeros(len(bpms))
        variances = np.zeros(len(bpms))
        for i, bpm in enumerate(bpms):
            data = self.get_bpm_data(bpms=[bpm], return_type='list')[0]
            # print(data)
            # data = self.df.iloc[0].loc[bpm]
            mean, std = np.mean(data), np.std(data)
            averages[i] = mean
            variances[i] = std
            stats[bpm] = (mean, std)
            if np.isnan(mean) or np.isnan(std):
                raise Exception(f'NaN encountered computing stats in kick ({self.idx})')
        columns = ['avg_' + bpm for bpm in bpms] + ['sig_' + bpm for bpm in bpms]
        columns_extra = [c for c in columns if c not in self.df.columns]
        if len(columns) != len(columns_extra):
            if len(columns_extra) != 0:
                raise Exception(
                    f'Columns are mixed up - they should all exist or all not exist, but have ({len(columns)}) vs ({len(columns_extra)})! ({columns})({columns_extra})')
        if columns_extra:
            # print(f'Adding extra cols: {columns_extra}')
            # print(np.hstack([averages, variances]).shape)
            df_temp = pd.DataFrame(columns=columns_extra,
                                   index=[0],
                                   data=np.hstack([averages, variances])[np.newaxis, :])
            # print(df_temp)
            self.df = pd.concat([self.df, df_temp], axis=1)
        else:
            print(len(columns), np.hstack([averages, variances])[np.newaxis, :].shape, self.df.loc[0, columns])
            self.df.loc[0, columns] = np.hstack([averages, variances])[np.newaxis, :]

        return stats


class KickSequence:
    BPMS_ACTIVE = []

    def __init__(self, kicks: list, demean=True):
        self.kicks = kicks
        # dflist = [k.df for k in self.kicks]
        # self.df = pd.concat(dflist)
        self.df = None
        self.update_df()
        # self.BPMS_GOOD = bpm_list.copy()
        # self.HG = self.BPMS_HG = ["N:" + i + "H" for i in bpm_list]
        # self.VG = self.BPMS_VG = ["N:" + i + "V" for i in bpm_list]
        # self.SG = self.BPMS_SG = ["N:" + i + "S" for i in bpm_list]
        # self.BPMS_ALLG = self.BPMS_HG + self.BPMS_VG + self.BPMS_SG
        # self.bpm_families = {'H': self.BPMS_HG, 'V': self.BPMS_VG, 'S': self.BPMS_SG}
        self.kicksV = self.df.loc[:, 'kickv']
        self.kicksH = self.df.loc[:, 'kickh']
        self.naff = None

        # demean data
        # if demean:
        #     df.loc[:, self.BPMS_HG + self.BPMS_VG] = df.loc[:, self.BPMS_HG + self.BPMS_VG].applymap(
        #         lambda x: x - np.mean(x))

    def __len__(self):
        return len(self.df)

    def check_dataset_integrity(self, include_octupoles: bool = True, include_nl: bool = True):
        """
        Checks if key state parameters are the same for all kicks, and that none are contained
        """
        invariant_devices = iota.DIPOLES.ALL_I + \
                            iota.CORRECTORS.ALL + \
                            iota.QUADS.ALL_CURRENTS + \
                            iota.SKEWQUADS.ALL_CURRENTS + \
                            iota.SEXTUPOLES.ALL_CURRENTS + \
                            ['N:IRFLLA', 'N:IRFMOD', 'N:IRFEAT', 'N:IRFEPC'] + \
                            ['N:IKPSVX', 'N:IKPSVD']
        if include_octupoles:
            invariant_devices += iota.OCTUPOLES.ALL_CURRENTS
        if include_nl:
            invariant_devices += iota.DNMAGNET.ALL_CURRENTS
        invariant_devices = set(invariant_devices)
        kicks = self.kicks
        #
        states = [k.get_optics() for k in kicks]
        assert all(x == states[0] for x in states)
        states = [k.get_correctors() for k in kicks]
        assert all(x == states[0] for x in states)
        states = [k.get_sextupoles() for k in kicks]
        assert all(x == states[0] for x in states)

        times = [k.state('aq_timestamp') for k in kicks]
        assert all(np.diff(times) > 0)
        times = [k.state('N:EA5TRG.READING') for k in kicks]
        assert all(np.diff(times) > 0)

        states = [k.get_full_state() for k in kicks]
        keys = [set(s.keys()) for s in states]
        shared_keys = set.intersection(*keys)
        assert all(len(shared_keys) == len(k) for k in keys)
        kvtuples = [set(s.items()) for s in states]
        shared_keys = set.intersection(*kvtuples)
        outliers = set()
        for kvt in kvtuples:
            outliers.update(kvt - shared_keys)
        outlier_keys = set()
        for e in outliers:
            outlier_keys.update((e[0],))
        outlier_devs = set([x.split('.')[0] for x in outlier_keys])
        if len(invariant_devices.intersection(outlier_devs)) != 0:
            raise Exception(f'Found invariants in outlier devices: {invariant_devices.intersection(outlier_devs)}')
        return outlier_devs

    # Deprecated
    def purge_kicks(self, min_intensity: float, max_intensity: float):
        """
        Remove kicks with average intensity outside of specified limits
        :param min_intensity:
        :param max_intensity:
        """
        bad_kicks = []
        for k in self.kicks:
            intensity = k.calculate_sum_signal()
            if not min_intensity < intensity < max_intensity:
                print(f'Removed kick {k.idx} - intensity {intensity} outside allowed limits')
                bad_kicks.append(k)
        for k in bad_kicks:
            self.kicks.remove(k)
        self.update_df()

    def remove_kicks(self, kick_ids: List[Union[int, Kick]]):
        """
        Remove specified kicks from sequence
        :param kick_ids: Kick id or object itself
        """
        for kid in kick_ids:
            if isinstance(kid, int):
                try:
                    k = self.get_kick(kid)
                except:
                    continue
            elif isinstance(kid, Kick):
                k = kid
            else:
                raise Exception
            self.kicks.remove(k)
        self.update_df()

    def get_kick(self, kick_id: int):
        """
        Gets kick object with specified id
        :param kick_id:
        :return: Kick object
        """
        if kick_id not in self.df.idx.values:
            raise Exception(f'Kick id {kick_id} not in sequence')
        df_row = self.df.loc[self.df.idx == kick_id]
        if df_row.shape[0] != 1:
            raise Exception(f'Kick id {kick_id} did not yield unique kick - result df is {df_row.shape}')
        else:
            return df_row.iloc[0, self.df.columns.get_loc('kick')]

    # def get_kick(self, idx):
    #    return Kick(self.df.loc[idx, :], kick_id=idx, parent_sequence=self)

    def update_df(self):
        dflist = [k.df for k in self.kicks]
        self.df = pd.concat(dflist).sort_values(['kickv', 'kickh'])

    def calculate_sum_signal(self):
        for k in self.kicks:
            k.calculate_sum_signal()
        self.update_df()

    def calculate_fft(self, naff: NAFF, families: List[str] = None):
        naff = naff or self.naff
        families = families or ['H', 'V', 'C']
        for r in self.df.itertuples():
            r.kick.calculate_fft(naff, families=families)

    def calculate_tune(self,
                       naff: NAFF,
                       families: List[str] = None,
                       selector: Callable = None,
                       search_kwargs: Dict[str, int] = None):
        naff = naff or self.naff
        assert selector
        for r in self.df.itertuples():
            r.kick.calculate_tune(naff, families, selector, search_kwargs=search_kwargs)

    def get_kick_magnitudes(self):
        return self.df.loc[:, 'kickV'].values

    def get_bpm_datadict(self, idx, family, bpmlist=None):
        datadict = {}
        if bpmlist is None:
            if family in ['H', 'V', 'S']:
                bpms = self.bpm_families[family]
                for b in bpms:
                    # datadict[b] = df.loc[idx, b].values[0]
                    datadict[b] = self.df.loc[idx, b]
            else:
                raise Exception
        else:
            raise Exception
        return datadict

    def get_bpm_matrix(self, idx, family, bpm_list=None, scaled=False):
        """
        Returns a matrix containing all data as np.ndarray of shape [NP x NBPM]

        Args:
            idx: kick number in the sequence
            family: the desired channel (H,V,S)
            bpm_list: optional bpm list - all data is returned if not provided
            scaled: optional, whether to normalize signals by magnitude
        """
        if bpm_list is None:
            if family in ['H', 'V', 'S']:
                bpms = self.bpm_families[family]
                data = np.vstack(self.df.loc[idx, bpms])
                if scaled:
                    data = data / np.mean(np.abs(data), axis=1)[:, np.newaxis]
            else:
                raise Exception
        else:
            raise Exception("This is not yet supported")
        return data

    def get_bpm_singlearray(self, idx, family, bpmlist=None):
        data = []
        if bpmlist is None:
            if family in ['H', 'V', 'S']:
                bpms = self.bpm_families[family]
                data = np.concatenate(self.df.loc[idx, bpms])
            else:
                raise Exception
        else:
            raise Exception
        return data
