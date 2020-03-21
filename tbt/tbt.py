from typing import Union, Callable, Dict, List

import numpy as np
import pyIOTA.acnet.utils
import pyIOTA.iota.run2
import scipy as sc
import pandas as pd
from scipy.signal import hilbert, chirp, butter, filtfilt

# special_keys = ['idx', 'kickv', 'kickh', 'state', 'custom']
from pyIOTA.tbt.naff import NAFF

special_keys = pyIOTA.acnet.utils.special_keys.copy()

critical_keys = ['kickv', 'kickh', 'idx']


def FilteredSignalButter(signal, fs, cutoff):
    B, A = butter(1, cutoff / (fs / 2), btype='low')
    filtered_signal = filtfilt(B, A, signal, axis=0)
    return filtered_signal


class Kick:
    def __init__(self, df: pd.DataFrame, kick_id: int = None, bpm_list: list = None, parent_sequence=None):
        for ck in critical_keys:
            if ck not in df.columns:
                raise Exception(f'Missing critical key ({ck}')

        if not bpm_list:
            bpm_list = set([k[:-1] for k in pyIOTA.iota.run2.BPMS.ALLA if k not in special_keys])
            # bpm_list = set([k[:-1] for k in df.columns if k not in special_keys])
            # print(f'BPM list not specified - deducing {len(bpm_list)}: {bpm_list}')

        self.HG = [i + "H" for i in bpm_list]
        self.VG = [i + "V" for i in bpm_list]
        self.SG = [i + "S" for i in bpm_list]
        self.ALLG = self.HG + self.VG + self.SG
        self.CG = self.BPMS_CG = []  # Calculated
        self.bpm_families = {'H': self.HG, 'V': self.VG, 'S': self.SG, 'C': self.BPMS_CG}

        self.df = df
        self.idx = kick_id
        # if df.iloc[0].loc['idx'] == 0:
        df.loc[0, 'idx'] = kick_id
        self.collection_id = df.iloc[0].loc['idx']
        self.ks = parent_sequence
        self.nux = None  # main tune
        self.nuy = None  # main tune
        self.matrix_cache = {}
        self.n_turns = self.get_turns()
        self.fft_pwr = self.fft_freq = self.peaks = None
        self.v = self.kickv = self.df.iloc[0, self.df.columns.get_loc('kickv')]
        self.h = self.kickh = self.df.iloc[0, self.df.columns.get_loc('kickh')]
        # print('Read in kick')
        # Determine which BPMs are behaving ok using a rolling window average

    def copy(self) -> 'Kick':
        df2 = self.df.copy(deep=True)
        return Kick(df=df2, kick_id=self.idx)

    def col(self, column: str):
        return self.df.iloc[0, self.df.columns.get_loc(column)]

    def col_fft(self, bpm: str):
        return self.col('fft_freq_'+bpm), self.col('fft_pwr_'+bpm)

    def state(self, param: str):
        return self.df.iloc[0, self.df.columns.get_loc('state')][param]

    def determine_active_bpms(self, cutoff=0.05):
        pass

    def disable_bpm(self, bpm, plane: str = 'A'):
        if plane == 'A':
            rm_cnt = 0
            for sp, bl in zip(['H', 'V', 'S'], [self.HG, self.VG, self.SG]):
                if bpm + sp in bl:
                    bl.remove(bpm + sp)
                    rm_cnt += 1
            if rm_cnt != 3 or rm_cnt != 0:
                raise Exception(f'BPM {bpm} only got removed from {rm_cnt} lists, not 3 or 0')
            self.ALLG = self.HG + self.VG + self.SG

        else:
            raise Exception(f'Removing BPM {bpm} in only 1 plane is not supported')

    def compute_tunes_naff(self):
        pass

    def get_bpm_matrix(self, family: str = 'V', remove_sequence_data: bool = False) -> np.ndarray:
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

    def as_dict(self, family: str = 'A', bpmlist: list = None, trim: tuple = (1, -1)) -> dict:
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

    def get_tune_data(self, bpm: str):
        return self.fft_freq[bpm], self.fft_pwr[bpm], self.peaks[bpm]

    def get_bpms(self, family: Union[list, str] = 'A') -> List[str]:
        bpms = []
        if isinstance(family, str):
            family = [family]
        for fam in family:
            if fam in ['H', 'V', 'S']:
                bpms += self.bpm_families[fam]
            elif fam == 'A':
                bpms += self.ALLG
            else:
                raise Exception
        if len(bpms) == 0:
            raise Exception(f'No BPMs found for families: {family}')
        return list(set(bpms))

    def get_turns(self) -> int:
        bpm = self.get_bpms()[0]
        return len(self.df.iloc[0].loc[bpm])

    def calculate_tune(self, naff: NAFF, selector: Callable = None, search_kwargs: Dict[str, int] = None,
                       use_precalculated: bool = True):
        """
        Calculates tune by finding peaks in FFT data, optionally using precomputed data to save time
        :param naff:
        :param selector:
        :param search_kwargs:
        :param use_precalculated:
        :return:
        """
        bpms = self.get_bpms(['H', 'V'])
        freq = {}
        pwr = {}
        peaks = {}
        average_tunes = {'H': [], 'V': []}
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
        #self.fft_freq = freq
        #self.fft_pwr = pwr
        self.peaks = peaks
        self.nux = np.mean(average_tunes['H'])
        self.nuy = np.mean(average_tunes['V'])
        self.df['nux'] = self.nux
        self.df['sig_nux'] = np.std(average_tunes['H'])
        self.df['nuy'] = self.nuy
        self.df['sig_nuy'] = np.std(average_tunes['V'])
        return freq, pwr, peaks, self.nux, self.nuy

    def calculate_fft(self, naff: NAFF):
        """
        Calculates FFT for each bpms and stores in dataframe
        :param naff:
        :return:
        """
        bpms = self.get_bpms(['H', 'V'])
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
            data = self.df.iloc[0].loc[bpm]
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


    def purge_kicks(self, min_intensity, max_intensity):
        bad_kicks = []
        for k in self.kicks:
            intensity = k.calculate_sum_signal()
            if not min_intensity < intensity < max_intensity:
                print(f'Removed kick {k.idx} - intensity {intensity} outside allowed limits')
                bad_kicks.append(k)
        for k in bad_kicks:
            self.kicks.remove(k)
        self.update_df()

    def update_df(self):
        dflist = [k.df for k in self.kicks]
        self.df = pd.concat(dflist).sort_values(['kickv', 'kickh'])

    def calculate_sum_signal(self):
        for k in self.kicks:
            k.calculate_sum_signal()
        self.update_df()

    def calculate_fft(self, naff: NAFF):
        naff = naff or self.naff
        for r in self.df.itertuples():
            r.kick.calculate_fft(naff)

    def calculate_tune(self, naff: NAFF, selector: Callable, search_kwargs: Dict[str, int]):
        naff = naff or self.naff
        for r in self.df.itertuples():
            r.kick.calculate_tune(naff, selector, search_kwargs=search_kwargs)

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

    def get_kick(self, idx):
        return Kick(self.df.loc[idx, :], kick_id=idx, parent_sequence=self)

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
