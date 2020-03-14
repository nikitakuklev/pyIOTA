from typing import Union

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

        self.HG = self.BPMS_HG = [i + "H" for i in bpm_list]
        self.VG = self.BPMS_VG = [i + "V" for i in bpm_list]
        self.SG = self.BPMS_SG = [i + "S" for i in bpm_list]
        self.BPMS_ALLG = self.BPMS_HG + self.BPMS_VG + self.BPMS_SG
        self.bpm_families = {'H': self.BPMS_HG, 'V': self.BPMS_VG, 'S': self.BPMS_SG}

        self.df = df
        self.idx = kick_id
        if df.iloc[0].loc['idx'] == 0:
            df.loc[0, 'idx'] = kick_id
        self.collection_id = df.iloc[0].loc['idx']
        self.ks = parent_sequence
        self.nux = None  # main tune
        self.nuy = None  # main tune
        self.matrix_cache = {}
        self.n_turns = self.get_turns()
        self.fft_pwr = self.fft_freq = self.peaks = None

        # print('Read in kick')
        # Determine which BPMs are behaving ok using a rolling window average

    def copy(self):
        df2 = self.df.copy(deep=True)
        return Kick(df=df2, kick_id=self.idx)



    def determine_active_bpms(cutoff=0.05):

        pass

    def compute_tunes_naff(self):
        pass



    def get_bpm_matrix(self, family: str = 'V', remove_sequence_data: bool = False):
        if family in ['H', 'V', 'S']:
            if family in self.matrix_cache:
                data = self.matrix_cache[family]
            else:
                bpms = self.get_bpms(family)
                data = np.vstack(self.df.iloc[0].loc[bpms])
                self.matrix_cache[family] = data
            if remove_sequence_data:
                return data[:, 1:]
            else:
                return data
        else:
            raise Exception("Invalid family specified")

    def as_dict(self, family: str = 'A', bpmlist: list = None, trim: tuple = (1, -1)):
        datadict = {}
        if bpmlist is None:
            bpms = self.get_bpms(family)
            for b in bpms:
                # datadict[b] = df.loc[idx, b].values[0]
                if trim:
                    datadict[b] = self.df.iloc[0].loc[b][trim[0]:trim[1]]
                else:
                    datadict[b] = self.df.iloc[0].loc[b]
        else:
            raise Exception(f'BPM list not supported yet')
        return datadict

    def compute_tune_fft(self, naff: NAFF, selector=None):
        bpms = self.get_bpms(['H', 'V'])
        freq = {}
        pwr = {}
        peaks = {}
        average_tunes = {'H': [], 'V': []}
        for i, bpm in enumerate(bpms):
            top_tune, peak_tunes, peak_idx, peak_props, (pf, pp) = naff.fft_peaks(self.df.iloc[0].loc[bpm],
                                                                                  search_peaks=True)
            # a, b = naff.fft(self.df.iloc[0].loc[bpm])
            freq[bpm] = pf
            pwr[bpm] = pp
            peaks[bpm] = (peak_tunes, peak_props)
            if selector:
                nu = selector(self, peaks[bpm], bpm[-1])
                self.df['nu_' + bpm] = nu
            else:
                raise
            average_tunes[bpm[-1]].append(nu)
        self.fft_freq = freq
        self.fft_pwr = pwr
        self.peaks = peaks
        self.nux = np.mean(average_tunes['H'])
        self.nuy = np.mean(average_tunes['V'])
        self.df['nux'] = self.nux
        self.df['nuy'] = self.nuy
        return freq, pwr, peaks, self.nux, self.nuy

    def get_tune_data(self, bpm: str):
        return self.fft_freq[bpm], self.fft_pwr[bpm], self.peaks[bpm]

    def get_bpms(self, family: Union[list, str] = 'A'):
        bpms = []
        if isinstance(family, str):
            family = [family]
        for fam in family:
            if fam in ['H', 'V', 'S']:
                bpms += self.bpm_families[fam]
            elif fam == 'A':
                bpms += self.BPMS_ALLG
            else:
                raise Exception
        if len(bpms) == 0:
            raise Exception(f'No BPMs found for families: {family}')
        return list(set(bpms))

    def get_turns(self):
        bpm = self.get_bpms()[0]
        return len(self.df.iloc[0].loc[bpm])

    def calculate_sum_signal(self):
        avg = 0
        bpms = self.get_bpms('S')
        for bpm in bpms:
            avg += np.mean(self.df.iloc[0].loc[bpm])

        self.df['intensity'] = avg / len(bpms)
        if np.isnan(self.df.iloc[0]['intensity']):
            raise Exception
        return avg / len(bpms)


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

    #     def sort(self, col):
    #         self.df = self.df.sort_values(col)

    def calculate_sum_signal(self):
        for k in self.kicks:
            k.calculate_sum_signal()
        self.update_df()

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

    def compute_tunes_fft(self, naff: NAFF, selector):
        naff = naff or self.naff
        for r in self.df.itertuples():
            r.kick.compute_tune_fft(naff, selector)

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
