import numpy as np
import pyIOTA.acnet.utils
import scipy as sc
import pandas as pd
from scipy.signal import hilbert, chirp, butter, filtfilt

# special_keys = ['idx', 'kickv', 'kickh', 'state', 'custom']
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
            bpm_list = set([k[:-1] for k in df.columns if k not in special_keys])
            print(f'BPM list not specified - deducing {len(bpm_list)}: {bpm_list}')

        self.HG = self.BPMS_HG = [i + "H" for i in bpm_list]
        self.VG = self.BPMS_VG = [i + "V" for i in bpm_list]
        self.SG = self.BPMS_SG = [i + "S" for i in bpm_list]
        self.BPMS_ALLG = self.BPMS_HG + self.BPMS_VG + self.BPMS_SG
        self.bpm_families = {'H': self.BPMS_HG, 'V': self.BPMS_VG, 'S': self.BPMS_SG}

        self.df = df
        self.idx = kick_id
        self.collection_id = df.iloc[0].loc['idx']
        self.ks = parent_sequence
        self.nux = None  # main tune
        self.nuy = None  # main tune
        self.matrix_cache = {}

        # print('Read in kick')
        # Determine which BPMs are behaving ok using a rolling window average

    def determine_active_bpms(cutoff=0.05):

        pass

    def compute_tunes_naff(self):
        pass

    def get_bpm_matrix(self, family: str = 'V', remove_sequence_data: bool = True):
        if family in ['H', 'V', 'S']:
            if family in self.matrix_cache:
                data = self.matrix_cache[family]
            else:
                bpms = self.bpm_families[family]
                data = np.vstack(self.df.iloc[0].loc[bpms])
                self.matrix_cache[family] = data
            if remove_sequence_data:
                return data[:, 1:]
            else:
                return data
        else:
            raise Exception("Invalid family specified")

    def as_dict(self, family: str = 'V', bpmlist: list = None):
        datadict = {}
        if bpmlist is None:
            if family in ['H', 'V', 'S']:
                bpms = self.bpm_families[family]
                for b in bpms:
                    # datadict[b] = df.loc[idx, b].values[0]
                    datadict[b] = self.df.iloc[0].loc[b]
            else:
                raise Exception
        else:
            raise Exception(f'BPM list not supported yet')
        return datadict


class KickSequence:
    BPMS_ACTIVE = []

    def __init__(self, df, bpm_list, demean=True):
        self.df = df
        self.BPMS_GOOD = bpm_list.copy()
        self.HG = self.BPMS_HG = ["N:" + i + "H" for i in bpm_list]
        self.VG = self.BPMS_VG = ["N:" + i + "V" for i in bpm_list]
        self.SG = self.BPMS_SG = ["N:" + i + "S" for i in bpm_list]
        self.BPMS_ALLG = self.BPMS_HG + self.BPMS_VG + self.BPMS_SG
        self.bpm_families = {'H': self.BPMS_HG, 'V': self.BPMS_VG, 'S': self.BPMS_SG}
        self.kicksV = df.loc[:, 'kickV']
        self.kicksH = df.loc[:, 'kickH']

        # demean data
        if demean:
            df.loc[:, self.BPMS_HG + self.BPMS_VG] = df.loc[:, self.BPMS_HG + self.BPMS_VG].applymap(
                lambda x: x - np.mean(x))

    def __len__(self):
        return len(self.df)

    #     def sort(self, col):
    #         self.df = self.df.sort_values(col)

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
