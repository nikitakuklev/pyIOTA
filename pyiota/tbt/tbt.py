import copy
import enum
import itertools
import json
import logging
from json import JSONDecodeError
from typing import Union, Callable, Dict, List, Iterable, Tuple, Optional, Any, Set, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..iota import run2 as iota
from ..acnet import utils as acutils

# special_keys = ['idx', 'kickv', 'kickh', 'state', 'custom']

from ocelot import Twiss, Monitor
# from pyIOTA.tbt.naff import NAFF
from .naff import NAFF

logger = logging.getLogger(__name__)

special_keys = acutils.special_keys.copy()

critical_keys = ['kickv', 'kickh', 'idx']

if TYPE_CHECKING:
    from ..lattice import LatticeContainer
# def FilteredSignalButter(signal, fs, cutoff):
#     B, A = butter(1, cutoff / (fs / 2), btype='low')
#     filtered_signal = filtfilt(B, A, signal, axis=0)
#     return filtered_signal

class Util:
    @staticmethod
    def check_folders(folders):
        """ Check folders with kick data into lists """
        files_ll = []
        props_dicts = []

        def parse(p, soft_fail: bool = False):
            if p.exists():
                with p.open('r') as f:
                    try:
                        d = json.load(f)
                        return d
                    except JSONDecodeError as e:
                        logger.error(f'JSON file {p} failed to parse')
                        if not soft_fail:
                            raise e
            return {}

        for folder in folders:
            # Return in name sorted order - should be time ordered too since all names have timestamp
            fs = sorted(list(folder.glob('*.hdf5')))
            files_ll.append(fs)
            # Options for analysis
            props_fpath = folder / 'analysis_props.json'
            props_fpath2 = folder / 'analysis_props_manual.json'

            props = parse(props_fpath)
            props2 = parse(props_fpath2)
            if props or props2:
                # Manual overrides auto generated values
                logger.info(
                    f'{folder} - {len(fs)} files (cfg LOADED - {len(props)} gen + {len(props2)} manual keys)')
                props.update(props2)
            else:
                # No config
                logger.info(f'{folder} - {len(fs)} files')
            props_dicts.append(props)
        return files_ll, props_dicts

    class Ocelot:
        @staticmethod
        def twiss_to_dataframe(tws: List[Twiss]):
            """ Convert list of twiss objects from Ocelot to a TWISS dataframe """
            keys = ['beta_x', 'beta_y', 'alpha_x', 'alpha_y', 'Dx', 'Dy']
            index = [tw.id for tw in tws]
            data = {}
            for tw in tws:
                data[tw.id] = [getattr(tw, k) for k in keys]
            return pd.DataFrame(index=index, data=data)


class COORD(enum.Enum):
    X = 'x'
    PX = 'px'
    Y = 'y'
    PY = 'py'


class TrackFrame(pd.DataFrame):
    """ Thin container to store TBT data """

    @property
    def df(self):
        return self

    def mat(self, idx, coord):
        assert coord in COORD
        return self.as_matrix(idx, self.columns.str.endswith(coord))

    def as_matrix(self, idx, mask):
        return np.vstack(self.loc[idx, mask])

    def xmat(self, idx=0):
        return self.as_matrix(idx, self.columns.str.endswith(COORD.X.value))

    def pxmat(self, idx=0):
        return self.as_matrix(idx, self.columns.str.endswith(COORD.PX.value))

    def ymat(self, idx=0):
        return self.as_matrix(idx, self.columns.str.endswith(COORD.Y.value))

    def pymat(self, idx=0):
        return self.as_matrix(idx, self.columns.str.endswith(COORD.PY.value))


class SimKick:
    """
    SimKick stores data of a particle tracking simulation, along with related metadata.
    Dataframe format has particles as rows, with only mandatory column 'N' (the number of turns)
    Typical data columns are 'track' sub-dataframe and 'twiss' dictionary
    """

    @enum.unique
    class Datatype(enum.Enum):
        """
        Enum containing the most common types of data in a kick. Intended to be used with data retrieval methods.
        """
        RAW = ''
        FFT_FREQ = '_fft_freq'
        FFT_POWER = '_fft_pwr'
        NUX = '_nux'
        NUY = '_nuy'
        NU = '_nu'
        INTERPX = '_ix'
        INTERPY = '_iy'
        CSX = 'CSx'
        CSY = 'CSy'
        I1 = 'I1'
        stdevCSX = 'stdevCSx'
        stdevCSY = 'stdevCSy'
        stdevI1 = 'stdevI1'

    @enum.unique
    class Dataclass(enum.Enum):
        # TODO: try subclass str like IntEnum?
        """
        Enum for data classes (tracking, twiss, etc.)
        """
        TRACK = 'track'
        TURN_CNT = 'N'
        TWISS = 'twiss'

        @property
        def v(self):
            """ Shorthand method """
            return self.value

    @staticmethod
    def convert_tbt_matrices_to_track_df(bpms: Iterable[str],
                                         mx: np.ndarray, mpx: np.ndarray, my: np.ndarray,
                                         mpy: np.ndarray,
                                         families: Iterable[str] = None
                                         ):
        """
        Convert data matrices from M bpms x N turns format into a track dataframe of single particle
        """
        DT, DC = SimKick.Datatype, SimKick.Dataclass
        families = families or ('x', 'xp', 'y', 'yp')
        assert mx.shape == my.shape == mpx.shape == mpy.shape
        assert mx.shape[0] == len(bpms)

        matrices = (mx, mpx, my, mpy)
        # This is per-plane version, not needed
        # for f, mat in zip(families, matrices):
        #     data_track = {}
        #     for i, b in enumerate(bpms):
        #         data_track[b] = [mat[i,:]]
        #     dft = pd.DataFrame(columns=bpms, index=[0], data=data_track)
        #     data[DC.TRACK.v + '_' + f] = [dft]
        # columns = [DC.TURN_CNT.v] + [DC.TRACK.v + '_' + f for f in families]
        data_track = {DC.TURN_CNT.v: mx.shape[1]}
        for f, mat in zip(families, matrices):
            for i, b in enumerate(bpms):
                data_track[b + '_' + f] = [mat[i, :]]
        dft = TrackFrame(index=[0], data=data_track)

        data = {DC.TURN_CNT.v: mx.shape[1], DC.TRACK.v: [dft]}
        df = pd.DataFrame(index=[0], data=data)
        return df

    def __init__(self,
                 df: pd.DataFrame,
                 families: Tuple = None
                 ):
        """
        Stores data from a tracking simulation. Many methods are analogous to experimental Kick class.
        :param df: master dataframe
        """
        self.df = df
        self.families = families or ('x', 'px', 'y', 'py')
        self.pairs = (('x', 'px'), ('y', 'py'))
        self.has_track = self.Dataclass.TRACK.v in df.columns
        self.has_twiss = self.Dataclass.TWISS.v in df.columns
        # if self.track:
        #    self.tdata = self.df.iloc[0, self.df.columns.get_loc('track')].df
        self.N = self.tdata.N.max()
        self.n_part = len(self.tdata) if self.has_track else None

    @property
    def tdata(self):
        if self.has_track:
            return self.df.iloc[0, self.df.columns.get_loc(self.Dataclass.TRACK.v)].df
        else:
            return None

    @tdata.setter
    def tdata(self, value):
        self.df.iloc[0, self.df.columns.get_loc(self.Dataclass.TRACK.v)].df = value

    @property
    def track(self):
        return self.tdata

    def rotate(self, bpm):
        """
        During analysis it is important to rotate starting point to just past main nonlinearity
        This method does so, and adjusts phases appropriately
        :param bpm: New starting bpm
        """
        dnew = np.vstack([data_matrix[idx:, :-1], data_matrix[:idx, 1:]])
        bpms_root = bpms_root[idx:] + bpms_root[:idx]
        return dnew, bpms_root

    def prune_lost_particles(self):
        self.tdata = self.tdata[self.tdata.N == self.N].copy()
        self.n_part = len(self.tdata)

    def calculate_fft(self,
                      naff: NAFF,
                      families: List = None,
                      spacing: float = 1.0,
                      data_trim: slice = None,
                      store: bool = True
                      ):
        """ Calculates FFT for each particle and store results """
        assert self.has_track
        data = self.tdata
        families = families or self.families
        results = []
        for f in families:
            fft_freq_l, fft_power_l = [], []
            for i in data.index:
                vec = data.loc[i, f].copy() - np.mean(data.loc[i, f])
                if data_trim:
                    # Use provided trims
                    fft_freq, fft_power = naff.fft(vec[data_trim],
                                                   data_trim=np.s_[:],
                                                   spacing=spacing)
                else:
                    # Use NAFF trims
                    fft_freq, fft_power = naff.fft(vec,
                                                   spacing=spacing)
                fft_freq_l.append(fft_freq)
                fft_power_l.append(fft_power)

            if store:
                ser_freq = pd.Series(fft_freq_l, index=data.index)
                ser_power = pd.Series(fft_power_l, index=data.index)
                data.loc[:, f + self.Datatype.FFT_FREQ.value] = ser_freq
                data.loc[:, f + self.Datatype.FFT_POWER.value] = ser_power
            results.append((fft_freq_l, fft_power_l))
        return results

    def calculate_tunes(self,
                        naff: NAFF,
                        method: str = 'FFT',
                        selector: Callable = None,
                        search_kwargs: Dict[str, int] = None,
                        use_precalculated: bool = True,
                        data_trim: slice = None,
                        pairs: bool = False
                        ):
        """
        Calculates tune by finding peaks in FFT data, optionally using precomputed data to save time

        :param naff: tbt.NAFF object to be used - its setting will take priority over kick values
        :param method: Peak finding method - NAFF or FFT
        :param families: Families to perform calculation on - typically H, V, or C
        :param selector: Function that picks correct peak from list
        :param search_kwargs: Method specific extra parameters to be used in the search
        :param use_precalculated:
        :param data_trim: Trim override - if not provided, use whatever NAFF object has
        :param pairs:
        :return:
        """
        data = self.tdata
        if not pairs:
            for f in self.families:
                if method.upper() == 'FFT':
                    tunes = []
                    col_fr = f + self.Datatype.FFT_FREQ.value
                    col_pwr = f + self.Datatype.FFT_POWER.value

                    if use_precalculated and col_fr in data and col_pwr in data:
                        fft_freq_l = data.loc[:, col_fr]
                        fft_power_l = data.loc[:, col_pwr]
                    else:
                        fft_freq_l, fft_power_l = \
                        self.calculate_fft(naff, [f], data_trim=data_trim, store=False)[0]

                    for fft_freq, fft_power in zip(fft_freq_l, fft_power_l):
                        tunes.append(fft_freq[np.argmax(fft_power)])

                    ser_tunes = pd.Series(tunes, index=data.index)
                    data.loc[:, f + self.Datatype.NU.value] = ser_tunes
                else:
                    raise Exception

    def calculate_invariants(self, beta_x, alpha_x, beta_y, alpha_y,
                             i1_str=0.0,
                             data_trim: slice = None
                             ):
        from .optics import Coordinates
        from .optics import Invariants
        assert self.has_track
        data_trim = data_trim or np.s_[:]
        data = self.df.iloc[0, self.df.columns.get_loc('track')].df

        csxl = []
        csyl = []
        i1l = []
        for i in range(1, self.n_part + 1):
            x = data.loc[i, self.pairs[0][0]][data_trim]
            xp = data.loc[i, self.pairs[0][1]][data_trim]
            xn, xpn = Coordinates.normalize(x, xp, beta_x, alpha_x)
            csx = Invariants.compute_CS_2D(xn, xpn, True)

            y = data.loc[i, self.pairs[1][0]][data_trim]
            yp = data.loc[i, self.pairs[1][1]][data_trim]
            yn, ypn = Coordinates.normalize(y, yp, beta_y, alpha_y)
            csy = Invariants.compute_CS_2D(yn, ypn, True)

            i1 = Invariants.compute_I1(xn, xpn, yn, ypn, i1_str, c=None, normalized=True)

            csxl.append(csx)
            csyl.append(csy)
            i1l.append(i1)

        data[self.Datatype.CSX.value] = csxl
        data[self.Datatype.CSY.value] = csyl
        data[self.Datatype.I1.value] = i1l

    def calculate_invariants_jitter(self):
        assert self.has_track
        data = self.df.iloc[0, self.df.columns.get_loc('track')].df

        data[self.Datatype.stdevCSX.value] = [np.std(x) / np.mean(x) for x in
                                              data[self.Datatype.CSX.value]]
        data[self.Datatype.stdevCSY.value] = [np.std(x) / np.mean(x) for x in
                                              data[self.Datatype.CSY.value]]
        data[self.Datatype.stdevI1.value] = [np.std(x) / np.mean(x) for x in
                                             data[self.Datatype.I1.value]]


class Kick:
    @enum.unique
    class Datatype(enum.Enum):
        """
        Enum containing the most common types of data in a kick. Intended to be used with data retrieval methods.
        """
        RAW = ''
        ORIG = '_orig'
        FFT_FREQ = '_fft_freq'
        FFT_POWER = '_fft_pwr'
        NUX = '_nux'
        NUY = '_nuy'
        NU = '_nu'
        INTERPX = '_ix'
        INTERPY = '_iy'
        AMP = '_a'
        AMPSIG = '_asig'

    def __init__(self,
                 df: pd.DataFrame,
                 kick_id: int = -1,
                 id_offset: int = 0,
                 bpm_list: Optional[list[str]] = None,
                 parent_sequence: Optional['KickSequence'] = None,
                 file_name: str = None,
                 trim: Tuple = None,
                 iota_defaults: bool = False
                 ):
        """
        Represents a single kick (continuous TBT time series) measured by multiple BPMs

        :param df: DataFrame row with all data
        :param kick_id: Integer kick id, will be used to sort
        :param bpm_list: Optional sequence of good BPMs
        :param parent_sequence: Optional parent KickSequence
        :param file_name:
        :param trim:
        """
        for ck in critical_keys:
            if ck not in df.columns:
                raise Exception(f'Missing critical dataframe column: ({ck})')

        self.v2 = False

        assert len(df) == 1
        self.df = df
        self.idx = kick_id
        self.idx_offset = id_offset
        self.idxg = kick_id + id_offset
        self.set('idx', self.idx)
        self.set('idxg', self.idxg)
        self.set('kick', self)
        self.ks = parent_sequence
        self.file_name = file_name
        self.matrix_cache = {}  # old way of caching BPM matrices
        self.trim = copy.copy(trim)
        self.default_trim = copy.copy(trim)
        self.force_own_trim = False  # if True, all analysis method will use kick trim instead of their own

        if bpm_list is None:
            if iota_defaults:
                bpm_list = set([k[:-1] for k in iota.BPMS.ALLA])
                logger.info(
                    f'BPM list not specified - deducing ({len(bpm_list)}) IOTA BPMs: ({bpm_list})')
            else:
                bpm_list = set(
                        [k[:-1] for k in df.columns if
                         k not in special_keys and k.endswith(('V', 'H', 'S', 'C'))])
                logger.warning('Deducing BPMs!')
                logger.info(
                    f'BPM list not specified - deducing ({len(bpm_list)}) BPMs: ({bpm_list})')

        self.bpm_list_orig = list(copy.copy(bpm_list))
        self.bpm_list = list(copy.copy(bpm_list))

        self.H = [i + "H" for i in bpm_list if i + "H" in df.columns]
        self.PX = [i + "PX" for i in bpm_list if i + "PX" in df.columns]
        self.V = [i + "V" for i in bpm_list if i + "V" in df.columns]
        self.PY = [i + "PY" for i in bpm_list if i + "PY" in df.columns]
        self.S = [i + "S" for i in bpm_list if i + "S" in df.columns]
        self.ALL = []
        self.C = []  # Calculated
        self.CH = []
        self.CV = []
        self.bpm_default_families = ['H', 'PX', 'V', 'PY', 'S']
        self.bpms_update()

        self.bpm_families_active = {'H': self.H, 'V': self.V, 'PX': self.PX, 'PY': self.PY,
                                    'S': self.S,
                                    'A': self.ALL,
                                    'C': self.C, 'CH': self.CH, 'CV': self.CV
                                    }
        self.bpm_families_all = copy.deepcopy(self.bpm_families_active)

        # Store original data
        for f, bpms in self.bpm_families_active.items():
            for i, b in enumerate(bpms):
                self.set(b + self.Datatype.ORIG.value, self.col(b).copy())

        # Convenience attributes
        self.n_turns = self.get_turns()
        self.kickv = self.get('kickv')
        self.kickh = self.get('kickh')
        self.tag = f'{self.idx} {self.kickh:.3f} {self.kickv:.3f}'
        self.nux = None  # main tune
        self.nuy = None  # main tune
        self.fft_pwr = self.fft_freq = self.peaks = None

        # v2
        self.bpm_roll = None
        self.props: Dict[str, pd.DataFrame] = {}
        self.svd: Dict[str, Tuple] = {}
        self.tracks: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.roll_bpm: Optional[str] = None
        self.box: Optional[LatticeContainer] = None

    def upgrade(self, box: "LatticeContainer", silent: bool = False):
        """ Upgrade this kick to v2 """
        if self.v2:
            # pass
            self.bpm_roll = None
            self.roll_bpm: Optional[str] = None
            logger.warning(f'Kick {self.idx=} was already upgraded, resetting')
            # raise AttributeError('Already upgraded!')
        self.box = box
        self.svd = {}  # Stores SVD data
        self.tracks = {'raw': {}, 'orig': {}}  # Stores matrix-format BPM data
        self.props = {}  # Other properties

        # Make sure bpms match
        bpm_names = box.bpm_names
        if bpm_names != self.bpm_list_orig:
            diff = set(bpm_names) - set(self.bpm_list_orig)
            drop_missing = True
            if drop_missing:
                if len(diff) > 0:
                    logger.warning(f'Dropping BPMs in box: {diff}')
                    els = [box.get_one(b, Monitor, exact=True) for b in diff]
                    assert all(el.l == 0.0 for el in els)
                    box.remove_elements(els)
            else:
                raise Exception(f'BPM mismatch - ({diff})')
        bpm_names = box.bpm_names
        if bpm_names != self.bpm_list_orig:
            logger.warning(f'BPM list order does not match box, adjusting')
            logger.warning(f'Box: {bpm_names}')
            logger.warning(f'List: {self.bpm_list_orig}')
            assert len(self.bpm_list_orig) == len(bpm_names)
            self.bpm_list_orig = bpm_names
            self.bpm_list = [b for b in bpm_names if b in self.bpm_list]
        for bpm_name in self.bpm_list_orig:
            bpm = box.get_first(el_name=bpm_name, el_type=Monitor, exact=True)
            bpm.active = True if bpm_name in self.bpm_list else False

        bpms = box.bpms
        active, disabled = [], []
        for b in bpms:
            if b.active:
                active.append(b.id)
            else:
                disabled.append(b.id)
        if not silent:
            logger.info(f'V2 upgrade:')
            logger.info(f'({len(active)}) active ({active})')
            logger.info(f'({len(disabled)}) disabled ({disabled}) ')

        for f in self.bpm_families_active:
            if len(self.bpm_families_active[f]) > 0:
                df = pd.DataFrame(index=self.bpm_list)
                self.props[f] = df

        self.v2 = True
        self.sync()

        if np.any([np.any(v.isnull()) for v in self.tracks['raw'].values()]):
            raise Exception(f'Null values found in data!')
        if np.any([np.any(v.isnull()) for v in self.tracks['orig'].values()]):
            raise Exception(f'Null values found in data!')

        if not silent:
            logger.info(
                f'Upgraded to v2 - ({len(self.tracks["orig"])})/({len(self.tracks["raw"])}) orig/raw frames')
            logger.info(f'Sizes o: ({ {k: v.shape for k, v in self.tracks["orig"].items()} })')
            logger.info(f'Sizes r: ({ {k: v.shape for k, v in self.tracks["raw"].items()} })')

    def sync(self, to_v2: bool = True):
        """ Syncs v1 and v2 data structures - used to keep matrices/dfs current """
        assert getattr(self, 'v2', False)
        if to_v2:
            families = self.bpm_default_families
            for f in families:
                if len(self.bpm_families_active[f]) == 0:
                    continue
                # Get original data
                data = self.get_bpm_data(family=f,
                                         data_type=self.Datatype.ORIG,
                                         no_trim=True)
                array_lengths = [len(v) for v in data.values()]
                assert len(set(array_lengths)) == 1
                assert len(data) == len(self.get_bpms(f))
                ld = array_lengths[0]
                track_data = np.zeros((len(data), ld))
                index = list([k[:-len(f)] for k in data.keys()])
                assert set(index) == set(self.bpm_list_orig)
                assert len(index) == len(self.bpm_list_orig)
                for i, (k, v) in enumerate(data.items()):
                    track_data[i, :] = v
                df = pd.DataFrame(index=index, data=track_data)
                df = df.reindex(index=self.bpm_list_orig)
                assert len(df) == len(self.bpm_list_orig)
                self.tracks['orig'][f] = df

                self.v2_raw_from_orig(f)
        else:
            raise Exception

    def roll(self, bpm_name: str):
        """ Rotate dataset and lattice to a new BPM """
        assert bpm_name in self.bpm_list_orig and bpm_name in self.bpm_list
        bpm = self.box.get_first(bpm_name, el_type=Monitor, exact=True)
        assert bpm.active
        idx_orig = self.bpm_list_orig.index(bpm_name)
        idx = self.bpm_list.index(bpm_name)
        if self.bpm_list[0] == bpm_name:
            logger.warning(f'Kick already starts at ({bpm_name}), {idx=}, {idx_orig=}')
            return
        logger.info(f'Rolling from ({self.bpm_list[0]}) to ({bpm_name})')
        # Rotate list
        self.bpm_list = self.bpm_list[idx:] + self.bpm_list[:idx]
        bpm_list_orig_rolled = self.bpm_list_orig[idx_orig:] + self.bpm_list_orig[:idx_orig]

        # Rotate lattice
        self.box.rotate_lattice(bpm)
        self.box.update()

        self.bpm_roll = bpm_name

        inactive_bpms = [b for b in self.bpm_list_orig if b not in self.bpm_list]

        self.v2_drop_stale_tracks()

        families = self.bpm_default_families
        # Regenerate raw data in default order and roll
        for f in families:
            if len(self.bpm_families_active[f]) == 0:
                continue
            df_orig = self.tracks['orig'][f]
            assert np.all(df_orig.index == self.bpm_list_orig)
            data_rolled = self._roll_bpm_matrix(df_orig.values, idx_orig)
            df = pd.DataFrame(index=bpm_list_orig_rolled, data=data_rolled[:, self.trim])
            df.drop(index=inactive_bpms, inplace=True)
            self.tracks['raw'][f] = df
            assert len(df) == len(self.bpm_list)
            assert np.all(df.index == self.bpm_list)
            assert df.index[0] == bpm_name
            if df.iloc[0, 0] != df_orig.loc[bpm_name, :].values[self.trim][0]:
                raise ValueError(
                        f'Error rolling ({f}): {df.iloc[0, 0:5]} != {df_orig.loc[bpm_name, :].values[self.trim][0:5]}')
        rlen = self.tracks['raw'][families[0]].shape[1]
        olen = self.tracks['orig'][families[0]].shape[1]

        logger.info(f'Rolled to ({bpm_name}), orig length ({olen}), raw length ({rlen}),'
                    f' ({len(self.bpm_list)})/({len(inactive_bpms)}) active/inactive bpms')

    @property
    def N(self):
        """ Current number of turns in raw set """
        return self.tracks['raw'][self.bpm_default_families[0]].shape[1]

    @property
    def NB(self):
        """ Current number of enabled default family BPMs """
        return len(self.bpm_list)

    @property
    def custom(self):
        return self.df.iloc[0,self.df.columns.get_loc('custom')]

    def v2_raw_from_orig(self, f):
        """ Raw matrices from orig ones """
        df_orig = self.tracks['orig'][f]
        assert np.all(df_orig.index == self.bpm_list_orig)
        inactive_bpms = [b for b in self.bpm_list_orig if b not in self.bpm_list]
        v = df_orig.values[:, self.trim].copy()
        v -= np.mean(v, axis=1, keepdims=True)
        df = pd.DataFrame(index=df_orig.index, data=v)
        assert np.all(df.index == self.bpm_list_orig)
        df.drop(index=inactive_bpms, inplace=True)
        assert np.all(df.index == self.bpm_list)
        self.tracks['raw'][f] = df
        return df

    def v2_drop_stale_tracks(self, extras_to_save: List = None):
        """ Dump all data except orig """
        extras_to_save = extras_to_save or []
        to_save = set(['orig'] + extras_to_save)
        keys_track_other = set(self.tracks.keys()) - to_save
        if len(keys_track_other) > 0:
            logger.info(f'Wiping tracks: ({keys_track_other})')
            for k in keys_track_other:
                del self.tracks[k]
            if 'raw' not in to_save:
                self.tracks['raw'] = {}

    def v2_drop_bpms(self, bpms: list[str]):
        if len(bpms) == 0:
            raise Exception('Empty bpm update?')
        logger.info(f'Removing bpm ({bpms})')
        self.v2_drop_stale_tracks(extras_to_save=['raw'])

        for k, df in self.tracks['orig'].items():
            assert np.all(df.index == self.bpm_list_orig)

        for k, df in self.tracks['raw'].items():
            for b in bpms:
                if b in df.index:
                    df.drop(index=b, inplace=True)
            assert np.all(df.index == self.bpm_list)

        for k, df in self.props.items():
            for b in bpms:
                if b in df.index:
                    df.drop(index=bpms, inplace=True)
            assert np.all(df.index == self.bpm_list)
        self.svd = {}

    def _roll_bpm_matrix(self, mat, idx):
        return np.vstack([mat[idx:, :-1], mat[:idx, 1:]])

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def copy(self) -> 'Kick':
        df2 = self.df.copy(deep=True)
        kick = Kick(df=df2, kick_id=self.idx, id_offset=self.idx_offset, trim=self.trim,
                    parent_sequence=self.ks)
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
        # print(value, type(value))
        if column in self.df.columns:
            # self.df.iloc[0, self.df.columns.get_loc(column)] = [value]
            self.df.iat[0, self.df.columns.get_loc(column)] = value
        else:
            # Pandas is horrible with list/array cell contents
            s = pd.Series([value])
            self.df = self.df.assign(**{column: s})
            # self.df.loc[:, column] = [value]
            # self.df.at[self.df.index[0], column] = [value]

    def get(self, column: Union[str, int]):
        """
        Direct column getter
        :param column:
        :return:
        """
        if column in self.df.columns:
            return self.df.iloc[0, self.df.columns.get_loc(column)]
        else:
            if isinstance(column, int) and len(self.df.columns) > column >= 0:
                return self.df.iloc[0, column]
            else:
                raise Exception(
                        f'Key ({column}) is neither a column nor valid integer index - have: ({self.df.columns})')

    def col(self, column: Union[str, int]):
        """
        Return a column of underlying dataframe as value or array (different from KickSequence!!!!)
        :param column:
        :return:
        """
        return self.get(column)
        # return self.df.iloc[0, self.df.columns.get_loc(column)]

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
        print(
                f'Kick idx ({self.idx}) idxglobal ({self.idxg}): ({self.get_turns()}) turns, trim ({self.trim}), at ({self.kickh:.5f})H ({self.kickv:.5f})V')
        self.bpms_summarize_status()

    def set_trim(self, trim: slice):
        self.trim = trim

    def reset_trim(self):
        self.trim = self.default_trim

    def suggest_trim(self, min_idx: int, max_idx: int, threshold: float = 0.2,
                     min_turns: int = None,
                     n_refturns: int = 50, families: list = None, verbose: bool = False,
                     silent: bool = True
                     ):
        """
        Finds longest signal trim within constraints, based on local SNR
        :param n_refturns: Initial signal interval past minidx to compare to
        :param min_idx: Starting index
        :param max_idx: Maximum end index
        :param threshold: Minimum signal fraction
        :param verbose:
        :return: Trim tuple
        """
        offsets = {}
        families = families or ['H', 'V']
        for fam in families:
            for k, v in self.get_bpm_data(family=fam, no_trim=True).items():
                v = v.copy()
                v = v[min_idx:max_idx] - np.mean(v[min_idx:max_idx])
                # Reference is first n_refturns after min_idx
                initial_ampl = np.mean(np.abs(v[:n_refturns]))
                offset = 0
                while True:
                    ampl = np.mean(np.abs(v[offset:30 + offset]))
                    if ampl < initial_ampl * threshold:
                        break
                    elif offset + 30 > max_idx - min_idx:
                        if not silent:
                            logger.warning(
                                f'Trim search on BPM ({k}) reached END OF SIGNAL ({offset})+({min_idx})')
                        break
                    elif offset + 30 > len(v):
                        # logger.warning('Suggested trim search reached END OF SIGNAL')
                        # break
                        raise Exception('Out of bounds problem - check trim logic')
                    offset += 5
                if verbose: print(
                    f'BPM {k}: Iampl {initial_ampl:.3f}, Fampl {ampl:.3f}, offset {offset}')
                offsets[k] = offset
        if verbose: print(f'Found trims: {min_idx} + {offsets}')
        offset_avg = int(np.round(np.mean(list(offsets.values()))))
        if min_turns is not None and offset_avg + 30 < min_turns:
            offset_avg = min_turns - 30
        if verbose: print(f'Average offset: {offset_avg}')
        return np.s_[min_idx:min_idx + offset_avg + 30]

    def search_state(self, search_string: str):
        """
        Searches state for keys matching regex expression
        """
        import re
        r = re.compile(search_string)
        state = self.col('state')
        match_keys = list(filter(r.match, state.keys()))
        return {k: state[k] for k in match_keys}

    def get_column_names(self, bpms: List = None, family: str = None,
                         data_type: Datatype = Datatype.RAW
                         ):
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
        if not all(isinstance(v, np.ndarray) for v in data_list):
            raise Exception(f'List elements are not all arrays')
        if not all(v.ndim == 1 for v in data_list):
            raise Exception(f'List elements are not all 1D: ({[v.ndim for v in data_list]})')
        lengths = np.array([len(v) for v in data_list])
        if not np.all(lengths == lengths[0]):
            raise Exception(f'Data lengths are not equal: ({lengths})')
        matrix = np.vstack(data_list)
        return matrix

    def get_bpm_data(self,
                     columns: Union[List[str], str] = None,
                     bpms: Union[List[str], str] = None,
                     family: str = 'A',
                     data_type: Datatype = Datatype.RAW,
                     return_type: str = None,
                     use_cache: bool = True,
                     add_to_cache: bool = True,
                     no_trim: bool = False,
                     data_trim: slice = None
                     ):
        """
        General data retrieval method for data that is per-bpm

        :param columns: List of literal column names, supercedes all other parameters
        :param bpms: List of bpms (supercedes family), will be resolved agaisnt data_type
        :param family: Family (aka plane) of BPMs, used along with data_type if no bpms/columns provided
        :param data_type: The type of data to return - by default, it is raw TBT data
        :param return_type: 'dict', 'list', 'matrix', 'single'
        :param add_to_cache: In future, result of operations like matrix building will be cached
        :param use_cache:
        :param no_trim: Force full data, ignoring kick trim. Overrides data_trim parameter.
        :param data_trim: Override kick trim
        :return:
        """
        # catch shorthand 1 bpm notation
        if not isinstance(columns, list) and columns is not None:
            columns = [columns]
            if return_type is None:
                return_type = 'single'

        if columns is not None:
            if data_type is not self.Datatype.RAW:
                raise Exception(
                    f'Column list used with nondefault datatype {data_type} - this seems like an error')

        if not isinstance(bpms, list) and bpms is not None:
            bpms = [bpms]
            if return_type is None:
                return_type = 'single'

        if isinstance(columns, list):
            if len(columns) == 0:
                raise Exception(f'Requested BPM data with empty column list - use None instead')

        return_type = return_type or 'dict'
        assert columns is None or isinstance(columns, list)
        assert bpms is None or isinstance(bpms, list)
        assert columns or bpms or family

        if columns is not None and bpms is not None:
            raise Exception(f'Both columns and bpms specified - this is deprecated')

        if columns is None:
            columns = self.get_column_names(bpms, family, data_type)
            if not columns:
                raise Exception(f'No BPMs available for family ({family}) and bpms ({bpms})')
            columns_roots = self.get_column_names(bpms, family, self.Datatype.RAW)
        else:
            columns_roots = columns

        data = {k: self.get(c) for k, c in zip(columns_roots, columns)}
        if (self.trim or data_trim) and not no_trim:
            is_array = [isinstance(v, np.ndarray) for v in data.values()]
            if all(is_array):
                if data_trim:
                    data = {k: v[data_trim] for k, v in data.items()}
                else:
                    data = {k: v[self.trim] for k, v in data.items()}
            elif all([not i for i in is_array]):
                pass  # trim does not apply, return is uniform
            else:
                raise Exception('Data has both arrays and non-arrays, trim is ambiguous')

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

    def drop_bpm_data(self,
                      bpms: Union[List[str], str] = None,
                      family: str = 'A',
                      data_type: Datatype = None,
                      ):
        if data_type is not None:
            dtypes = [data_type]
        else:
            dtypes = [x for x in Kick.Datatype]
        for d in dtypes:
            columns = self.get_column_names(bpms, family, d)
            self.df.drop(columns, axis=1, inplace=True, errors='ignore')

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

    def col_fft(self, bpm: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return FFT of specified BPM
        :param bpm: BPM name
        :return: Frequency, Power arrays
        """
        return self.col(bpm + self.Datatype.FFT_FREQ.value), self.col(
            bpm + self.Datatype.FFT_POWER.value)

    def get_fft_data(self, bpm: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.col(bpm + self.Datatype.FFT_FREQ.value), self.col(
            bpm + self.Datatype.FFT_POWER.value)

    def get_tune_data(self, bpm: str):
        return self.fft_freq[bpm], self.fft_pwr[bpm], self.peaks[bpm]

    def get_tunes(self, bpms: List[str], i: int = None):
        i = str(i) if i is not None else ''
        i = '' if i == 1 else i
        if isinstance(bpms, str):
            return self.get(bpms + self.Datatype.NU.value + i)
        else:
            return (*[self.get(b + self.Datatype.NU.value + i) for b in bpms],)

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
        :param soft_fail:
        :return: List of string BPM keys
        """
        if len(self.bpm_list_orig) == 0:
            return []
        bpms = []
        if isinstance(family, str):
            family = [family]
        for fam in family:
            if fam in self.bpm_families_active:
                bpms += self.bpm_families_active[fam]
            else:
                raise Exception
        if len(bpms) == 0:
            if not soft_fail:
                raise Exception(f'No BPMs found for families: ({family})')
        # Order-preserving unique list conversion
        seen = set()
        bpms = [x for x in bpms if not (x in seen or seen.add(x))]
        return bpms

    def bpms_update(self):
        """ Process BPM list update """
        self.ALL = self.H + self.V + self.S
        if self.v2:
            self.ALL = self.H + self.V + self.S + self.PX + self.PY
            bpm_sets = {}
            for sp in self.bpm_default_families:
                bpm_sets[sp] = {b[:-len(sp)] for b in self.bpm_families_active[sp]}
            # print(bpm_sets)
            nonzero_sets = [v for k, v in bpm_sets.items() if len(v) > 0]
            assert all(nonzero_sets[0] == nz for nz in nonzero_sets)
            assert len(self.bpm_list) >= len(nonzero_sets[0])  # only removals supported
            assert not nonzero_sets[0] - set(self.bpm_list)
            diff = set(self.bpm_list) - nonzero_sets[0]
            for b in diff:
                self.bpm_list.remove(b)
                bpm = self.box.get_first(el_name=b, el_type=Monitor, exact=True)
                bpm.active = False
            logger.info(f'BPM update - removed ({diff}), remaining ({self.bpm_list})')
            self.v2_drop_bpms(diff)

    def bpms_add(self, bpms: List, family: str = 'C'):
        """ Add bpm id to an active family - typically implies it already has data """
        #raise Exception  # temp disable
        for b in bpms:
            fam = family or 'C'
            fam_list = self.bpm_families_active[fam]
            if b not in fam_list:
                fam_list.append(b)

            fam_list = self.bpm_families_all[fam]
            if b not in fam_list:
                fam_list.append(b)
        self.bpms_update()

    def bpms_disable(self, bpms: List, plane: str = 'A'):
        """ Disable BPM, remove from active list - it will not be used in any future calcs """
        # Old signature fix
        if isinstance(bpms, str):
            bpms = [bpms]
        for bpm in bpms:
            if plane == 'A':
                planes_missing = []
                for sp in self.bpm_default_families:
                    bpm_list = self.bpm_families_active[sp]
                    if len(bpm_list) > 0:
                        if bpm + sp in bpm_list:
                            bpm_list.remove(bpm + sp)
                            if len(bpm_list) == 0:
                                logger.warning(f'Plane {sp} has no BPMs left!')
                        else:
                            planes_missing.append(sp)
                if planes_missing:
                    raise Exception(f'BPM ({bpm}) not active in planes ({planes_missing})')
                for sp in ['C']:
                    bpm_list = self.bpm_families_active[sp]
                    if bpm + sp in bpm_list:
                        bpm_list.remove(bpm + sp)
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
            if delta:
                print(
                    f'Plane ({k:2s}): ({len(v_all):<2d}) BPMs total, ({len(v):<2d}) active, disabled: ({delta})')
            else:
                print(
                    f'Plane ({k:2s}): ({len(v_all):<2d}) BPMs total, ({len(v):<2d}) active, disabled: (None)')

    def bpms_apply_filters(self,
                           plane: str,
                           methods: List[Tuple[Callable, Tuple]],
                           data_type: Datatype = Datatype.RAW,
                           delete_on_fail: bool = False,
                           debug: bool = True
                           ):
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
            assert set(results.keys()) == set(data.keys())
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
                          silent: bool = True
                          ) -> Tuple[Dict, Dict]:
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
                             use_percentile_std: bool = True
                             ):
        if method == 'mean':
            means = {k: np.mean(v) for k, v in data.items()}
            means_vals = np.array([v for v in means.values()])
            mean = np.mean(means_vals)
            if use_percentile_std:
                pcs = np.percentile(means_vals, [10, 90])
                std = np.std(means_vals[(means_vals > pcs[0]) & (means_vals < pcs[1])])
            else:
                std = np.std(means_vals)
            outliers = {k: v for k, v in means.items() if
                        v > mean + sigma * std or v < mean - sigma * std}
            results = {k: k not in outliers for k, v in data.items()}
            results_data = {k: {'method': 'mean', 'values': (mean, std * sigma, v)} for k, v in
                            means.items()}
        else:
            raise Exception(f'Method ({method}) is unrecognized')
        return results, results_data

    def bpms_filter_absval(self,
                           data: Dict,
                           method: str = 'abs',
                           threshold: float = 10.0,
                           neg_threshold: float = None
                           ):
        """
        Filters data by absolute signal threshold. Specify negative value, or -positive will be assumed.
        :param data:
        :param method:
        :param threshold:
        :param neg_threshold:
        :return:
        """
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
                             threshold: float = 1.2
                             ):
        if method == 'mean':
            data_demean = {k: v[1:] - np.mean(v[1:]) for k, v in data.items()}
            vals1 = {k: (np.mean(v[v > 0.0]), -np.mean(v[v < 0.0])) for k, v in data_demean.items()}
            outliers = {k: v for k, v in vals1.items() if
                        v[0] / v[1] > threshold or v[1] / v[0] > threshold}
            results = {k: k not in outliers for k, v in data.items()}
            results_data = {k: {'method': method,
                                'values': (threshold, v[0], v[1], v[0] / v[1], v[1] / v[0])
                                } for (k, v) in
                            vals1.items()}
        elif method == 'pilot':
            pilot_slice = np.s_[1:171]
            data_demean = {k: v[1:] - np.mean(v[pilot_slice]) for k, v in data.items()}
            vals1 = {k: (np.mean(v[v > 0.0]), -np.mean(v[v < 0.0])) for k, v in data_demean.items()}
            outliers = {k: v for k, v in vals1.items() if
                        v[0] / v[1] > threshold or v[1] / v[0] > threshold}
            results = {k: k not in outliers for k, v in data.items()}
            results_data = {k: {'method': method,
                                # 'values': (threshold, v[0], v[1], v[0]/v[1], v[1]/v[0])} for (k, v) in vals1.items()}
                                'values': f'+avg:{v[0]:.3f} vs -avg:{v[1]:.3f} above ratio of {threshold}'
                                }
                            for (k, v) in vals1.items()}
        elif method == 'endpilot':
            pilot_slice = np.s_[500:]
            data_demean = {k: v[172:] - np.mean(v[pilot_slice]) for k, v in data.items()}
            vals1 = {k: (np.mean(v[v > 0.0]), -np.mean(v[v < 0.0])) for k, v in data_demean.items()}
            outliers = {k: v for k, v in vals1.items() if
                        v[0] / v[1] > threshold or v[1] / v[0] > threshold}
            results = {k: k not in outliers for k, v in data.items()}
            results_data = {k: {'method': method,
                                'values': (threshold, v[0], v[1], v[0] / v[1], v[1] / v[0])
                                } for (k, v) in
                            vals1.items()}
        else:
            raise Exception(f'Method ({method}) is unrecognized')
        return results, results_data

    def bpms_filter_signal_ratio(self,
                                 data: Dict,
                                 method: str = 'std',
                                 threshold: int = 2,
                                 splits: int = 4,
                                 data_region: slice = None,
                                 relative: bool = True
                                 ):
        """
        Tests if first region of signal has specified property, optionally relative to the last region
        """
        results = {}
        results_data = {}
        for k, signal in data.items():
            assert len(signal) >= splits
            split_signals = np.array_split(signal, splits)
            if data_region:
                split_signals[0] = signal[data_region]
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
                results_data[k] = {'test': method + '_relative',
                                   'values': (threshold, val1, val2, val1 / val2)
                                   }
        return results, results_data

    # Helpers for state categories

    def get_optics(self) -> Dict[str, float]:
        """ Gets all relevant optics settings """
        quads = [q + '.SETTING' for q in iota.QUADS.ALL_CURRENTS]
        squads = [q + '.SETTING' for q in iota.SKEWQUADS.ALL_CURRENTS]
        return {k: self.state(k) for k in quads + squads}

    def get_correctors(self, include_physical=False) -> Dict[str, float]:
        """ Gets all available corrector settings """
        if include_physical:
            elements = [q + '.SETTING' for q in iota.CORRECTORS.ALL]
        else:
            elements = [q + '.SETTING' for q in iota.CORRECTORS.ALL_VIRTUAL]
        return {k: self.state(k) for k in elements}

    def get_quadrupoles(self) -> Dict[str, float]:
        """ Gets quarupole settings """
        elements = [q + '.SETTING' for q in iota.QUADS.ALL_CURRENTS]
        return {k: self.state(k) for k in elements}

    def get_skewquads(self) -> Dict[str, float]:
        """ Gets quarupole settings """
        elements = [q + '.SETTING' for q in iota.SKEWQUADS.ALL_CURRENTS]
        return {k: self.state(k) for k in elements}

    def get_sextupoles(self) -> Dict[str, float]:
        """ Gets sextupole settings """
        elements = [q + '.SETTING' for q in iota.SEXTUPOLES.ALL_CURRENTS]
        return {k: self.state(k) for k in elements}

    def get_turns(self):
        bpms = self.get_bpms()
        # print(self.col(bpm))
        if len(bpms) > 0:
            return len(self.col(bpms[0]))
        else:
            return None

    ### Physics starts here

    def v2_add_noise(self, families, noise_amp, output_key='raw_noised'):
        """ For simulation studies, add fake gaussian noise """
        families = families or self.bpm_default_families
        if output_key not in self.tracks:
            self.tracks[output_key] = {}
        for family in families:
            df = self.tracks['raw'][family]
            v = df.values.copy()
            v += np.random.randn(*v.shape) * noise_amp
            if family in self.tracks[output_key]:
                logger.warning(f'Key ({family}) in dict ({output_key}) already exists, overwriting')
            self.tracks[output_key][family] = pd.DataFrame(index=df.index, data=v)
            assert np.all(self.tracks[output_key][family].index == df.index)

    add_noise = v2_add_noise

    def v2_demean(self, families=None, key='raw'):
        families = families or self.bpm_default_families
        if key not in self.tracks:
            raise Exception
        for family in families:
            if family in self.tracks['raw']:
                df = self.tracks['raw'][family]
                if key != 'raw':
                    df = df.copy()
                self.tracks[key][family] = df.sub(df.mean(axis=1), axis=0)

    def v2_envelope(self,
                    key: str = 'clean',
                    key_out: str = 'env',
                    families: List[str] = None,
                    ):
        from scipy.signal import hilbert
        # from scipy.signal import savgol_filter
        families = families or self.bpm_default_families
        for family in families:
            if len(self.bpm_families_active[family]) == 0:
                continue
            df = self.tracks[key][family]
            matrix = df.values
            transform = np.abs(hilbert(matrix))

            if key_out not in self.tracks:
                self.tracks[key_out] = {}
            self.tracks[key_out][family] = pd.DataFrame(index=df.index, data=transform)

    def v2_apply(self,
                 fun: Callable,
                 key: str = 'clean',
                 out: str = None,
                 families: List[str] = None,
                 method: str = 'matrix',
                 ):
        families = families or self.bpm_default_families
        for family in families:
            if len(self.bpm_families_active[family]) == 0:
                continue
            df = self.tracks[key][family]
            if method == 'matrix':
                matrix = df.values
                result = fun(matrix)
                if out is not None:
                    if out not in self.tracks:
                        self.tracks[out] = {}
                    self.tracks[out][family] = pd.DataFrame(index=df.index, data=result)
                else:
                    return result
            else:
                raise Exception(f' Method {method} unrecognized')

    def v2_fit_decoherence(self,
                           fun=None,
                           amp=None,
                           bounds=None,
                           key_data: str = 'clean',
                           key_env: str = 'env',
                           out: str = 'dec',
                           out_fit: str = 'envfit',
                           families: List[str] = None,
                           fit_trim: slice = None,
                           individual_fit: bool = True,
                           dry_run: bool = False
                           # output_trim: slice = None
                           ):
        """ Fit supplied function to data - typically used to envelope decoherence fits """
        import scipy.optimize as scopt
        families = families or self.bpm_default_families
        for family in families:
            if len(self.bpm_families_active[family]) == 0:
                continue
            df = self.tracks[key_env][family]
            matrix = df.values
            matrix_data = self.tracks[key_data][family].values

            # if output_trim is not None:
            #    data_dec = data_dec[:, output_trim]
            if fit_trim is not None:
                matrix = matrix[:, fit_trim]
                matrix_data = matrix_data[:, fit_trim]
            data_dec = np.zeros_like(matrix)
            data_env = np.zeros_like(matrix)

            reslist = []
            reslist_x = []
            data_x = np.arange(matrix.shape[1])
            if individual_fit:
                for row in range(matrix.shape[0]):
                    env = matrix[row, :]
                    data_y = matrix_data[row, :]

                    res = scopt.differential_evolution(fun, bounds=bounds, args=(data_x, env),
                                                       popsize=10)
                    reslist.append(res)
                    reslist_x.append(res.x)
                    data_env[row, :] = amp(data_x, res.x)
                    data_dec[row, :] = data_y * data_env[row, :].max() / data_env[row, :]
                    # if output_trim is None:
                    #    data_dec[row, :] = data_y * data_env[row, :].max() / data_env[row, :]
                    # else:
                    #    data_dec[row, :] = (data_y * data_env[row, :].max() / data_env[row, :])[output_trim]
            else:
                # Fit for single envelope up to constant
                env = matrix
                if dry_run:
                    res = [0]
                    reslist = [res for i in range(matrix.shape[0])]
                    for row in range(matrix.shape[0]):
                        data_y = matrix_data[row, :]
                        data_env[row, :] = np.ones_like(data_y)
                        data_dec[row, :] = data_y.copy()
                        reslist_x.append([0])
                else:
                    res = scopt.differential_evolution(fun, bounds=bounds, args=(data_x, env),
                                                       popsize=100)
                    reslist = [res for i in range(matrix.shape[0])]
                    for row in range(matrix.shape[0]):
                        data_y = matrix_data[row, :]
                        data_env[row, :] = amp(data_x, res.x, row)
                        data_dec[row, :] = data_y * data_env[row, :].max() / data_env[row, :]
                        reslist_x.append(res.x)

            if out not in self.tracks:
                self.tracks[out] = {}
            self.tracks[out][family] = pd.DataFrame(index=df.index, data=data_dec)

            if out_fit not in self.tracks:
                self.tracks[out_fit] = {}
            self.tracks[out_fit][family] = pd.DataFrame(index=df.index, data=data_env)

            self.props[family]['env_res'] = pd.DataFrame(index=df.index, data={'res': reslist})

            self.props[family]['env_x'] = pd.Series(index=df.index, data=reslist_x)

    def v2_clean_svd(self,
                     n_comp: int = 5,
                     families: List[str] = None,
                     key: str = 'raw',
                     key_out: str = 'clean',
                     swap_raw: bool = True,
                     dominance: bool = False,
                     ):
        """
        Clean kick using SVD, reconstructing each BPM from specified number of components
        """
        families = families or self.bpm_default_families
        for family in families:
            if len(self.bpm_families_active[family]) == 0:
                continue
            # matrix = self.get_bpm_data(family=family, return_type='matrix')
            df = self.tracks[key][family]
            matrix = df.values
            matrix = matrix - np.mean(matrix, axis=1)[:, np.newaxis]
            U, S, vh = np.linalg.svd(matrix, full_matrices=False)
            # V = vh.T  # transpose it back to conventional U @ S @ V.T
            self.svd[family] = (U, S, vh, np.diag(S) @ vh)

            # Check dominance
            if not np.all(np.max(np.abs(U), axis=0) < 0.95):
                # bpmmax = df.index[np.argmax(np.abs(U))]
                cmax = np.argmax(np.max(np.abs(U), axis=0))
                if dominance:
                    raise ValueError(
                        f'Dominance detected C{cmax}{family}: {np.max(np.abs(U), axis=0)}')
                else:
                    logger.warning(f'Dominance {family}: {np.max(np.abs(U), axis=0)}')

            # Reconstruct signal
            signal = U[:, :n_comp] @ np.diag(S[:n_comp]) @ vh[:n_comp, :]
            assert signal.shape == matrix.shape

            self.props[family]['svd_noise'] = (signal - matrix).std(axis=1)
            self.props[family]['clean_mean'] = signal.mean(axis=1)
            self.props[family]['clean_std'] = signal.std(axis=1)

            if key_out not in self.tracks:
                self.tracks[key_out] = {}
            self.tracks[key_out][family] = pd.DataFrame(index=df.index, data=signal)

            if swap_raw:
                for i, b in enumerate(self.tracks[key_out][family].index):
                    self.set(b + family + self.Datatype.RAW.value, signal[i, :].copy())

    clean_svd = v2_clean_svd

    def v2_normalize_coordinates(self,
                                 families: List[str] = None,
                                 families_mom: List[str] = None,
                                 key: str = 'clean',
                                 out: str = 'norm',
                                 optics_map: Dict[str, str] = None,
                                 momentum: bool = False,
                                 ):
        from . import Coordinates
        assert families is not None
        # If converting momentum, momentum families should be given as well
        if momentum:
            assert families_mom is not None and len(families) == len(families_mom)
        else:
            families_mom = [None] * len(families)
        optics_map = optics_map or {'H': 'X', 'V': 'Y'}
        df_optics = self.box.compute_bpm_tables(self.bpm_list, active_only=True)
        for family, family_mom in zip(families, families_mom):
            if len(self.bpm_families_active[family]) == 0:
                continue
            assert family in optics_map
            df = self.tracks[key][family]
            matrix = df.values
            data_xnorm = np.zeros_like(matrix)
            if momentum:
                matrix_p = self.tracks[key][family_mom].values
                data_pxnorm = np.zeros_like(matrix)
                for row in range(matrix.shape[0]):
                    bx = df_optics.loc[df.index[row], 'B' + optics_map[family]]
                    ax = df_optics.loc[df.index[row], 'A' + optics_map[family]]
                    data_xnorm[row, :], data_pxnorm[row, :] = Coordinates.normalize(matrix[row, :],
                                                                                    matrix_p[row,
                                                                                    :],
                                                                                    beta=bx,
                                                                                    alpha=ax)
            else:
                for row in range(matrix.shape[0]):
                    bx = df_optics.loc[df.index[row], 'B' + optics_map[family]]
                    data_xnorm[row, :] = Coordinates.normalize_x(matrix[row, :], beta=bx)

            if out not in self.tracks:
                self.tracks[out] = {}
            self.tracks[out][family] = pd.DataFrame(index=df.index, data=data_xnorm)
            if momentum:
                self.tracks[out][family_mom] = pd.DataFrame(index=df.index, data=data_pxnorm)

    def v2_momentum(self,
                    families: List[str],
                    key: str = 'norm',
                    out: Union[str, None] = 'normmom',
                    bpm: str = None,
                    pairs: list = None,
                    optics_map: dict = None,
                    normalized: bool = True,
                    debug: bool = False
                    ):
        """ Compute momentum pairwise for either all pairs or only those starting at particular BPM """
        from . import Coordinates
        assert families is not None
        optics_map = optics_map or {'H': 'X', 'V': 'Y'}
        df_optics = self.box.compute_bpm_tables(self.bpm_list, active_only=True)
        if bpm is not None:
            assert bpm in self.bpm_list and bpm in df_optics.index
            others = self.bpm_list.copy()
            others.remove(bpm)
            combinations = [(bpm, b2) for b2 in others]
            if debug:
                logger.info(f'Computing for BPM {bpm} - pairs {combinations}')
        elif pairs is not None:
            assert all(len(b) == 2 for b in pairs)
            assert all(b[0] in self.bpm_list and b[0] in df_optics.index for b in pairs)
            assert all(b[1] in self.bpm_list for b in pairs)
            combinations = pairs
            if debug:
                logger.info(f'Computing pairs {combinations}')
        else:
            combinations = list(itertools.permutations(self.bpm_list, 2))
            if debug:
                logger.info(f'Computing all permutations: {combinations}')

        output = {}
        for family in families:
            if len(self.bpm_families_active[family]) == 0:
                continue
            df = self.tracks[key][family]
            index_list = []
            data = np.zeros((len(combinations), df.shape[1]))
            if debug:
                logger.info(f'Kick momentum - {key=} | {family=} | {normalized=}')
            if normalized:
                for row, (bpm1, bpm2) in enumerate(combinations):
                    index_list.append((bpm1, bpm2))
                    x1 = df.loc[bpm1, :]
                    x2 = df.loc[bpm2, :]
                    dp = df_optics.loc[bpm2, 'MU' + optics_map[family]] - df_optics.loc[
                        bpm1, 'MU' + optics_map[family]]
                    px1n = Coordinates.calc_pxn_from_normalized_bpms(x1n=x1, x2n=x2, dphase=dp)
                    data[row, :] = px1n
                    if debug:
                        logger.info(f'Pair ({(bpm1, bpm2)}) - {dp=:.3f}')
            else:
                for row, (bpm1, bpm2) in enumerate(combinations):
                    index_list.append((bpm1, bpm2))
                    x1 = df.loc[bpm1, :]
                    x2 = df.loc[bpm2, :]
                    b1 = df_optics.loc[bpm1, 'B' + optics_map[family]]
                    b2 = df_optics.loc[bpm2, 'B' + optics_map[family]]
                    a1 = df_optics.loc[bpm1, 'A' + optics_map[family]]
                    a2 = df_optics.loc[bpm2, 'A' + optics_map[family]]
                    dp = df_optics.loc[bpm2, 'MU' + optics_map[family]] - df_optics.loc[
                        bpm1, 'MU' + optics_map[family]]
                    px1n = Coordinates.calc_px_from_bpms(x1, x2, beta1=b1, beta2=b2, a1=a1, a2=a2,
                                                         dphase=-dp)
                    data[row, :] = px1n
                    if debug:
                        logger.info(
                            f'Pair ({(bpm1, bpm2)}) - {b1=:.3f}, {b2=:.3f}, {a1=:.3f}, {a2=:.3f}, {dp=:.3f}')
            # if out is None:
            output[family] = pd.DataFrame(index=index_list, data=data)
            if out is not None:
                if out not in self.tracks:
                    self.tracks[out] = {}
                self.tracks[out][family] = output[family]
        if out is None:
            return output

    def calculate_tune(self,
                       naff: NAFF,
                       method: str = 'NAFF',
                       families: List[str] = None,
                       selector: Callable = None,
                       search_kwargs: Dict[str, int] = None,
                       use_precalculated: bool = True,
                       data_trim: slice = None,
                       freq_trim: tuple = None,
                       pairs: bool = False,
                       append_results: bool = False,
                       **kwargs
                       ):
        """
        Calculates tune by finding peaks in FFT data, optionally using precomputed data to save time

        :param naff: tbt.NAFF object to be used - will take priority over kick values
        :param method: Peak finding method - NAFF or FFT
        :param families: Families to perform calculation on - typically H, V, or C
        :param selector: Function that picks correct peak from list
        :param search_kwargs: Method specific extra parameters to be used in the search
        :param use_precalculated: Whether to use pre-calculated FFT data
        :param data_trim: Trim override - if not provided, use whatever NAFF object has
        :param pairs: If true, both planes will be resolved at same time. Selectors needs to
        handle signature like (self, results, (bh, bv), search_kwargs)
        :return:
        """
        families = families or ['H', 'V']
        search_kwargs = search_kwargs or {}
        freq = {}
        pwr = {}
        if append_results:
            if hasattr(self, 'peaks'):
                peaks = self.peaks
            else:
                peaks = {}
        else:
            peaks = {}
        average_tunes = {f: [] for f in families}
        if not pairs:
            bpms = self.get_bpms(families)
            for i, bpm in enumerate(bpms):
                family = bpm[-1]
                if method.upper() == 'FFT':
                    if selector is None:
                        raise Exception('FFT tune requires a selector method!')
                    col_fr = bpm + self.Datatype.FFT_FREQ.value
                    col_pwr = bpm + self.Datatype.FFT_POWER.value
                    if use_precalculated and col_fr in self.df.columns and col_pwr in self.df.columns:
                        res = naff.fft_peaks(data=None,
                                             search_peaks=True,
                                             search_kwargs=search_kwargs,
                                             fft_freq=self.df.iloc[0].loc[col_fr],
                                             fft_power=self.df.iloc[0].loc[col_pwr])

                    else:
                        res = naff.fft_peaks(
                                data=self.df.iloc[0].loc[bpm],
                                search_peaks=True,
                                search_kwargs=search_kwargs,
                        )
                    top_tune, peak_tunes, peak_idx, peak_props, (pf, pp) = res
                    # a, b = naff.fft(self.df.iloc[0].loc[bpm])
                    freq[bpm] = pf
                    pwr[bpm] = pp
                    peaks[bpm] = (peak_tunes, peak_props)
                    nu = selector(self, top_tune, peaks[bpm], bpm, search_kwargs)
                    self.df['nu_' + bpm] = nu
                    average_tunes[family].append(nu)
                elif method == 'NAFF':
                    n_components = search_kwargs.get('n_components', 2)
                    if data_trim:
                        # Use provided trims
                        nfresult = naff.run_naff_v2(self.get_bpm_data(bpm, no_trim=True)[data_trim],
                                                    n_components=n_components,
                                                    data_trim=np.s_[:])
                    else:
                        # Use NAFF trims
                        nfresult = naff.run_naff_v2(self.get_bpm_data(bpm, no_trim=True),
                                                    n_components=n_components)
                    peaks[bpm] = ([n['tune'] for n in nfresult], nfresult)
                    top_tune = nfresult[np.argmax([n['absamps'] for n in nfresult])]['tune']
                    if selector:
                        nu = selector(self, top_tune, peaks[bpm], bpm, search_kwargs)
                        self.df[bpm + self.Datatype.NU.value] = nu
                        average_tunes[family].append(nu)
                    else:
                        raise Exception('NAFF tune requires a selector method!')
                else:
                    raise Exception
        else:
            # Requires H and V simultaneously
            bpmsh = self.get_bpms(families[0])
            bpmsv = self.get_bpms(families[1])
            for bh, bv in zip(bpmsh, bpmsv):
                if method == 'NAFF':
                    n_components = search_kwargs.get('n_components', 2)
                    # Allow for variations in H/V component number
                    if isinstance(n_components, int):
                        n_components = (n_components, n_components)
                    results = []
                    for i, (bpm, nc) in enumerate(zip([bh, bv], n_components)):
                        ft = None
                        if freq_trim is not None:
                            if len(freq_trim) == 2 and isinstance(freq_trim[0], tuple):
                                ft = freq_trim[i]
                            elif len(freq_trim) == 2:
                                ft = freq_trim
                            else:
                                raise Exception
                        data = self.get_bpm_data(bpm, no_trim=True).copy()
                        if data_trim:
                            # Use provided trims
                            nfresult = naff.run_naff_v2(data[data_trim],
                                                        n_components=nc,
                                                        data_trim=np.s_[:],
                                                        freq_trim=ft)
                        else:
                            # Use NAFF trims
                            nfresult = naff.run_naff_v2(data,
                                                        n_components=nc,
                                                        freq_trim=ft)
                        peaks[bpm] = ([n['tune'] for n in nfresult], nfresult)
                        results.append(peaks[bpm])
                    if selector:
                        nux, nuy = selector(self, results, (bh, bv), search_kwargs)
                        self.df[bh + self.Datatype.NU.value] = nux
                        self.df[bv + self.Datatype.NU.value] = nuy
                        average_tunes[families[0]].append(nux)
                        average_tunes[families[1]].append(nuy)
                    else:
                        raise Exception('NAFF tune requires a selector method!')
                else:
                    raise Exception('For pair peak finding, only NAFF implemented')
        # self.fft_freq = freq
        # self.fft_pwr = pwr
        self.peaks = peaks
        if 'H' in families:
            self.nux = np.nanmean(average_tunes['H'])
            self.df['nux'] = self.nux
            self.df['sig_nux'] = np.nanstd(average_tunes['H'])
        if 'V' in families:
            self.nuy = np.nanmean(average_tunes['V'])
            self.df['nuy'] = self.nuy
            self.df['sig_nuy'] = np.nanstd(average_tunes['V'])
        return freq, pwr, peaks

    def update_mean_tune(self, families):
        families = families or ['H', 'V']
        if 'H' in families:
            bpms = self.get_bpms('H')
            tunes = [self.get(b + self.Datatype.NU.value) for b in bpms]
            self.nux = np.nanmean(tunes)
            self.df['nux'] = self.nux
            self.df['sig_nux'] = np.nanstd(tunes)
        if 'V' in families:
            bpms = self.get_bpms('V')
            tunes = [self.get(b + self.Datatype.NU.value) for b in bpms]
            self.nuy = np.nanmean(tunes)
            self.df['nuy'] = self.nuy
            self.df['sig_nuy'] = np.nanstd(tunes)

    def calculate_fft(self,
                      naff: NAFF,
                      families: List[str] = None,
                      spacing: float = 1.0,
                      data_trim: slice = None,
                      data_type: Datatype = Datatype.RAW
                      ):
        """
        Calculates FFT for each bpms and stores in dataframe
        :param naff: NAFF object that will do the FFT
        :param families:
        :param naff:
        :return:
        """
        families = families or ['H', 'V', 'C']
        bpms = self.get_bpms(families, soft_fail=True)
        for i, bpm in enumerate(bpms):
            if data_trim:
                # Use provided trims
                data = self.get_bpm_data(bpms=bpm, no_trim=True, data_type=data_type).copy()[
                    data_trim]
                fft_freq, fft_power = naff.fft(
                        data,
                        data_trim=np.s_[:],
                        spacing=spacing)
            else:
                # Use NAFF trims
                data = self.get_bpm_data(bpms=bpm, no_trim=True, data_type=data_type).copy()
                fft_freq, fft_power = naff.fft(data,
                                               spacing=spacing)
            # fft_freq, fft_power = naff.fft(self.get(bpm))
            self.df[bpm + self.Datatype.FFT_FREQ.value] = [fft_freq]
            self.df[bpm + self.Datatype.FFT_POWER.value] = [fft_power]

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

    def calculate_amplitude(self, families=None, data_trim=None):
        families = families or ['H', 'V']
        data_trim = data_trim or self.trim
        bpms = self.get_bpms(families)
        for bpm in bpms:
            data = self.get_bpm_data(bpm, data_trim=data_trim).copy()
            data -= np.mean(data)
            data = np.abs(data)
            data = np.sort(data)
            self.df[bpm + self.Datatype.AMP.value] = np.mean(data[-2:])
            self.df[bpm + self.Datatype.AMPSIG.value] = np.std(data[-3:])

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
            print(len(columns), np.hstack([averages, variances])[np.newaxis, :].shape,
                  self.df.loc[0, columns])
            self.df.loc[0, columns] = np.hstack([averages, variances])[np.newaxis, :]

        return stats


class KickSequence:
    BPMS_ACTIVE = []

    def __init__(self, kicks: list, demean=True, eid=None, props=None):
        # assert all(k.idx_offset == kicks[0].idx_offset[0] for k in kicks)
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
        self.id = eid or 'GenericKS'
        self.props = props or {}

        # demean data
        # if demean:
        #     df.loc[:, self.BPMS_HG + self.BPMS_VG] = df.loc[:, self.BPMS_HG + self.BPMS_VG].applymap(
        #         lambda x: x - np.mean(x))

    def __len__(self):
        return len(self.kicks)

    def check_dataset_integrity(self,
                                quadrupoles: bool = True,
                                skewquads: bool = True,
                                correctors: bool = True,
                                sextupoles: bool = True,
                                octupoles: bool = True,
                                nl: bool = True,
                                skew_tol=3e-3,
                                bad_word_strings: List[str] = None,
                                bad_word_keys: List[str] = None,
                                exclusions: List[int] = None
                                ) -> Set[str]:
        """
        Checks if certain state parameters are the same for all kicks.
        Includes all standard elements + RF. Some categories can be disabled when differences are expected.
        :param exclusions:
        :param quadrupoles:
        :param skewquads: Combined skewquads+HV correctors
        :param correctors:
        :param skew_tol: Tolerance in skew quads - necessary because these are virtual devices with resolution jitter
        :param sextupoles: Sextupoles
        :param octupoles: Octupoles
        :param nl: DN magnet
        :param bad_word_strings: Parameters (keys) to check for bad words
        :param bad_word_keys: List of bad words
        :return:
        """
        logger.info(f'Checking ({self.id}) states')
        invariant_devices = iota.DIPOLES.ALL_I + \
                            ['N:IRFLLA', 'N:IRFMOD', 'N:IRFEAT', 'N:IRFEPC'] + \
                            ['N:IKPSVX', 'N:IKPSVD']
        if quadrupoles:
            invariant_devices += iota.QUADS.ALL_CURRENTS
        if skewquads:
            invariant_devices += iota.SKEWQUADS.ALL_CURRENTS + iota.CORRECTORS.ALL
        if sextupoles:
            invariant_devices += iota.SEXTUPOLES.ALL_CURRENTS
        if octupoles:
            invariant_devices += iota.OCTUPOLES.ALL_CURRENTS
        if nl:
            invariant_devices += iota.DNMAGNET.ALL_CURRENTS
        invariant_devices = set(invariant_devices)
        kicks = self.kicks

        def state_filter(sd_list):
            # Given list of state kv dicts, return decision on whether differences are fatal
            abort = False
            if not all(x == sd_list[0] for x in sd_list):
                def f(key, vals):
                    dev = key.split('.')[0]
                    if dev in iota.SKEWQUADS.ALL_CURRENTS + iota.CORRECTORS.ALL:  # COMBINED_VIRTUAL:
                        # Compare within tolerances
                        return [np.abs(v - vals[0]) > skew_tol for v in vals]
                    elif dev in iota.DNMAGNET.ALL_CURRENTS + iota.OCTUPOLES.ALL_CURRENTS:
                        # Only compare nonzeroes
                        uniques, counts = np.unique([v for v in vals if v != 0.],
                                                    return_counts=True)
                        if len(uniques) > 1:
                            return [True if v == 0. else v == uniques[0] for v in vals]
                    else:
                        return [v != vals[0] for v in vals]

                for key in sd_list[0].keys():
                    # Scan all kicks for each key
                    values = [sd[key] for (kick, sd) in zip(kicks, sd_list)]
                    mask = f(key, values)
                    # print(key, mask)
                    if any(mask):
                        # uniques, counts, idxs = np.unique()
                        ok_tuples = [(kick.idx, sd[key]) for (kick, sd, m) in
                                     zip(kicks, sd_list, mask) if not m]
                        bad_tuples = [(kick.idx, sd[key], kick.kickh, kick.kickv) for (kick, sd, m)
                                      in
                                      zip(kicks, sd_list, mask) if m]
                        # If we have exclusions, check their global id against exclusions
                        if exclusions:
                            idxs = [kick.idxg for (kick, sd, m) in zip(kicks, sd_list, mask) if m]
                            idxs_excluded = [idx for idx in idxs if idx in exclusions]
                            idxs_not_excluded = [idx for idx in idxs if idx not in exclusions]
                            if not idxs_not_excluded:
                                logger.warning(
                                    f'Inconsistent kicks {idxs_excluded} overriden, rest passed!')
                                continue
                        logger.error(f'{key} inconsistent')
                        logger.error(f'>OK:{ok_tuples}')
                        logger.error(f'>BAD:{bad_tuples}')
                        abort = True
                        # return True
                # for kick, sd in zip(kicks, sd_list):
                #     def f(k, v1, v2):
                #         dev = k.split('.')[0]
                #         if dev in iota.SKEWQUADS.ALL_CURRENTS + iota.CORRECTORS.ALL:#COMBINED_VIRTUAL:
                #             return np.abs(v1 - v2) > skew_tol
                #         elif dev in iota.DNMAGNET.ALL_CURRENTS:
                #             if v1 == 0. or v2 == 0.:
                #                 return False
                #         else:
                #             return v1 - v2 != 0.
                #
                #     delta = {k: (sd[k], sd_list[0][k]) for k in sd if f(k, sd[k], sd_list[0][k])}
                #     if delta:
                #         abort = True
                #         logger.error(f'Kick {kick.idx} is not consistent: delta {delta}')
            return abort
            # return False

        # Sanity checks
        times = [k.state('aq_timestamp') for k in kicks]
        assert all(np.diff(times) > 0)  # All in order
        assert max(times) - min(times) < 3600  # 1 hour delta max
        seq_nums = [k.state('N:EA5TRG.READING') for k in kicks]
        assert all(np.diff(seq_nums) > 0)  # Counter counting up

        # Check all states have same contents
        states = [k.get_full_state() for k in kicks]
        keys = [set(s.keys()) for s in states]
        shared_keys = set.intersection(*keys)
        assert all(len(shared_keys) == len(k) for k in keys)

        # Start detailed checks by category
        sum_states = [{} for i in range(len(kicks))]
        abort = False

        def check_category(states):
            nonlocal sum_states, abort
            abort |= state_filter(states)
            [ss.update(s) for ss, s in zip(sum_states, states)]

        # Quadrupoles
        if quadrupoles: check_category([k.get_quadrupoles() for k in kicks])
        # Skewquads
        if skewquads: check_category([k.get_skewquads() for k in kicks])
        # Orbit
        if correctors: check_category([k.get_correctors(include_physical=True) for k in kicks])
        # Sextupoles
        if sextupoles: check_category([k.get_sextupoles() for k in kicks])
        # Octupoles
        if octupoles: check_category(
                [{dev: k.state(dev + '.SETTING') for dev in iota.OCTUPOLES.ALL_CURRENTS} for k in
                 kicks])
        # NL
        if nl: check_category(
                [{dev: k.state(dev + '.SETTING') for dev in iota.DNMAGNET.ALL_CURRENTS} for k in
                 kicks])
        # Abort if any fail
        if abort:
            raise Exception('Sequence states inconsistent')

        # Check that remaining keys all match exactly
        kvtuples_rem = [set(k.get_full_state().items()) - set(ss.items()) for k, ss in
                        zip(kicks, sum_states)]
        shared_kvts = set.intersection(*kvtuples_rem)
        outliers = set()
        for kvt in kvtuples_rem:
            outliers.update(kvt - shared_kvts)
        outlier_keys = set()
        for e in outliers:
            outlier_keys.update((e[0],))
        outlier_devs = set([x.split('.')[0] for x in outlier_keys])
        if len(invariant_devices.intersection(outlier_devs)) != 0:
            raise Exception(
                f'Found invariants in outlier devices: {invariant_devices.intersection(outlier_devs)}')

        # Check for bad strings in custom keys (i.e. kicker status)
        if bad_word_keys and bad_word_strings:
            for k in kicks:
                for word in bad_word_strings:
                    for key in bad_word_keys:
                        value = k.state(key)
                        if word in value:
                            logging.warning(
                                f'Found bad word ({word}) in ({key}) for kick ({k.idx})({k.idxg})')
                            raise Exception(
                                f'Found bad word ({word}) in ({key}) for kick ({k.idx})({k.idxg})')
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

    def remove_kicks(self, kick_ids: List[Union[int, Kick]], local: bool = True):
        """
        Remove specified kicks from sequence
        :param local:
        :param kick_ids: Kick ids or objects themselves
        """
        removed_list = []
        for kid in kick_ids:
            if isinstance(kid, int):
                try:
                    k = self.get_kick(kid, local=local)
                except AttributeError as e:
                    logger.warning(f'Kick ({kid}) not found in KS ({self.id}) - reason {e}')
                    continue
            elif isinstance(kid, Kick):
                k = kid
            else:
                raise Exception(f'Unrecognized kick identified ({kid})')
            self.kicks.remove(k)
            removed_list.append(k.idx)
        logger.info(f'Removed kicks ({removed_list}) from ({self.id})')
        self.update_df()

    def __getitem__(self, item: int) -> Kick:
        """
        Gets the specified kick BY POSITION!!!
        :param item:
        :return:
        """
        assert isinstance(item, int)
        while item < 0:
            item += len(self.kicks)  # wrap negative index
        if item >= len(self.kicks):
            raise IndexError(f'Sequence has ({len(self.kicks)}) kicks, so ({item}) out of bounds')
        return self.kicks[item]

    def summarize(self):
        kv = [k.kickv for k in self.kicks]
        kh = [k.kickh for k in self.kicks]
        print(
            f'KickSequence ({self.id}): ({len(self.kicks)}) kicks, max H ({max(kh):.5f}), max V ({max(kv):.5f})')
        print(f'Config dictionary: {self.props}')
        ids = [k.idx for k in self.kicks]
        print(f'IDs: {ids}')
        gids = [k.idxg for k in self.kicks]
        print(f'GIDs: {gids}')

    def get_kick(self, kick_id: int, local: bool = False) -> Kick:
        """
        Gets kick object in this KickSequence with specified index
        :param local:
        :param kick_id: Kick integer id
        :return: Kick object
        """
        if local:
            indices = self.df.idx
        else:
            indices = self.df.idxg
        tp = 'local' if local else 'global'
        if kick_id not in indices.values:
            raise AttributeError(f'Kick ({tp}) id ({kick_id}) not found - have ({indices.values})')
        df_row = self.df.loc[indices == kick_id]
        if df_row.shape[0] != 1:
            raise AttributeError(
                f'Kick ({tp}) id ({kick_id}) is not unique - result size is ({df_row.shape})')
        else:
            return df_row.iloc[0, self.df.columns.get_loc('kick')]

    def get_tune_data(self, families: Union[List[str], str], filter_fun: Callable = None,
                      i: int = None
                      ):
        """
        Get previously computed tune data (mean and std) for the specified families.
        :param i: Which result set to return
        :param families:
        :return:
        """
        nu = []
        sig = []
        if isinstance(families, str):
            families = [families]
        kicks = self.kicks
        if filter_fun is not None:
            kicks = [k for k in kicks if filter_fun(k)]
            if not kicks:
                return ([], []), ([], []), ([], [])
        kicksh = np.array([k.kickh for k in kicks])
        kicksv = np.array([k.kickv for k in kicks])
        for family in families:
            tunes = np.nanmean([k.get_tunes(k.get_bpms(family=family), i=i) for k in kicks], axis=1)
            tunes_std = np.nanstd([k.get_tunes(k.get_bpms(family=family), i=i) for k in kicks],
                                  axis=1)
            nu.append(tunes)
            sig.append(tunes_std)
        return (kicksh, kicksv), nu, sig

    # def get_kick(self, idx):
    #    return Kick(self.df.loc[idx, :], kick_id=idx, parent_sequence=self)

    def update_df(self):
        dflist = [k.df for k in self.kicks]
        index = [k.idxg for k in self.kicks]
        self.df = pd.concat(dflist).sort_values(['kickv', 'kickh'])
        self.df.index = index

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
                       method: str = 'NAFF',
                       families: List[str] = None,
                       selector: Callable = None,
                       search_kwargs: Dict[str, int] = None
                       ):
        naff = naff or self.naff
        assert selector
        for r in self.df.itertuples():
            r.kick.calculate_tune(naff, method=method,
                                  families=families, selector=selector,
                                  search_kwargs=search_kwargs)

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
