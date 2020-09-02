__all__ = ['MADXOptics', 'TFS', 'parse_lattice']

from pathlib import Path
import numpy as np
import pandas as pd
from pyIOTA.lattice import NLLens
from ocelot.cpbd.elements import *

element_mapping = {
    'MARKER': Marker,
    'MONITOR': Monitor,
    'DRIFT': Drift,
    'KICKER': Drift,
    'VKICKER': Drift,
    'SBEND': SBend,
    'DIPEDGE': Edge,
    'QUADRUPOLE': Quadrupole,
    'SEXTUPOLE': Sextupole,
    'OCTUPOLE': Octupole,
    'NLLENS': NLLens,
    'SOLENOID': Drift,
    'RFCAVITY': Cavity,
    'MULTIPOLE': Multipole,
}

unit = lambda x, el: x
nz = lambda x, el: x if x != 0.0 else None
nz2 = lambda x, el: 2.0 * x if x != 0.0 else None
nzdivl = lambda x, el: (x / el.l if el.l > 0.0 else x) if x != 0.0 else None

attribute_mapping = {'L': ('l', unit),
                     'ANGLE': ('angle', nz),
                     'TILT': ('tilt', nz),
                     'VOLT': ('v', nz),  # MV->MV
                     'HGAP': ('gap', nz2),
                     'FINT': ('fint', nz)}
attribute_mapping.update({f'K{i}L': (f'k{i}', nzdivl) for i in range(1, 4)})
k_attrs = [f'k{i}' for i in range(1, 4)]

critical_columns = ['NAME', 'KEYWORD']


def parse_lattice(fpath: Path, verbose: bool = False):
    """
    Parse MADX TFS file (TWISS output) into native lattice
    :param fpath: Full file path
    :param verbose:
    :return:
    """
    tf = TFS(fpath)
    df = tf.df
    s = 0.0
    seq = []
    # Basic logic - parse one element per line, assign any attributes from columns
    columns = [c for c in df.columns if c not in critical_columns and c in attribute_mapping]
    if not df.NAME.is_unique:
        print('WARN - Element names are not unique, this might cause issues')
    for r in df.itertuples():
        r = r._asdict()
        l = r['L']
        etype = r['KEYWORD'].upper()
        s += l
        assert np.isclose(s, r['S'])
        if etype in element_mapping:
            el = element_mapping[etype](eid=r['NAME'])
            for col in columns:
                attr_name, filter_fun = attribute_mapping[col]
                v = filter_fun(r[col], el)
                if v is not None:
                    setattr(el, attr_name, v)
            if isinstance(el, Multipole):
                ks = [getattr(el, ka, None) for ka in k_attrs]
                while ks[-1] == 0.0:
                    ks.pop()
                while len(ks) < 2:
                    ks.append(0.0)
                el.kn = np.ndarray(ks)
                el.n = len(ks)
            seq.append(el)
        else:
            raise Exception(f'Unrecognized element {r.NAME} of type {etype}')

    # Handle edges
    seq_temp = seq.copy()
    edge_count = 0
    for i, el in enumerate(seq_temp):
        if isinstance(el, Edge):
            # Can't have unpaired edges
            edge_count += 1
            assert isinstance(seq_temp[i + 1], SBend) or isinstance(seq_temp[i - 1], SBend)
        elif isinstance(el, SBend):
            e1 = seq_temp[i - 1]
            e2 = seq_temp[i + 1]
            if isinstance(e1, Edge) and isinstance(e2, Edge):
                assert e1.gap == e2.gap
                el.fint = e1.fint
                el.fintx = e2.fint
                el.gap = e1.gap
                seq.remove(e1)
                seq.remove(e2)
                edge_count -= 2
            elif isinstance(e1, Edge):
                el.fint = e1.fint
                el.fintx = 0.0
                el.gap = e1.gap
                seq.remove(e1)
                edge_count -= 1
            elif isinstance(e2, Edge):
                el.fint = 0.0
                el.fintx = e2.fint
                el.gap = e2.gap
                seq.remove(e2)
                edge_count -= 1
            else:
                el.fint = el.fintx = el.gap = 0.0
    assert edge_count == 0

    return seq, None, None, None, None


class TFS:
    def __init__(self, fpath: Path):
        from tfs import read, write
        self.path = fpath
        self.df = read(fpath)
        self.ro = True

    def reload(self):
        from tfs import read, write
        self.df = read(self.path)

    def commit(self):
        if not self.ro:
            from tfs import read, write
            write(self.path, self.df)
        else:
            raise Exception("File is read-only")


class MADXOptics:
    """
    This class contains several previous-gen optics io methods.
    There are DEPRECATED - it is strongly suggested to use native OCELOT extensions in lattice module.
    """

    def __init__(self):
        import pymadx
        self.bpm_s = None
        self.bpm_beta = None
        self.bpm_phase = None
        self.bpm_alpha = None
        # if pathlib.Path.home().as_posix() not in sys.path:
        #     sys.path.insert(1, pathlib.Path.home().as_posix())

    def load_optics(self, lattice_file: Path):
        """
        DEPRECATED
        Loads optics from MADX TFS file
        :param self:
        :param lattice_file:
        :return:
        """
        iota = pymadx.Data.Tfs(lattice_file)
        bpms = iota.GetElementsOfType('MONITOR')
        print(f'Found {len(bpms)} BPMS: {bpms}')

        bpm_s_temp = {bpm['NAME'].replace('BPM', 'B'): bpm['S'] for bpm in bpms}
        bpm_s = {'IBA1C': bpm_s_temp['IBA1']}
        del bpm_s_temp['IBA1']
        bpm_s.update(bpm_s_temp)

        bpm_beta_temp = {bpm['NAME'].replace('BPM', 'B'): (bpm['BETX'], bpm['BETY']) for bpm in bpms}
        bpm_beta = {'IBA1C': bpm_beta_temp['IBA1']}
        del bpm_beta_temp['IBA1']
        bpm_beta.update(bpm_beta_temp)

        bpm_alpha_temp = {bpm['NAME'].replace('BPM', 'B'): (bpm['ALFX'], bpm['ALFY']) for bpm in bpms}
        bpm_alpha = {'IBA1C': bpm_alpha_temp['IBA1']}
        del bpm_alpha_temp['IBA1']
        bpm_alpha.update(bpm_alpha_temp)

        bpm_phase = {bpm['NAME'].replace('BPM', 'B'): (bpm['MUX'], bpm['MUY']) for bpm in bpms}
        bpm_phase['IBA1C'] = bpm_phase['IBA1']
        del bpm_phase['IBA1']

        self.bpm_s = bpm_s
        self.bpm_beta = bpm_beta
        self.bpm_phase = bpm_phase
        self.bpm_alpha = bpm_alpha
        return bpm_s, bpm_phase, bpm_beta, bpm_alpha

    def load_transfer_maps(self, file: Path):
        """
        DEPRECATED
        Reads in MADX transfer matrix file. You should really just use OCELOT...
        :param file:
        :param filetype:
        :return:
        """
        f = pymadx.Data.Tfs(file)
        df = pd.DataFrame.from_dict(f.data, orient='index', columns=f.columns)
        df = df.loc[:, 'NAME':'R66']

        matrix_entries = ['R{}{}'.format(i, j) for i in range(1, 7) for j in range(1, 7)]
        data_matrix = df.loc[:, matrix_entries].values.reshape((-1, 6, 6))
        split_matrices = [np.asfortranarray(data_matrix[i, ...]) for i in range(len(df))]

        elements = df.index
        assert len(np.unique(elements)) == len(elements)
        assert len([el for el in elements if 'START' in el]) == 1

        df.loc[:, 'M'] = split_matrices
        self.maps_transfer = df
        return df

    def load_oneturn_maps(self, file: Path):
        """
        DEPRECATED
        Reads in MADX transfer matrix file. You should really just use OCELOT...
        :param file:
        :param filetype:
        :return:
        """
        f = pymadx.Data.Tfs(file)
        df = pd.DataFrame.from_dict(f.data, orient='index', columns=f.columns)
        df = df.loc[:, 'NAME':'RE66']

        matrix_entries = ['RE{}{}'.format(i, j) for i in range(1, 7) for j in range(1, 7)]
        data_matrix = df.loc[:, matrix_entries].values.reshape((-1, 6, 6))
        split_matrices = [data_matrix[i, ...] for i in range(len(df))]

        elements = df.index
        assert len(np.unique(elements)) == len(elements)
        assert len([el for el in elements if 'START' in el]) == 1

        df.loc[:, 'M'] = split_matrices
        self.maps_oneturn = df
        return df
