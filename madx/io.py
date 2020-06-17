__all__ = ['MADXOptics']

from pathlib import Path
import numpy as np
import pandas as pd


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
