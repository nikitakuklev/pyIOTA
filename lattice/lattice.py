import functools
import pathlib
import sys
import numpy as np
import pandas as pd

"""
Holds information about the lattice, mostly optics
"""


class Lattice:
    omega62 = np.array(
        [[0, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, -1, 0]])
    omega6 = np.array(
        [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [-1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0],
         [0, 0, -1, 0, 0, 0]])
    omega4 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]])
    omega42 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    omega2 = np.array([[0, 1], [-1, 0]])

    def __init__(self):
        self.bpm_s = None
        self.bpm_beta = None
        self.bpm_phase = None
        self.bpm_alpha = None
        if pathlib.Path.home().as_posix() not in sys.path: sys.path.insert(1, pathlib.Path.home().as_posix())
        import pymadx

    def load_optics(self, lattice_file):
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

    def load_transfer_maps(self, file, filetype='madx'):
        # Read in matrices
        if filetype == 'madx':
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
            self.maps_source = 'madx'
            return df
        elif filetype == '6dsim':
            with open(file, 'r') as f:
                tmaps = {}
                lines = f.readlines()
                lines = lines[14:]
                for i, l in enumerate(lines):
                    if i % 7 == 0:
                        src, to = l.split('->')[0].strip(), l.split('->')[1].strip()
                    else:
                        j = i % 7 - 1
                        if j == 0: matr = []
                        # print(list(filter(None, l.strip().split(' '))))
                        matr.append(np.array(list(filter(None, l.strip().split(' ')))).astype(np.float))
                        if j == 5: tmaps[to] = (src, np.stack(matr));  # print(dataarr)
                tmaps['start'] = ('end', np.identity(6))
            self.maps_transfer = tmaps
            self.maps_source = '6dsim'
            return tmaps
        else:
            raise

    def load_oneturn_maps(self, file, filetype='madx'):
        # Read in matrices
        if filetype == 'madx':
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
        else:
            raise

    def get_transport_matrix(self, start, end, df=None):
        if df is None:
            df = self.maps_transfer
        elements = np.array([s.upper() for s in df.index])
        start = start.split('x')[0].split('y')[0].upper()
        end = end.split('x')[0].split('y')[0].upper()

        if start not in elements:
            raise Exception("Missing start", start)
        if end not in elements:
            raise Exception("Missing end", end)

        idx_start = np.argwhere(elements == start)[0, 0]
        idx_end = np.argwhere(elements == end)[0, 0]
        if idx_end > idx_start:
            matrices = df.iloc[idx_start:idx_end + 1, :].loc[:, 'M']
        else:
            m1 = df.iloc[idx_start:, :].loc[:, 'M']
            m2 = df.iloc[0:idx_end + 1, :].loc[:, 'M']
            # print(len(m1),len(m2))
            matrices = np.concatenate([m1, m2])

        print('{}({}) -> {}({}): {} elements'.format(start, idx_start, end, idx_end, len(matrices)))
        assert len(matrices) > 1
        # M = np.linalg.multi_dot(matrices[::-1])
        M = functools.reduce(np.dot, matrices[::-1])

        return M, matrices

    # matrixcache = {}
    # def getM_6dsim(start, end):
    #     endpoint = end.split('x')[0].split('y')[0]
    #     elem = start.split('x')[0].split('y')[0]
    #     if elem in matrixcache and endpoint in matrixcache[elem]:
    #         return matrixcache[elem][endpoint]
    #     else:
    #         matrixlist = []
    #         if elem not in tmaps:
    #             raise Exception("Missing start", start)
    #         if endpoint not in tmaps:
    #             raise Exception("Missing end", end)
    #         while True:
    #             prev_elem = tmaps[elem][0]
    #             M = tmaps[elem][1]
    #             matrixlist.append(M)
    #             if prev_elem == endpoint:
    #                 break
    #             elem = prev_elem
    #         matrixlist = matrixlist[::-1]
    #         M = np.linalg.multi_dot(matrixlist[::-1])
    #         if not elem in matrixcache:
    #             matrixcache[elem] = {}
    #         matrixcache[elem][endpoint] = M
    #         return M

    # def getMv2_6dsim(start, end):
    #     elem = end.split('x')[0].split('y')[0]
    #     startpoint = start.split('x')[0].split('y')[0]
    #     matrixlist = []
    #     if elem not in tmaps:
    #         raise Exception("Missing start", start)
    #     if startpoint not in tmaps:
    #         raise Exception("Missing end", end)
    #     while True:
    #         prev_elem = tmaps[elem][0]
    #         M = tmaps[elem][1]
    #         matrixlist.append(M)
    #         if prev_elem == startpoint:
    #             break
    #         elem = prev_elem
    #     matrixlist = matrixlist[::-1]
    #     if len(matrixlist) > 1:
    #         # lhs addition of next step
    #         M = np.linalg.multi_dot(matrixlist[::-1])
    #     else:
    #         M = matrixlist[0]
    #     return M, matrixlist