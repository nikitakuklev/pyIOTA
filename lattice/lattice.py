import functools
import pathlib
import sys


"""
Holds information about the lattice, mostly optics
"""


class Lattice:







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