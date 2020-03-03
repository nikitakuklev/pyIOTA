import numpy as np
from pyIOTA.tbt.tbt import Kick


class Invariants:
    @staticmethod
    def compute_CS(x, px, y, py, normalized=True):
        """Compute Courant-Snyder invariants in x and y"""
        assert normalized
        return (x ** 2 + px ** 2), (y ** 2 + py ** 2)

    @staticmethod
    def compute_I1(x, px, y, py, alpha, normalized=True):
        """Compute first DN invariant"""
        assert normalized
        I = x ** 2 + y ** 2 + px ** 2 + py ** 2 + alpha * (x ** 4 + y ** 4 + 3 * y ** 2 * x ** 2) / 2
        return I


class Coordinates:
    @staticmethod
    def normalize_x(x, beta):
        """
        Compute normalized transverse position
        :param x:
        :param beta:
        :return:
        """
        return x / np.sqrt(beta)

    @staticmethod
    def normalize(x, px, beta, alpha):
        """
        Compute normalized transverse position and momentum
        :param x:
        :param px:
        :param beta:
        :param alpha:
        :return:
        """
        return x / np.sqrt(beta), x * alpha / np.sqrt(beta) + np.sqrt(beta) * px

    @staticmethod
    def calc_px_from_bpms(x1, x2, beta1, beta2, alpha1, alpha2, dphase):
        """
        Compute momentum at location 1 from position readings at locations 1 and 2
        :param x1:
        :param x2:
        :param beta1:
        :param beta2:
        :param alpha1:
        :param alpha2:
        :param dphase:
        :return:
        """
        return x2 * (1 / np.sin(dphase)) * (1 / np.sqrt(beta1 * beta2)) - \
               x1 * (1 / np.tan(dphase)) * (1 / beta1) - x1 * alpha1 / beta1


class Phase():
    pass


class Decoherence():
    pass


class SVD:
    @staticmethod
    def decompose2D(data: Kick, plane):
        if isinstance(data, Kick):
            matrix = data.get_bpm_matrix(plane)
        elif isinstance(data, np.ndarray):
            matrix = data
        else:
            raise Exception

        matrix -= np.mean(matrix, axis=1)[:, np.newaxis]
        U, S, vh = np.linalg.svd(matrix, full_matrices=False)
        V = vh.T  # transpose it back to conventional U @ S @ V.T
        return U, S, V, vh
