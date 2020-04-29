from typing import Tuple

import numpy as np
from pyIOTA.tbt.tbt import Kick
from numba import jit
from scipy.optimize import curve_fit
from scipy.signal import hilbert


class Invariants:
    @staticmethod
    def compute_CS(x, px, y, py, normalized=True):
        """Compute Courant-Snyder invariants in x and y"""
        assert normalized
        return (x ** 2 + px ** 2), (y ** 2 + py ** 2)

    @staticmethod
    def compute_CS_2D(x, px, normalized=True):
        """Compute Courant-Snyder invariants in x and y"""
        assert normalized
        return (x ** 2 + px ** 2)

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


class Envelope:
    """
    Provides several models of signal amplitude envelopes, and associated fitting methods
    """
    def __init__(self, data_trim=None, output_trim=None):
        self.data_trim = data_trim
        self.output_trim = output_trim

    # def budkerfit(xdata, amplitude, tau, c2, freq, ofsx, ofsy):
    #     return amplitude*np.exp(-tau*(xdata-ofsx)**2)*np.exp(-c2*(1-np.cos(freq*(xdata-ofsx)))) + ofsy
    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def budkerfit(xdata: np.ndarray, amplitude: float, tau: float, c2: float, freq: float, ofsx: float,
                  ofsy: float, c3: float):
        ans = amplitude * np.exp(-tau * (xdata - ofsx) ** 2 - c2 * (np.sin(freq * (xdata - ofsx)) ** 2)) + ofsy
        ans = ans * (1+c3*np.arange(len(ans)))
        return ans

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def coupled_envelope(x: np.ndarray,
                         amplitude: float,
                         c1: float, c2: float, c3: float, nu:float,
                         ofsx:float, ofsy: float):
        """
        Envelope that includes chromaticity and octupolar nonlinear decoherence, multiplied by coupling cosine envelope
        In other words, a 2D model with additional coupling 'beating' envelope
        :param xdata:
        :param amplitude:
        :param tau:
        :param c2:
        :param freq:
        :param ofsx:
        :param ofsy:
        :param c3:
        :return:
        """
        xsc = x * c2
        xscsq = xsc**2
        ans = 1/(1+xscsq)   *   np.exp(-xscsq*(c1**2)/(1+xscsq))    *   np.exp(-((np.sin[nu*x])*(c3/nu))**2)
        #ans = ans * (1+c3*np.arange(len(ans)))
        return ans

    def find_envelope(self, data_raw, normalize=False, p0=None, lu=None, full=False):
        if np.any(np.isnan(data_raw)):
            raise Exception(f'Some of given data is NaN!')
        data_trim = self.data_trim
        data_raw = data_raw[data_trim]
        if p0 is None:
            p0 = [1, 1e-05, 1e-3, 1e-5, -40, 0]
        if lu is None:
            lu = [5e3, np.inf, np.inf, 0.5e-1, 0, 1]

        data_mean = np.mean(data_raw)
        data = data_raw - data_mean
        analytic_signal = hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        fl = 15
        xdata = np.array(range(len(amplitude_envelope)))
        xdatafl = xdata[fl:-fl]
        ydatafl = amplitude_envelope[fl:-fl]
        try:
            popt, pcov = curve_fit(self.budkerfit, xdatafl, ydatafl,
                                  # bounds=([0, 0, -np.inf, -1e-1, -np.inf, -1],
                                   #        lu),
                                   p0=p0,
                                   # p0=[1,1e-5,1e-2,1,-1,1e-2],
                                   maxfev=10000)
            #print(popt)
            envelope = self.budkerfit(xdata, *popt)
            if np.any(np.isnan(envelope)):
                raise Exception()
        except Exception as e:
            print(e, 'FAIL')
            envelope = np.ones_like(xdata)
            popt = pcov = None
            # env = np.ones(len(data))*np.mean(data)
        env_return = envelope / max(envelope) if normalize else envelope
        if full:
            return env_return, amplitude_envelope+data_mean, popt, pcov
        else:
            return env_return

    def normalize_bpm_signal(self, data, p0=None, lu=None):
        data_mean = np.mean(data)
        envelope = self.find_envelope(data, normalize=True, p0=p0, lu=lu)
        return (data - data_mean) / envelope + data_mean

    def normalize_bpm_signal_v2(self, data: Kick, p0=None, lu=None):
        bpms = data.get_bpms(['H','V'])

        data_mean = np.mean(data)
        envelope = self.find_envelope(data, normalize=True, p0=p0, lu=lu)
        return (data - data_mean) / envelope + data_mean


class SVD:

    def __init__(self, data_trim: Tuple = None, output_trim: Tuple = None):
        self.data_trim = data_trim
        self.output_trim = output_trim

    def decompose2D(self, data: Kick, plane: str = None, use_kick_trim: bool = True):
        """
        Decompose any matrix-castable object using SVD
        :param data:
        :param plane:
        :param use_kick_trim:
        :return:
        """
        if isinstance(data, Kick):
            #matrix = data.get_bpm_matrix(plane)
            matrix = data.get_bpm_data(family=plane, return_type='matrix')
            if self.data_trim and not use_kick_trim:
                matrix = matrix[:, self.data_trim]
        elif isinstance(data, np.ndarray):
            matrix = data
            if self.data_trim:
                matrix = matrix[:, self.data_trim]
        else:
            raise Exception(f'Unknown data type: {data}')

        matrix -= np.mean(matrix, axis=1)[:, np.newaxis]
        U, S, vh = np.linalg.svd(matrix, full_matrices=False)
        V = vh.T  # transpose it back to conventional U @ S @ V.T
        return U, S, V, vh

    def decompose_kick_2D(self, kick: Kick, tag: str = 'SVD', use_kick_trim: bool = True, add_virtual_bpms: bool = True):
        """
        Decompose kick using SVD and store results
        :param kick:
        :param tag:
        :return:
        """
        assert isinstance(kick, Kick)
        for plane in ['H', 'V']:
            #matrix = kick.get_bpm_matrix(plane)
            U, S, V, vh = self.decompose2D(kick, plane=plane, use_kick_trim=use_kick_trim)
            #kick.df[f'{tag}_{plane}_M0'] = vh[0, :]
            #kick.df[f'{tag}_{plane}_M1'] = vh[1, :]
            kick.df[f'{tag}_{plane}_S'] = [S]
            kick.df[f'{tag}_{plane}_U'] = [U]
            kick.df[f'{tag}_{plane}_vh'] = [vh]
            if add_virtual_bpms:
                kick.bpms_add([f'SVD2D_{plane}_1C', f'SVD2D_{plane}_2C'])
                kick.df[f'SVD2D_{plane}_1C'] = [vh[0, :]]
                kick.df[f'SVD2D_{plane}_2C'] = [vh[1, :]]
        return U, S, V, vh

    def decompose4D(self, data: Kick, use_kick_trim: bool = True):
        """
        Decompose any matrix-castable object using SVD
        :param data:
        :param plane:
        :param use_kick_trim:
        :return:
        """
        if isinstance(data, Kick):
            #matrix = data.get_bpm_matrix(plane)
            matrix1 = data.get_bpm_data(family='H', return_type='matrix')
            matrix2 = data.get_bpm_data(family='V', return_type='matrix')
            matrix = np.vstack([matrix1, matrix2])
            if self.data_trim and not use_kick_trim:
                matrix = matrix[:, self.data_trim]
        elif isinstance(data, np.ndarray):
            matrix = data
            if self.data_trim:
                matrix = matrix[:, self.data_trim]
        else:
            raise Exception(f'Unknown data type: {data}')

        matrix -= np.mean(matrix, axis=1)[:, np.newaxis]
        U, S, vh = np.linalg.svd(matrix, full_matrices=False)
        V = vh.T  # transpose it back to conventional U @ S @ V.T
        return U, S, V, vh

    def decompose_kick_4D(self, kick: Kick,
                          tag: str = 'SVD4D',
                          use_kick_trim: bool = True,
                          add_virtual_bpms: bool = True):
        """
        Decompose kick using SVD and store results
        :param kick:
        :param tag:
        :return:
        """
        assert isinstance(kick, Kick)
        plane = 'HV'
        U, S, V, vh = self.decompose4D(kick, use_kick_trim=use_kick_trim)
        #kick.df[f'{tag}_{plane}_M0'] = [vh[0, :]]
        #kick.df[f'{tag}_{plane}_M1'] = [vh[1, :]]
        kick.df[f'{tag}_{plane}_S'] = [S]
        kick.df[f'{tag}_{plane}_U'] = [U]
        kick.df[f'{tag}_{plane}_vh'] = [vh]
        if add_virtual_bpms:
            kick.bpms_add(['SVD4D_1C', 'SVD4D_2C', 'SVD4D_3C', 'SVD4D_4C'])
            kick.df['SVD4D_1C'] = [vh[0, :]]
            kick.df['SVD4D_2C'] = [vh[1, :]]
            kick.df['SVD4D_3C'] = [vh[2, :]]
            kick.df['SVD4D_4C'] = [vh[3, :]]
        return U, S, V, vh

    def get_4D(self, kick: Kick, tag: str = 'SVD4D'):
        plane = 'HV'
        S = kick.df[f'{tag}_{plane}_S']
        U = kick.df[f'{tag}_{plane}_U']
        vh = kick.df[f'{tag}_{plane}_vh']
        return U, S, vh.T, vh

    def decompose2D_into_kicks(self, kick: Kick, plane: str):
        assert isinstance(kick, Kick)
        new_kicks = []

        for plane in ['H', 'V']:
            k2 = kick.copy()
            bpms = kick.get_bpms(plane)
            matrix = kick.get_bpm_matrix(plane)
            matrix -= np.mean(matrix, axis=1)[:, np.newaxis]
            U, S, vh = np.linalg.svd(matrix, full_matrices=False)
            V = vh.T  # transpose it back to conventional U @ S @ V.T
            c1 = 0
            c2 = 1
            # for bpm in bpms:
            #    k2.df.loc[bpm] =
        return U, S, V, vh


class ICA:
    def __init__(self, data_trim=None, output_trim=None):
        self.data_trim = data_trim
        self.output_trim = output_trim

    def decompose2D(self, data: Kick, plane: str):
        if isinstance(data, Kick):
            matrix = data.get_bpm_matrix(plane)
        elif isinstance(data, np.ndarray):
            matrix = data
        else:
            raise Exception(f'Unknown data type: {data}')

        if self.data_trim:
            matrix = matrix[:, self.data_trim]

        matrix -= np.mean(matrix, axis=1)[:, np.newaxis]
        U, S, vh = np.linalg.svd(matrix, full_matrices=False)
        V = vh.T  # transpose it back to conventional U @ S @ V.T
        return U, S, V, vh
