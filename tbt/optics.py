__all__ = ['Invariants', 'Coordinates', 'Phase', 'Twiss', 'SVD', 'Interpolator']

import logging
from typing import Tuple, List

import numpy as np
from pyIOTA.tbt.tbt import Kick
from numba import jit
from scipy.optimize import curve_fit
from scipy.signal import hilbert

import pyIOTA.math as pmath

logger = logging.getLogger(__name__)


class Routines:
    @staticmethod
    def linear_optics_2D_SVD(kick: Kick, family: str = 'H'):
        svd = SVD()
        U, S, V, vh = svd.get_data_2D(kick, family)
        Vf = U @ np.diag(S)
        wrap = pmath.Wrapper(-np.pi, np.pi)
        phases_exp_rel = Phase.calc_from_modes(mode2=Vf[:, 1], mode1=Vf[:, 0])
        phases_exp_abs = Phase.relative_to_absolute(phases_exp_rel)
        betax_exp_unscaled = Twiss.from_SVD(U, S, V)
        return phases_exp_rel, betax_exp_unscaled


class NIO:
    @staticmethod
    def convert_t_to_kappa(t, c):
        """ Compute conversion from t strength of DN magnet to kappa notation of QI"""
        # From mathematica, have 6*kappa = 16*t/c^2
        if c != 0.01:
            logger.warning(f'Unusual geometric constant c={c} has been supplied')
        return (8.0 / 3.0) * t / (c * c)

    @staticmethod
    def convert_kappa_to_t(kappa, c):
        if c != 0.01:
            logger.warning(f'Unusual geometric constant c={c} has been supplied')
        """ Convert QI kappa to DN strength t"""
        return (3.0 / 8.0) * c * c * kappa


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
        I = x ** 2 + y ** 2 + px ** 2 + py ** 2 + alpha * (x ** 4 + y ** 4 - 3 * (y ** 2) * (x ** 2)) / 2
        return I


class Coordinates:
    @staticmethod
    def normalize_x(x: np.ndarray, beta: float) -> np.ndarray:
        """
        Compute normalized transverse position
        """
        return x / np.sqrt(beta)

    @staticmethod
    def normalize(x: np.ndarray, px: np.ndarray, beta: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normalized transverse position and momentum from local (x,px) and twiss functions
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


class Phase:
    @staticmethod
    def calc_from_modes(mode2: np.ndarray, mode1: np.ndarray) -> np.ndarray:
        """
        Calculates phases from (sin/cos) spatial modes.
        Mode order/sign flip doesnt change relative phase distances, but has bounds complications, so don't do it.
        This is since arctan[z] = -arctan[-z] = arccot[1/z] = pi/2-arctan[1/z]
        :param mode1:
        :param mode2:
        :return:
        """
        phases = np.arctan2(mode2, mode1)
        return phases

    @staticmethod
    def calc_from_2bpm(bpm1, bpm2):
        """
        Calculates phase phi12 between two signals. This requires finding phase of C/S components via FFT
        See thesis CERN SL/96-70 (BI)
        """
        raise Exception('Phase calculation requires spectral decomposition, not implemented')

    @staticmethod
    def relative_to_absolute(phases: np.ndarray) -> np.ndarray:
        """
        Converts relative phases to absolutes, assuming first phase is 0. Only works if all deltas < 2pi
        :param phases:
        :return:
        """
        phases_rel = phases.copy()
        phases_cum = np.zeros_like(phases)
        import pyIOTA.math as pmath
        phases_rel = pmath.addsubtract_wrap(phases_rel, -phases_rel[0], -np.pi, np.pi)
        for i in range(1, len(phases)):
            phases_cum[i] = phases_cum[i - 1] + pmath.forward_distance(phases_rel[i - 1], phases_rel[i], -np.pi, np.pi)
        return phases_cum


class Twiss:
    @staticmethod
    def beta1_from_3bpm(beta1_model, dphase12, dphase12_model, dphase13, dphase13_model):
        """
        Model-dependent beta-function at BPM1, calculated via 3-BPM method
        See below for references
        """
        cot = np.arctan
        assert dphase12 > 0 and dphase13 > 0 and \
               dphase12_model > 0 and dphase13_model > 0
        beta1 = beta1_model * (cot(dphase12) - cot(dphase13)) / (cot(dphase12_model) - cot(dphase13_model))
        return beta1

    @staticmethod
    def beta_from_3bpm(beta1_model, beta2_model, beta3_model, dphase12, dphase12_model, dphase13, dphase13_model,
                       dphase23, dphase23_model):
        """
        Model-dependent beta-functions calculated via 3-BPM method
        Ref:
         Luminosity and beta function measurement at the electron - positron collider ring LEP
         CERN-SL-96-070-BI
        """
        cot = np.arctan
        assert dphase12 > 0 and dphase13 > 0 and dphase23 > 0 and \
               dphase12_model > 0 and dphase13_model > 0 and dphase23_model > 0
        beta1 = beta1_model * (cot(dphase12) - cot(dphase13)) / (cot(dphase12_model) - cot(dphase13_model))
        beta2 = beta2_model * (cot(dphase12) + cot(dphase23)) / (cot(dphase12_model) + cot(dphase23_model))
        beta3 = beta3_model * (cot(dphase23) - cot(dphase13)) / (cot(dphase23_model) - cot(dphase13_model))
        return beta1, beta2, beta3

    @staticmethod
    def from_SVD(U, S, V):
        Vf = U @ np.diag(S)
        return Vf[:, 0] ** 2 + Vf[:, 1] ** 2

    @staticmethod
    def from_amplitudes(data: List[np.ndarray]):
        return np.array([np.mean(np.abs(v - np.mean(v))) for v in data])

    @staticmethod
    def sigma_from_emittance(beta, emittance, dispersion, deltaE):
        """ Compute beam size from twiss parameters and emittance """
        return np.sqrt(beta*emittance + (dispersion*deltaE)**2)


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
        ans = ans * (1 + c3 * np.arange(len(ans)))
        return ans

    @staticmethod
    def budkerfit_norm(xdata: np.ndarray,
                       tau: float, c2: float, freq: float, ofsx: float, ofsy: float, c3: float):
        xo = (xdata - ofsx)  # * (1 + c3 * np.arange(len(xdata)))
        ans = np.exp(-tau * (xo ** 2) - c2 * (1 - np.cos(freq * xo))) + ofsy
        ans = ans * (1 + c3 * np.arange(len(ans)))
        return ans / np.max(ans)

    @staticmethod
    def budkerfit_norm2(xdata: np.ndarray,
                        tau: float, c2: float, freq: float, ofsx: float, ofsy: float, c3: float, phase: float):
        xo = (xdata - ofsx)  # * (1 + c3 * np.arange(len(xdata)))
        ans = np.exp(-tau * (xo ** 2) - c2 * (1 - np.cos(freq * xo + phase))) + ofsy
        ans = ans * (1 + c3 * np.arange(len(ans)))
        return ans / np.max(ans)

    @staticmethod
    # @jit(nopython=True, fastmath=True, nogil=True)
    def coupled_envelope(x: np.ndarray,
                         amplitude: float, c1: float, c2: float, c3: float, nu: float, ofsx: float, ofsy: float):
        """
        Envelope that includes chromaticity and octupolar nonlinear decoherence, multiplied by coupling cosine envelope
        In other words, a 2D model with additional coupling 'beating' envelope
        """
        xsc = (x - ofsx) * c2
        xscsq = xsc ** 2
        # ans = amplitude * (1 / (1 + xscsq)) * np.exp(-xscsq * (c1 ** 2) / (1 + xscsq)) * np.exp(-((np.sin(nu * x)) * (c3 / nu)) ** 2) + ofsy
        ans = amplitude * (1 / (1 + xscsq)) * np.exp(-xscsq * c1 / (1 + xscsq)) * np.exp(
            -((np.sin(nu * x)) * (c3 / nu)) ** 2) + ofsy
        # ans = ans * (1+c3*np.arange(len(ans)))
        return ans

    def find_envelope(self,
                      data_raw: np.ndarray,
                      normalize: bool = False,
                      p0=None,
                      lu=None,
                      full=False):
        if np.any(np.isnan(data_raw)):
            raise Exception(f'Some of data is NaN!')
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
            # print(popt)
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
            return env_return, amplitude_envelope + data_mean, popt, pcov
        else:
            return env_return

    def normalize_bpm_signal(self, data, p0=None, lu=None):
        data_mean = np.mean(data)
        envelope = self.find_envelope(data, normalize=True, p0=p0, lu=lu)
        return (data - data_mean) / envelope + data_mean

    def normalize_bpm_signal_v2(self, data: Kick, p0=None, lu=None):
        bpms = data.get_bpms(['H', 'V'])

        data_mean = np.mean(data)
        envelope = self.find_envelope(data, normalize=True, p0=p0, lu=lu)
        return (data - data_mean) / envelope + data_mean


class SVD:

    def __init__(self, data_trim: Tuple = None, output_trim: Tuple = None):
        self.data_trim = data_trim
        self.output_trim = output_trim

    def clean_kick_2D(self,
                      kick: Kick,
                      n_comp: int = 5,
                      families: List[str] = None,
                      use_kick_trim: bool = True):
        """
        Clean kick using SVD, reconstructing each BPM from specified number of components
        """
        families = families or ['H', 'V', 'S']
        assert kick.__class__.__name__ == 'Kick'  # assert isinstance(kick, Kick)
        for family in families:
            U, S, V, vh = self.decompose2D(kick, plane=family, use_kick_trim=use_kick_trim)
            signal = U[:, :n_comp] @ np.diag(S[:n_comp]) @ vh[:n_comp, :]
            bpms = kick.get_bpms(family)
            for i, b in enumerate(bpms):
                if b + kick.Datatype.ORIG.value not in kick.df.columns:
                    kick.set(b + kick.Datatype.ORIG.value, kick.get_bpm_data(b).copy())
                else:
                    raise Exception('Double cleaning!')
                kick.set(b + kick.Datatype.RAW.value, signal[i, :].copy())

    def decompose2D(self, data: Kick, plane: str = None, use_kick_trim: bool = True):
        """
        Decompose any matrix-castable object using SVD
        :param data:
        :param plane:
        :param use_kick_trim:
        :return:
        """
        if data.__class__.__name__ == 'Kick':
            # isinstance(data, Kick):
            # matrix = data.get_bpm_matrix(plane)
            if self.data_trim and not use_kick_trim:
                matrix = data.get_bpm_data(family=plane, return_type='matrix', no_trim=True)
                matrix = matrix[:, self.data_trim]
            else:
                matrix = data.get_bpm_data(family=plane, return_type='matrix')
        elif isinstance(data, np.ndarray):
            matrix = data
            if self.data_trim:
                matrix = matrix[:, self.data_trim]
        else:
            raise Exception(f'Unknown data type: {data.__class__.__name__}')

        matrix = matrix - np.mean(matrix, axis=1)[:, np.newaxis]
        U, S, vh = np.linalg.svd(matrix, full_matrices=False)
        V = vh.T  # transpose it back to conventional U @ S @ V.T
        return U, S, V, vh

    def decompose_kick_2D(self, kick: Kick, tag: str = 'SVD', use_kick_trim: bool = True,
                          add_virtual_bpms: bool = True, families: List[str] = None):
        """
        Decompose kick using SVD and store results
        :param add_virtual_bpms: Whether to add resulting components as virtual BPMs
        :param use_kick_trim: Whether to respect kick data trims
        :param kick: Kick to process
        :param tag: Column name tag
        :return:
        """
        families = families or ['H', 'V']
        assert kick.__class__.__name__ == 'Kick'  # assert isinstance(kick, Kick)
        for family in families:
            # matrix = kick.get_bpm_matrix(plane)
            U, S, V, vh = self.decompose2D(kick, plane=family, use_kick_trim=use_kick_trim)
            # kick.df[f'{tag}_{plane}_M0'] = vh[0, :]
            # kick.df[f'{tag}_{plane}_M1'] = vh[1, :]
            kick.df[f'{tag}_{family}_U'] = [U]
            kick.df[f'{tag}_{family}_S'] = [S]
            kick.df[f'{tag}_{family}_vh'] = [vh]
            if add_virtual_bpms:
                kick.bpms_add([f'SVD2D_{family}_1C', f'SVD2D_{family}_2C'])
                kick.df[f'SVD2D_{family}_1C'] = [vh[0, :]]
                kick.df[f'SVD2D_{family}_2C'] = [vh[1, :]]

    def get_data_2D(self, kick: Kick, family: str, tag: str = 'SVD'):
        U = kick.get(f'{tag}_{family}_U')
        S = kick.get(f'{tag}_{family}_S')
        vh = kick.get(f'{tag}_{family}_vh')
        V = vh.T
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
            # matrix = data.get_bpm_matrix(plane)
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
        # kick.df[f'{tag}_{plane}_M0'] = [vh[0, :]]
        # kick.df[f'{tag}_{plane}_M1'] = [vh[1, :]]
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


class Interpolator:

    def __init__(self, ratio: int = 10):
        self.ratio = ratio

    def __call__(self, *args, **kwargs):
        return self.interpolate(*args, **kwargs)

    def interpolate(self, x: np.ndarray = None, y: np.ndarray = None,
                    ratio: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cubic spline interpolation of the signal (aka upsampling)
        :param x:
        :param y:
        :param ratio: Ratio of how much to increase the sampling rate by
        :return:
        """
        import scipy.interpolate

        assert y is not None
        if x is None:
            x = np.arange(len(y))
        assert len(x) == len(y) and len(x) > 3
        delta = np.unique(np.diff(x))
        assert len(delta) == 1  # Check if spacing uniform

        ratio = ratio or self.ratio
        p = scipy.interpolate.CubicSpline(x, y)
        x_new = np.linspace(x[0], x[-1], len(x) * ratio, endpoint=True)
        return x_new, p(x_new)

    def interpolate_kick(self, kick: Kick, ratio: int = None, families=('H', 'V')):
        assert kick.__class__.__name__ == 'Kick'  # assert isinstance(kick, Kick)
        ratio = ratio or self.ratio
        for family in families:
            data = kick.get_bpm_data(family=family, no_trim=True, return_type='dict')
            for b, v in data.items():
                x, y = self.interpolate(None, v, ratio=ratio)
                kick.set(f'{b}{Kick.Datatype.INTERPX.value}', x)
                kick.set(f'{b}{Kick.Datatype.INTERPY.value}', y)


class Errors:
    @staticmethod
    def fft_errors_from_bpm_noise(N, amplitude, sigma_noise):
        """
        Statistical error propagation from BPM noise for amplitude and phase of signal
        N - number of turns/samples
        Ref: CERN-SL-96-070-BI
        """
        sigma_amp = np.sqrt(2 / N) * sigma_noise
        sigma_phase = (1 / amplitude) * np.sqrt(2 / N) * sigma_noise
        return sigma_amp, sigma_phase

    @staticmethod
    def twiss_errors_from_bpm_noise(dphase12, dphase13, dphase23, sig_phase1, sig_phase2, sig_phase3):
        """
        Statistical error of beta1 from 3BPM method based on phase uncertainty
        Ref: CERN-SL-96-070-BI
        """
        cot = np.arctan
        v = (cot(dphase12) + cot(dphase13)) ** 2 * sig_phase1 ** 2 + \
            (cot(dphase12) + cot(dphase23)) ** 2 * sig_phase2 ** 2 + \
            (cot(dphase23) - cot(dphase13)) ** 2 * sig_phase3 ** 2
        return np.sqrt(v)
