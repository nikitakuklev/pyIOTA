import logging, time
from typing import Union, Tuple, List

import numpy as np, scipy as sc
import numba
from numba import jit
import numexpr as ne
import scipy.integrate as sci
import scipy.signal
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# @jit(nopython=True, fastmath=True, nogil=True)
def _get_integral_v11(ztabs, twin, i_line, coeffs, FR, turns, order):
    """
    NATIVE
    Trapezoidal integrator = f[0]/2 + f[-1]/2 + f[1:-1]
    We don't care about absolute values, so deltax = 1, which simplifies calcs
    """
    # ZTF = np.zeros(len(ztabs), np.complex128)
    # ZTF[0] = ztabs[0]*twin[0]
    ZTF = ztabs[1:] * twin[1:] * np.exp(-1.0j * 2 * np.pi * FR * i_line)
    integral = (ztabs[0] * twin[0] + ZTF[-1]) / 2 + np.sum(ZTF[:-2])
    A = np.real(integral)
    B = np.imag(integral)
    RMD = np.abs(integral)
    return RMD, A, B


@jit(nopython=True, fastmath=True, nogil=True)
def _get_integral_v14(signal, window, i_line, coeffs, tune, turns, order):
    """
    NUMBA
    Dumb inner product integrator (fastest, not as accurate)
    """
    i_line = np.arange(0, len(signal))
    integral = np.sum(signal * window * np.exp(-1.0j * 2 * np.pi * tune * i_line))
    A = np.real(integral)
    B = np.imag(integral)
    RMD = np.abs(integral)
    return RMD, A, B


def _get_integral_v15(signal, window, i_line, coeffs, tune, turns, order):
    """
    NUMEXPR
    Dumb inner product integrator (fastest, not as accurate)
    """
    i_line = np.arange(0, len(signal))
    pi = np.pi
    integral = ne.evaluate("sum(signal * window * exp(-1.0j * 2 * pi * tune * i_line))")
    A = np.real(integral)
    B = np.imag(integral)
    RMD = np.abs(integral)
    return RMD, A, B


class NAFF:
    window_cache = {}
    linspace_cache = {}
    hann_cache = {}

    def __init__(self, window_type=None, window_power: int = 0, fft_pad_zeros_power=None, data_trim=None,
                 output_trim=None):
        self.window_type = window_type
        self.window_power = window_power
        self.data_trim = data_trim
        self.fft_pad_zeros_power = fft_pad_zeros_power
        self.output_trim = output_trim
        self.coeffs_cache = [sci.newton_cotes(i, 1)[0].astype(np.complex128) for i in range(1, 10)]
        self.naff_runtimes = []

    def fft_hanning_peaks(self, data: np.ndarray, power: int = None, just_do_fft: bool = False,
                          search_peaks: bool = False):
        """
        Preliminary guess of peak frequency based on FTT with hanning window
        :param just_do_fft:
        :param data:
        :param power:
        :param search_peaks: whether to use scipy peak finding or just return highest bin
        :return:
        """
        assert isinstance(data, np.ndarray) and data.ndim == 1
        # print(data.shape)
        fft_freq, fft_power = self.fft(data,
                                       window_power=power,
                                       pad_zeros_power=self.fft_pad_zeros_power,
                                       data_trim=np.s_[:],
                                       output_trim=np.s_[:])

        if just_do_fft:
            return fft_freq, fft_power

        if search_peaks:
            peak_idx, peak_props = sc.signal.find_peaks(fft_power)
            peak_tunes = fft_freq[peak_idx]
            if len(peak_idx) > 0:
                top_tune = peak_tunes[np.argmax(fft_power[peak_idx])]
            else:
                top_tune = None
            return top_tune, fft_freq, fft_power, peak_idx
        else:
            return fft_freq[np.argmax(fft_power)]

    def fft_peaks(self, data: np.ndarray = None, search_peaks: bool = False, search_kwargs: dict = None,
                  fft_freq: np.ndarray = None, fft_power: np.ndarray = None):
        """
        Preliminary guess of peak frequency based on FFT
        :param fft_freq:
        :param fft_power:
        :param search_kwargs:
        :param data:
        :param search_peaks: whether to use scipy peak finding or just return highest bin
        :return:
        """
        if not search_kwargs:
            search_kwargs = {'prominence': 1 / 8, 'distance': 1 / 70}
        if fft_freq is None or fft_power is None:
            if data is not None:
                fft_freq, fft_power = self.fft(data)
            else:
                raise Exception('Missing required data')
        if search_peaks:
            peak_idx, peak_props = sc.signal.find_peaks(fft_power,
                                                        prominence=np.max(fft_power) * search_kwargs['prominence'],
                                                        distance=len(fft_power) * search_kwargs['distance']
                                                        )
            peak_tunes = fft_freq[peak_idx]
            if len(peak_idx) > 0:
                top_tune = peak_tunes[np.argmax(fft_power[peak_idx])]
            else:
                top_tune = None
            return top_tune, peak_tunes, peak_idx, peak_props, (fft_freq, fft_power)
        else:
            return fft_freq[np.argmax(fft_power)]

    def fft(self,
            data: np.ndarray,
            window_power: int = None,
            pad_zeros_power: int = None,
            output_trim: slice = None,
            data_trim: slice = None) -> Tuple[np.ndarray, np.ndarray]:
        pad_zeros_power = pad_zeros_power or self.fft_pad_zeros_power
        window_power = window_power or self.window_power
        output_trim = output_trim or self.output_trim
        data_trim = data_trim or self.data_trim

        if data_trim:
            data = data[data_trim]
        data_centered = data - np.mean(data)
        n_turns = len(data_centered)

        # Windowing must happen on x[n], original data
        if window_power == 0:
            window = np.ones_like(data_centered)
        else:
            if (n_turns, window_power) not in self.hann_cache:
                window = np.hanning(n_turns) ** window_power
                self.hann_cache[(n_turns, window_power)] = window
            else:
                window = self.hann_cache[(n_turns, window_power)]
        data_centered = data_centered * window

        if pad_zeros_power is not None:
            len_padded = 2 ** pad_zeros_power
            # if n_turns < len_padded:
            #     # print(f'To pad: {len_padded - n_turns} (have {data_centered.shape} turns)')
            #     data_centered = np.pad(data_centered, ((0, len_padded - n_turns),))
            #     n_turns = len(data_centered)
            #     assert n_turns == len_padded
            #     # print(f'Padded to {n_turns} points')
        else:
            len_padded = n_turns
        fft_power = np.abs(np.fft.fft(data_centered, n=len_padded)) ** 2
        fft_freq = np.fft.fftfreq(len_padded)
        # Keep half
        fft_power = fft_power[fft_freq > 0]
        fft_freq = fft_freq[fft_freq > 0]

        if output_trim:
            fft_power = fft_power[output_trim]
            fft_freq = fft_freq[output_trim]

        return fft_freq, fft_power

    def __calculate_naff_turns(self, data, turns, order):
        """
        Returns max number of turns that can be used for NAFF, depending on integrator and order
        :param data:
        :param turns:
        :param order:
        :return:
        """
        if turns >= len(data) + 1:
            raise ValueError('#naff : Input data must be at least of length turns+1')

        if turns < order:
            raise ValueError('#naff : Minimum number of turns is {}'.format(order))

        if np.mod(turns, order) != 0:
            a, b = divmod(turns, order)
            turns = int(order * a)
        return turns

    # def calc_correlation_amplitude(self, data: np.ndarray, freq: float, turns: int = None, order: int = 6,
    #                                method: int = 11, only_amplitude: bool = False):
    #
    #     """
    #     Calculate signal correlation at a single frequency
    #     :param data:
    #     :param freq:
    #     :param turns:
    #     :param order:
    #     :param method:
    #     :param only_amplitude:
    #     :return:
    #     """
    #     # print(f)
    #     res = self.compute_correlations(data, [freq], turns=turns, integrator_order=order, method=method)[0]
    #     if only_amplitude:
    #         return res[0]
    #     else:
    #         return res

    def compute_correlations(self,
                             data: np.ndarray,
                             frequencies: Union[List[float], float, int],
                             turns: int = None,
                             skip_turns: int = 0,
                             window_order: int = 1,
                             integrator_order: int = 6,
                             method: int = 11,
                             data_trim: slice = None,
                             no_trim: bool = False,
                             magnitude_only: bool = True):
        """
        Calculate NAFF correlation with a list of frequencies
        :param data: Signal data array
        :param frequencies: Frequencies to compute correlations for
        :param no_trim: Override all trims and work on raw data
        :param data_trim:
        :param turns:
        :param skip_turns:
        :param window_order:
        :param integrator_order:
        :param method: Method to calculate numeric integral
        :param magnitude_only:
        :return:
        """
        single_freq_mode = False
        if isinstance(frequencies, float) or isinstance(frequencies, int):
            frequencies = [frequencies]
            single_freq_mode = True

        if not no_trim:
            data_trim = data_trim or self.data_trim
            if data_trim:
                if data_trim.stop is not None and data_trim.stop > len(data):
                    raise Exception(f"Trim end ({data_trim.stop}) exceeds available data length ({len(data)})")
                data = data[data_trim]

        # Check for deprecated parameters, temp hack
        if turns or skip_turns:
            raise Exception('Deprecated parameters used!')

        if not turns:
            turns = len(data) - 1

        turns_naff = self.__calculate_naff_turns(data, turns, integrator_order)
        data = data[skip_turns:skip_turns + turns_naff + 1]

        if (turns_naff, window_order) not in self.window_cache:
            # Hanning window
            T = np.linspace(0, turns_naff, num=turns_naff + 1, endpoint=True) * 2.0 * np.pi - np.pi * turns_naff
            TWIN = ((2.0 ** window_order * np.math.factorial(window_order) ** 2) / float(
                np.math.factorial(2 * window_order))) * (1.0 + np.cos(T / turns_naff)) ** window_order
            self.window_cache[(turns_naff, window_order)] = TWIN
        else:
            TWIN = self.window_cache[(turns_naff, window_order)]

        if turns_naff not in self.linspace_cache:
            # Just 1...N array
            i_line = np.linspace(1, turns_naff, num=turns_naff, endpoint=True)
            self.linspace_cache[turns_naff] = i_line
        else:
            i_line = self.linspace_cache[turns_naff]

        integral = []
        tic = time.perf_counter()
        for f in frequencies:
            if method == 11:
                integral.append(_get_integral_v11(data, TWIN, i_line,
                                                  self.coeffs_cache[integrator_order - 1],
                                                  f, turns_naff, integrator_order))
            elif method == 14:
                integral.append(_get_integral_v14(data, TWIN, i_line,
                                                  self.coeffs_cache[integrator_order - 1],
                                                  f, turns_naff, integrator_order))
            elif method == 15:
                integral.append(_get_integral_v15(data, TWIN, i_line,
                                                  self.coeffs_cache[integrator_order - 1],
                                                  f, turns_naff, integrator_order))
            # elif method == 3:
            #     integral.append(getIntegralv3(data, TWIN, f, turns_naff, order))
            # elif method == 4:
            #     integral.append(getIntegralv4(data, TWIN, f, turns_naff, order))
            # elif method == 5:
            #     integral.append(getIntegralv5(data, TWIN, f, turns_naff, order))
            # elif method == 6:
            #     integral.append(getIntegralv6(data, TWIN, i_line, coeffs_cache[order - 1], f, turns_naff, order))
            # elif method == 7:
            #     integral.append(getIntegralv7(data, TWIN, i_line, coeffs_cache[order - 1], f, turns_naff, order))
            # elif method == 8:
            #     integral.append(getIntegralv8(data, TWIN, i_line, coeffs_cache[order - 1], f, turns_naff, order))
            # elif method == 9:
            #     integral.append(getIntegralv9(data, TWIN, i_line, coeffs_cache[order - 1], f, turns_naff, order))
            # elif method == 10:
            #     integral.append(getIntegralv10(data, TWIN, i_line, coeffs_cache[order - 1], f, turns_naff, order))
            #
            # elif method == 12:
            #     integral.append(getIntegralv12(data, TWIN, i_line, coeffs_cache[order - 1], f, turns_naff, order))
            # elif method == 13:
            #     integral.append(getIntegralv13(data, TWIN, i_line, coeffs_cache[order - 1], f, turns_naff, order))
            else:
                raise Exception("incorrect method selected!")
        self.naff_runtimes.append((time.perf_counter() - tic) / len(frequencies))
        if single_freq_mode:
            return integral[0]
        else:
            if magnitude_only:
                return np.array(integral)[:, 0]
            else:
                return np.array(integral)

    def get_projection(self, u, v):
        """
        Get inner product projection of two vectors - <u,v>/<u,u>
        """
        # <v,v> = 1 * len(v) because frequency vector has entries of magnitude 1
        return np.sum(np.conjugate(u) * v) / len(u)

    def reset_perf_counters(self):
        self.naff_runtimes = []

    def run_naff(self, data: np.ndarray, data_trim=None, legacy: bool = False, xatol: float = 1e-6,
                 n_components: int = 2, full_data: bool = True):
        """
        Numeric analysis of fundamental frequencies
        Iterative method that maximizes inner product of data and oscillatory signal, finding best frequencies
        :param data: Signal
        :param data_trim: Optional trim override
        :param legacy: Use old return format
        :param xatol: Final optimization tolerance
        :param n_components: Number of frequencies to look for
        :param full_data: Return debug and other information in addition to tunes
        :return:
        """
        data_trim = data_trim or self.data_trim
        if data_trim:
            if data_trim.stop is not None and data_trim.stop > len(data):
                raise Exception(f"Trim end ({data_trim.stop}) exceeds available data length ({len(data)})")
            data = data[data_trim]
        data_centered = data - np.mean(data)
        n_turns = len(data_centered)

        last_eval = [0, 0, 0]

        def get_amplitude(freq: np.ndarray, signal: np.ndarray, order: int, method: int):
            nonlocal last_eval
            last_eval = self.compute_correlations(signal, freq[0], integrator_order=order, method=method, no_trim=True)
            return last_eval[0]

        freq_components = []
        results = []
        tunes = []
        for i in range(1, n_components + 1):
            tune0 = self.fft_hanning_peaks(data_centered, power=1, search_peaks=False)
            # logger.info(f'Component ({i}) - tune initial guess: {tune0}')

            res = minimize(lambda *args: -1 * get_amplitude(*args),
                           tune0,
                           args=(data_centered, 6, 15),
                           method='nelder-mead',
                           options={'disp': False,
                                    'initial_simplex': np.array([[tune0 - 1e-3], [tune0 + 1e-3]]),
                                    'xatol': xatol,
                                    'fatol': 1})

            tune = res.x[0]
            # Orthogonalize if not the first frequency and far enough
            if np.any(np.abs(np.array(tunes) - tune) < 1e-4):
                # Close frequency wasnt removed last time, remove without orthogonalization
                fc = (1.0 * last_eval[1] + 1.0j * last_eval[2]) * np.exp(1.0j * 2 * np.pi * tune * np.arange(n_turns))
                logger.warning(
                    f'Close frequency ({tune:.6f}) found (eps={np.min(np.abs(np.array(tunes) - tune)):.6f}), removing w/o orthogonalization')
                # raise Exception
            else:
                fc = np.exp(1.0j * 2 * np.pi * tune * np.arange(n_turns))
                for j in range(2, i):
                    # print(i,j, fc[0:5])
                    fc = fc - self.get_projection(freq_components[j - 1], fc) * freq_components[j - 1]
            tunes.append(tune)
            freq_components.append(fc)
            amplitude = self.get_projection(fc, data_centered)
            results.append([res.x[0], data_centered.copy(), tune0, fc, amplitude, res])
            data_centered = data_centered - amplitude * fc

        if legacy:
            return [[0, res[0], 0, 0, 0] for res in results]
        elif full_data:
            return results
        else:
            return [res[0] for res in results]
