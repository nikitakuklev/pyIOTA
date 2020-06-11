from typing import Union, Tuple

import numpy as np, scipy as sc
import numba
from numba import jit
import numexpr as ne
import scipy.integrate as sci
import scipy.signal
from scipy.optimize import minimize


# @jit(nopython=True, fastmath=True, nogil=True)
def _get_integral_v11(ztabs, twin, i_line, coeffs, FR, turns, order):
    """Numba-optimized trapezoidal integrator"""
    # ZTF = np.zeros(len(ztabs), np.complex128)
    # ZTF[0] = ztabs[0]*twin[0]
    ZTF = ztabs[1:] * twin[1:] * np.exp(-2.0 * (i_line) * np.pi * 1.0j * FR)
    integral = (ztabs[0] * twin[0] + ZTF[-1]) / 2 + np.sum(ZTF[:-2])
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
        power = power or self.window_power
        n_turns = len(data)
        if (n_turns, power) not in self.hann_cache:
            window = np.hanning(n_turns) ** power
            self.hann_cache[(n_turns, power)] = window
        else:
            window = self.hann_cache[(n_turns, power)]

        data_centered = data - np.mean(data)
        fft_power = np.abs(np.fft.rfft(data_centered * window)) ** 2
        fft_freq = np.fft.rfftfreq(n_turns)

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
            return np.fft.rfftfreq(n_turns)[np.argmax(fft_power)]

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

    def fft(self, data: np.ndarray, window_power: int = None, pad_zeros_power: int = None,
            output_trim=None, data_trim=None) -> Tuple[np.ndarray,np.ndarray]:
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
            if n_turns < len_padded:
                # print(f'To pad: {len_padded - n_turns} (have {data_centered.shape} turns)')
                data_centered = np.pad(data_centered, ((0, len_padded - n_turns),))
                n_turns = len(data_centered)
                assert n_turns == len_padded
                # print(f'Padded to {n_turns} points')

        fft_power = np.abs(np.fft.rfft(data_centered)) ** 2
        fft_freq = np.fft.rfftfreq(n_turns)

        if output_trim:
            # fft_power = fft_power[trim[0]:trim[1]]
            # fft_freq = fft_freq[trim[0]:trim[1]]
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

    def calc_correlation_amplitude(self, data: np.ndarray, freq: float, turns: int = None, order: int = 6,
                                   method: int = 11, only_amplitude: bool = False):

        """
        Calculate signal correlation at a single frequency
        :param data:
        :param freq:
        :param turns:
        :param order:
        :param method:
        :param only_amplitude:
        :return:
        """
        # print(f)
        res = self.compute_correlations(data, [freq], turns=turns, integrator_order=order, method=method)[0]
        if only_amplitude:
            return res[0]
        else:
            return res

    def compute_correlations(self, data: np.ndarray, frequencies: list, turns=None,
                             skip_turns=0,
                             window_order=1, integrator_order=6, method=11,
                             data_trim=None):
        """
        Calculate correlation with a list of frequencies
        :param data_trim:
        :param data:
        :param frequencies:
        :param turns:
        :param skip_turns:
        :param window_order:
        :param integrator_order:
        :param method:
        :return:
        """
        data_trim = data_trim or self.data_trim
        if data_trim:
            data = data[data_trim]

        if not turns:
            turns = len(data) - 1

        turns_naff = self.__calculate_naff_turns(data, turns, integrator_order)
        data = data[skip_turns:skip_turns + turns_naff + 1]

        if (turns_naff, window_order) not in self.window_cache:
            T = np.linspace(0, turns_naff, num=turns_naff + 1, endpoint=True) * 2.0 * np.pi - np.pi * turns_naff
            TWIN = ((2.0 ** window_order * np.math.factorial(window_order) ** 2) / float(
                np.math.factorial(2 * window_order))) * (
                           1.0 + np.cos(T / turns_naff)) ** window_order
            self.window_cache[(turns_naff, window_order)] = TWIN
        else:
            TWIN = self.window_cache[(turns_naff, window_order)]

        if turns_naff not in self.linspace_cache:
            i_line = np.linspace(1, turns_naff, num=turns_naff, endpoint=True)
            self.linspace_cache[turns_naff] = i_line
        else:
            i_line = self.linspace_cache[turns_naff]

        integral = []
        for f in frequencies:
            if method == 11:
                integral.append(_get_integral_v11(data, TWIN, i_line,
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

        return integral

    def run_naff(self, data: np.ndarray, n_points: int, n_skip: int = 0,
                 legacy: bool = False, xatol: float = 1e-6):
        if len(data) < n_points + n_skip:
            raise Exception(f"Number of points+skips exceeds available data length ({len(data)})")
        sample = data[n_skip:n_skip + n_points + 1]
        sample -= np.mean(sample)
        tune0 = self.fft_hanning_peaks(sample, power=1, search_peaks=False)
        print(f'Tune initial guess: {tune0}')

        def get_amplitude(freq: float, data: np.ndarray, order: int, method: int):
            return self.compute_correlations(data, [freq], integrator_order=order, method=method)[0]

        res = minimize(lambda *args: -1 * get_amplitude(*args),
                       np.array(tune0),
                       args=(sample, 6, 11),
                       method='nelder-mead',
                       options={'disp': False,
                                'initial_simplex': np.array([[tune0 - 1e-3], [tune0 + 1e-3]]),
                                'xatol': xatol,
                                'fatol': 1})

        tune = res.x[0]
        if legacy:
            return [[0, tune, 0, 0, 0]]
        else:
            return tune
