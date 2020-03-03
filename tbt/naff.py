import numpy as np, scipy as sc
import numba
from numba import jit
import numexpr as ne
import scipy.integrate as sci
import scipy.signal
from scipy.optimize import minimize


@jit(nopython=True, fastmath=True, nogil=True)
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
    coeffs_cache = [sci.newton_cotes(i, 1)[0].astype(np.complex128) for i in range(1, 10)]
    window_cache = {}
    linspace_cache = {}
    hann_cache = {}

    def __init__(self, window_type=None, window_power=None):
        self.window_type = window_type
        self.window_power = window_power

    def fft_hanning(self, data: np.ndarray, power: int = None, search_peaks: bool = False):
        """
        Preliminary guess of peak frequency based on FTT with hanning window
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



    def calc_correlation_amplitude(self, data: np.ndarray, freq: float, turns=None, order=6, method=11,
                                   only_amplitude=False):
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

    def compute_correlations(self, data: np.ndarray, frequencies: list, turns=None, skip_turns=1,
                             window_order=1, integrator_order=6, method=11):
        """
        Calculate correlation with a list of frequencies
        :param data:
        :param frequencies:
        :param turns:
        :param skip_turns:
        :param window_order:
        :param integrator_order:
        :param method:
        :return:
        """
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

    def run_naff(self, data: np.ndarray, n_points: int, n_skip: int = 1,
                 legacy: bool = False, xatol: float = 1e-6):
        if len(data) < n_points + n_skip:
            raise Exception(f"Number of points+skips exceeds available data length ({len(data)})")
        sample = data[n_skip:n_skip + n_points + 1]
        sample -= np.mean(sample)
        tune0 = self.fft_hanning(sample, power=1, search_peaks=False)

        def get_amplitude(freq: float, data: np.ndarray, order: int, method: int):
            return self.compute_correlations(data, [freq], integrator_order=order, method=method)[0]

        res = minimize(lambda *args: -get_amplitude(*args),
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

