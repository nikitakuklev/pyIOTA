import logging
import warnings

import math
import time
from typing import Union, Tuple, List, Dict

import numpy as np, scipy as sc
import scipy.integrate as sci
import scipy.signal
from scipy.optimize import minimize, minimize_scalar

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


def _get_integral_v14_wrap():
    from numba import jit

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
    return _get_integral_v14


def _get_integral_v15_wrap():
    import numexpr as ne

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
    return _get_integral_v15


class NAFF:
    """
    Numerical analysis of fundamental frequencies - a method for improving tune resolution.
    This object does many frequency-analysis related tasks.
    It can be faster for repeated use, since computations like windows are cached.

    Uses:
    FFT - simple fft
    Zero-padded FFT - effectively interpolated FFT with little performance impact
    NAFF - computation of amplitudes/phases
    Peak finding and selection
    """
    window_cache = {}
    linspace_cache = {}
    hann_cache = {}

    def __init__(self,
                 window_type: str = 'hann2',
                 window_power: int = 0,
                 window_opts: Dict = None,
                 wp: int = None,
                 fft_pad_zeros_power: int = None,
                 pp: int = None,
                 pc: int = None,
                 data_trim: slice = None,
                 output_trim: slice = None
                 ):
        """
        Create new NAFF object that can be passed to various methods or used directly

        :param window_type: The type of window to use - 'hann', 'hann2', 'halfhann', 'tukey'.
        :param window_power: Window power - there is theoretically better resolution at higher powers, but only on
        near perfect data. For experimental sources, 1 is ok.
        :param fft_pad_zeros_power: Optional FFT zero-pad to length 2**pad_power - can be used to find tunes directly,
        and also improves initial tune guess and convergence speed for NAFF
        :param pp: same as pad power
        :param pc: point count
        :param data_trim: Slice object to apply to all data inputs - useful to remove data before/after kick
        :param output_trim: Slice object to apply to all output (freq, freq_power) - useful to remove 0 frequency
        """
        self.window_type = window_type
        self.window_power = wp or window_power
        self.window_opts = window_opts
        self.data_trim = data_trim
        self.fft_pad_zeros_power = pp or fft_pad_zeros_power
        self.fft_pad_zeros = pc
        self.output_trim = output_trim
        self.coeffs_cache = [sci.newton_cotes(i, 1)[0].astype(np.complex128) for i in range(1, 10)]
        self.icache = {}
        self.naff_runtimes = []

    def fft(self,
            data: np.ndarray,
            spacing: float = 1.0,
            window_power: int = None,
            pad_zeros_power: int = None,
            pc: int = None,
            output_trim: slice = None,
            data_trim: slice = None,
            amplitude: bool = True
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        It is me, your personal FFT. Really...just FFT, so I don't have to rewrite same thing 10 times.

        :param spacing:
        :param data: Data
        :param window_power: Uses object default if not specified
        :param pad_zeros_power: Uses object default if not specified
        :param output_trim: Uses object default if not specified
        :param data_trim: Uses object default if not specified
        :return:
        """
        pad_zeros_power = pad_zeros_power or self.fft_pad_zeros_power
        pc = pc or self.fft_pad_zeros
        window_power = window_power or self.window_power
        output_trim = output_trim or self.output_trim
        data_trim = data_trim or self.data_trim

        if data_trim:
            data = data[data_trim]
        data_centered = data - np.mean(data)
        n_turns = len(data_centered)

        # Windowing must happen on x[n], original data
        window = self.window(n_turns, window_power)
        data_centered = data_centered * window

        if pc is not None:
            len_padded = pc
        else:
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

        if amplitude:
            # fft_power = np.abs(np.fft.fft(data_centered, n=len_padded)) ** 2 / (n_turns * n_turns)
            fft_power = np.abs(
                np.fft.fft(data_centered, n=len_padded)) / n_turns  # (n_turns * n_turns)
        else:
            fft_power = np.fft.fft(data_centered, n=len_padded) / n_turns
        fft_freq = np.fft.fftfreq(len_padded, d=spacing)
        # Keep half
        fft_power = fft_power[fft_freq >= 0]
        fft_freq = fft_freq[fft_freq >= 0]

        if output_trim:
            fft_power = fft_power[output_trim]
            fft_freq = fft_freq[output_trim]

        return fft_freq, fft_power

    def fft_hanning_peaks(self, data: np.ndarray, window_power: int = None,
                          just_do_fft: bool = False,
                          search_peaks: bool = False
                          ):
        """
        DEPRECATED
        Preliminary guess of peak frequency based on FFT with hanning window
        :param just_do_fft:
        :param data:
        :param window_power:
        :param search_peaks: whether to use scipy peak finding or just return highest bin
        :return:
        """
        assert isinstance(data, np.ndarray) and data.ndim == 1
        # print(data.shape)
        fft_freq, fft_power = self.fft(data,
                                       window_power=window_power,
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

    def fft_peaks(self, data: np.ndarray = None,
                  search_peaks: bool = False, search_kwargs: dict = None,
                  fft_freq: np.ndarray = None, fft_power: np.ndarray = None
                  ):
        """
        Preliminary guess of peak frequency based on FFT
        :param fft_freq:
        :param fft_power:
        :param search_kwargs:
        :param data:
        :param search_peaks: whether to use scipy peak finding or just return highest bin
        :return:
        """
        from scipy.signal import find_peaks
        if not search_kwargs:
            search_kwargs = {'prominence': 1 / 8, 'distance': 1 / 70}

        if fft_freq is None or fft_power is None:
            if data is not None:
                fft_freq, fft_power = self.fft(data)
            else:
                raise Exception('Either pre-computed FFT data or raw data must be provided')

        if search_peaks:
            peak_idx, peak_props = find_peaks(fft_power,
                                              prominence=np.max(fft_power) * search_kwargs[
                                                  'prominence'],
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

    def run_naff(self, data: np.ndarray, data_trim: slice = None, legacy: bool = False,
                 xatol: float = 1e-6,
                 n_components: int = 2, full_data: bool = True
                 ):
        """
        DEPRECATED - see v2
        Numeric analysis of fundamental frequencies
        Iterative method that maximizes inner product of data and oscillatory signal, finding best frequencies
        :param data: Signal
        :param data_trim: Uses object default if not specified
        :param legacy: Use old return format
        :param xatol: Final optimization absolute tolerance
        :param n_components: Number of frequencies to look for
        :param full_data: Return debug and other information in addition to tunes
        :return:
        """
        warnings.warn("This method is deprecated - use v2!")

        data_trim = data_trim or self.data_trim
        if data_trim:
            if data_trim.stop is not None and data_trim.stop > len(data):
                raise Exception(
                    f"Trim end ({data_trim.stop}) exceeds available data length ({len(data)})")
            data = data[data_trim]
        data_centered = data - np.mean(data)
        n_turns = len(data_centered)

        last_eval = [0, 0, 0]

        def get_amplitude(freq: np.ndarray, signal: np.ndarray, order: int, method: int):
            nonlocal last_eval
            last_eval = self.compute_correlations(signal, freq[0], integrator_order=order,
                                                  method=method, no_trim=True)
            return last_eval[0]

        freq_components = []
        results = []
        tunes = []
        for i in range(1, n_components + 1):
            tune0 = self.fft_hanning_peaks(data_centered, window_power=1, search_peaks=False)
            # logger.info(f'Component ({i}) - tune initial guess: {tune0}')

            res = minimize(lambda *args: -1 * get_amplitude(*args),
                           tune0,
                           args=(data_centered, 6, 15),
                           method='nelder-mead',
                           options={'disp': False,
                                    'initial_simplex': np.array([[tune0 - 1e-3], [tune0 + 1e-3]]),
                                    'xatol': xatol,
                                    'fatol': 1
                                    })
            tune = res.x[0]

            # Orthogonalize if not the first frequency and far enough
            if np.any(np.abs(np.array(tunes) - tune) < 1e-4):
                # Close frequency wasnt removed last time, remove without orthogonalization
                fc = (1.0 * last_eval[1] + 1.0j * last_eval[2]) * np.exp(
                    1.0j * 2 * np.pi * tune * np.arange(n_turns))
                logger.warning(f'NAFF - close frequency ({tune:.6f}) found '
                               f'(eps={np.min(np.abs(np.array(tunes) - tune)):.6f}), removing w/o orthogonalization')
            else:
                fc = np.exp(1.0j * 2 * np.pi * tune * np.arange(n_turns))
                for j in range(2, i):
                    # print(i,j, fc[0:5])
                    fc = fc - self.get_projection(freq_components[j - 1], fc) * freq_components[
                        j - 1]

            tunes.append(tune)
            freq_components.append(fc)
            amplitude = self.get_projection(fc, data_centered)
            results.append([res.x[0], data_centered.copy(), tune0, fc, amplitude, res])
            data_centered = data_centered - amplitude * fc

        if legacy:
            return [[0, r[0], 0, 0, 0] for r in results]
        elif full_data:
            return results
        else:
            return [r[0] for r in results]

    # Found in another implementation, but disagree on amplitude calc?
    # # Predefined functions
    # HANNING = lambda t: np.sin(np.pi * np.arange(t) / (t - 1)) ** 2  # hanning window filter
    # FREQ = lambda t: np.arange(t * 1.0) / t  # norm. freq
    # EXPO = lambda f, n: np.exp(-1j * 2 * np.pi * f * n)  # complex exponential
    # NAFFfreq = lambda f, X, W: -1 * np.abs(np.sum(EXPO(f, np.arange(1, len(X) + 1)) * X * W))  # NAFF freq algorithm
    # NAFFamp = lambda f, X: np.sum(np.real(EXPO(f, np.arange(len(X))) * X)) / len(X) + 1j * np.sum(
    #     np.imag(EXPO(f, np.arange(len(X))) * X)) / len(X)  # NAFF amp algorithm

    def run_naff_v2(self, data: np.ndarray, data_trim: slice = None, xatol: float = 1e-9,
                    n_components: int = 2, full_data: bool = True, spacing: float = 1.0,
                    perf_mode: bool = False, freq_trim: tuple = None,
                    orthogonal: bool = True
                    ):
        """
        Numeric analysis of fundamental frequencies
        Iterative method that maximizes inner product of data and oscillatory signal,
        finding best frequencies

        :param data: signal
        :param data_trim: a slice to trim data with, uses object default if not specified
        :param freq_trim: tuple of (low_limit, high_limit) used as frequency search window
        :param legacy: Use old return format
        :param xatol: final optimization absolute tolerance
        :param n_components: Number of frequencies to look for
        :param full_data: Return debug and other information in addition to tunes
        :param perf_mode: Enables certain optimizations and shortcuts to maximize performance
        :return:
        """
        assert data.dtype == np.float64 or data.dtype == np.float32
        # Preprocessing
        sp = spacing
        data_trim = data_trim or self.data_trim
        if data_trim:
            if data_trim.stop is not None and data_trim.stop > len(data):
                raise Exception(
                    f"Trim end ({data_trim.stop}) exceeds available data length ({len(data)})")
            data = data[data_trim]
        data = data - np.mean(data)  # NOT INPLACE, VERY IMPORTANT
        data = data.astype(np.complex128)
        n_turns = len(data)
        window = self.window(n_turns, self.window_power)

        # Internal functions

        def get_top_freq(signal: np.ndarray):
            """ Get top FFT bin frequency """
            # Cut fft to positive freqs, since we should only have real signal inputs

            if self.fft_pad_zeros_power is not None:
                len_padded = 2 ** self.fft_pad_zeros_power
                len_real = math.floor(len_padded / 2 + 1)
                fft_power = np.abs(np.fft.fft(signal, n=len_padded))[:len_real]
                fft_freq = np.fft.fftfreq(len_padded, d=spacing)[:len_real]
            else:
                len_real = math.floor(len(signal) / 2 + 1)
                fft_power = np.abs(np.fft.fft(signal))[:len_real]
                fft_freq = np.fft.fftfreq(len(signal), d=spacing)[:len_real]
            if freq_trim is not None:
                mask = np.argwhere((fft_freq > freq_trim[0]) & (fft_freq < freq_trim[1])).flatten()
                assert np.any(mask)
                return fft_freq[mask][np.argmax(fft_power[mask])]
            else:
                return fft_freq[np.argmax(fft_power)]

        def get_amplitude(freq: np.ndarray, signal: np.ndarray):
            """ Correlation integral """
            last_eval = self.compute_correlations_v2(signal, freq, no_trim=True, no_window=True)
            return last_eval[0]

        def freq_projection(fc, tunes, projections, n_freqs, window):
            Ai = np.zeros_like(fc, dtype=np.complex128)
            for j in range(0, n_freqs):
                Ai += projections[j] * np.exp(1.0j * 2 * np.pi * tunes[j] * np.arange(n_turns))
            return np.sum(fc * np.conjugate(Ai) * window) / np.sum(Ai * np.conjugate(Ai) * window)

        def signal_projection(signal, tunes, projections, n_freqs, window):
            Ai = np.zeros_like(signal, dtype=np.complex128)
            for j in range(0, n_freqs):
                Ai += projections[j] * np.exp(1.0j * 2 * np.pi * tunes[j] * np.arange(n_turns))
            return np.sum(signal * np.conjugate(Ai) * window) / np.sum(
                Ai * np.conjugate(Ai) * window)

        def remove_frequencies(signal, tunes, normalized_amplitudes, i, window, orthogonal=True):
            """ Remove previous frequencies from signal """
            if orthogonal:
                projections = np.zeros(i + 1, dtype=np.complex128)
                normalized_amplitudes.append(
                    projections)  # i.e. projections = normalized_amplitudes[i]
                fc = np.exp(1.0j * 2 * np.pi * tunes[i] * np.arange(n_turns))

                # First, compute the frequency correlations
                for j in range(0, i):
                    Pij = freq_projection(fc, tunes, normalized_amplitudes[j], j + 1, window)
                    for k in range(0, j + 1):
                        normalized_amplitudes[i][k] -= Pij * normalized_amplitudes[j][k]

                # Self projection is unity
                normalized_amplitudes[i][i] = 1.0

                # Scale frequency amplitudes by actual signal factor, except last (self) one
                freq_amp = signal_projection(signal, tunes, projections, i + 1, window)
                for k in range(0, i + 1):
                    normalized_amplitudes[i][k] *= freq_amp

                # Subtract out scaled frequencies
                for j in range(0, i + 1):
                    signal -= normalized_amplitudes[i][j] * np.exp(
                        1.0j * 2 * np.pi * tunes[j] * np.arange(n_turns))

                return freq_amp
            else:
                # Subtract latest tune
                # For compatibility, add blank objects
                projections = np.zeros(i + 1, dtype=np.complex128)
                normalized_amplitudes.append(projections)
                fc = np.exp(1.0j * 2 * np.pi * tunes[i] * np.arange(n_turns))
                fcn = np.exp(-1.0j * 2 * np.pi * tunes[i] * np.arange(n_turns))
                #  Note Ai * cAi = np.sum([1, 1, ...]) = n_turns, if no window
                # freq_amp = np.sum(fcn * signal * window) / np.sum(fc * np.conjugate(fc) * window)
                freq_amp = np.sum(fcn * signal) / n_turns
                signal -= freq_amp * fc

                return freq_amp

        f = lambda *args: -1 * get_amplitude(*args)

        # Main NAFF loop
        results = []
        tunes, tunes_neg = [], []
        normalized_amplitudes, normalized_amplitudes_neg = [], []  # list of arrays == 'upper diagonal matrix'
        for i in range(0, n_components):
            tic = time.perf_counter()
            dataw = data * window
            # Get FFT peak as guess
            tune0 = get_top_freq(dataw)
            # logger.debug(f'Component ({i}) - tune initial guess: {tune0}')

            # Find optimum
            res = minimize_scalar(f,
                                  bracket=(tune0 - 1e-3, tune0 + 1e-3),
                                  args=dataw,
                                  method='brent',
                                  options={'xtol': xatol})
            tune = res.x
            if perf_mode:
                tune_neg = -tune
            else:
                res2 = minimize_scalar(f,
                                       bracket=(-tune0 - 1e-3, -tune0 + 1e-3),
                                       args=dataw,
                                       method='brent',
                                       options={'xtol': xatol})
                tune_neg = res2.x
            tunes.append(tune)
            tunes_neg.append(tune_neg)
            # logger.debug(f'Component ({i}) - found +f={tune:.7f} and -f={tune_neg:.7f}')

            # Remove frequencies
            if not perf_mode:
                data_original = data.copy()
            amplitude = remove_frequencies(data, tunes, normalized_amplitudes, i, window,
                                           orthogonal)
            amplitude_neg = remove_frequencies(data, tunes_neg, normalized_amplitudes_neg, i,
                                               window, orthogonal)

            if perf_mode:
                results.append({'tune': tune / sp})
            else:
                results.append({'tune': tune / sp,
                                'tunes': (tune0 / sp, tune / sp, tune_neg / sp),
                                'namps': normalized_amplitudes.copy(),
                                'namps_neg': normalized_amplitudes_neg.copy(),
                                'amps': amplitude,
                                'amps_neg': amplitude_neg,
                                'absamps': np.abs(amplitude),
                                'absamps_neg': np.abs(amplitude_neg),
                                'fit': (res, res2),
                                'signal': data_original
                                })
            self.naff_runtimes.append(time.perf_counter() - tic)

        if full_data:
            return results
        else:
            return [r['tune'] for r in results]

    def __calculate_naff_turns(self, data: np.ndarray, turns: int, order: int):
        """ Returns max number of turns that can be used for NAFF, depending on integrator and order """
        if turns >= len(data) + 1:
            raise ValueError(
                f'Input data must be at least of length ({turns + 1}), only have ({len(data)})')

        if turns < order:
            raise ValueError(
                f'Minimum number of turns for integrator order ({order}) not met, only have ({turns})')

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
                             window_order: int = None,
                             integrator_order: int = 6,
                             method: int = 11,
                             data_trim: slice = None,
                             no_trim: bool = False,
                             magnitude_only: bool = True
                             ):
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
        warnings.warn("This method is known to produce slightly wrong output - use v2!")

        window_order = window_order or self.window_power

        single_freq_mode = False
        if isinstance(frequencies, float) or isinstance(frequencies, int):
            frequencies = [frequencies]
            single_freq_mode = True

        if not no_trim:
            data_trim = data_trim or self.data_trim
            if data_trim:
                if data_trim.stop is not None and data_trim.stop > len(data):
                    raise Exception(
                        f"Trim end ({data_trim.stop}) exceeds available data length ({len(data)})")
                data = data[data_trim]
                data = data - np.mean(data)

        # Check for deprecated parameters, temp hack
        if turns or skip_turns:
            raise Exception('Deprecated parameters used!')

        if not turns:
            turns = len(data) - 1

        turns_naff = self.__calculate_naff_turns(data, turns, integrator_order)
        data = data[skip_turns:skip_turns + turns_naff + 1]

        if (turns_naff, window_order) not in self.window_cache:
            # Hanning window
            T = np.linspace(0, turns_naff, num=turns_naff + 1,
                            endpoint=True) * 2.0 * np.pi - np.pi * turns_naff
            TWIN = ((2.0 ** window_order * np.math.factorial(window_order) ** 2) / float(
                    np.math.factorial(2 * window_order))) * (
                               1.0 + np.cos(T / turns_naff)) ** window_order
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
                fun = _get_integral_v14_wrap()
                integral.append(fun(data, TWIN, i_line,
                                                  self.coeffs_cache[integrator_order - 1],
                                                  f, turns_naff, integrator_order))
            elif method == 15:
                fun = _get_integral_v15_wrap()
                integral.append(fun(data, TWIN, i_line,
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

    def compute_correlations_v2(self,
                                data: np.ndarray,
                                frequencies: Union[List[float], float, int, np.ndarray],
                                window_order: int = None,
                                data_trim: slice = None,
                                no_trim: bool = False,
                                no_window: bool = False,
                                magnitude_only: bool = True
                                ):
        """
        Implementation v2 of correlation computation - simplified this time
        :param no_window:
        :param data: Signal data array
        :param frequencies: Frequencies to compute correlations for
        :param no_trim: Override all trims and work on raw data
        :param data_trim:
        :param window_order:
        :param magnitude_only:
        :return:
        """
        if not no_trim:
            data_trim = data_trim or self.data_trim
            if data_trim is not None:
                if data_trim.stop is not None and data_trim.stop > len(data):
                    raise Exception(
                        f"Trim end ({data_trim.stop}) exceeds available data length ({len(data)})")
                data = data[data_trim]
                data = data - np.mean(data)

        N = len(data)
        if N in self.icache:
            i_line = self.icache[N]
        else:
            self.icache[N] = i_line = np.arange(0, len(data))

        if no_window:
            dataw = data
        else:
            window_order = window_order or self.window_power
            window = self.window(N, window_order)
            dataw = data * window

        if isinstance(frequencies, float) or isinstance(frequencies, int):
            # Single freq mode
            correlation = np.sum(dataw * np.exp(-2.0j * np.pi * frequencies * i_line)) / N
            # pi = np.pi
            # correlation = ne.evaluate("sum(dataw * exp(-2.0j * pi * frequencies * i_line))")
            return np.abs(correlation), np.real(correlation), np.imag(correlation)
        else:
            if magnitude_only:
                integral = np.zeros((len(frequencies), 1))
                for i, tune in enumerate(frequencies):
                    correlation = np.sum(dataw * np.exp(-2.0j * np.pi * tune * i_line)) / N
                    integral[i, 0] = np.abs(correlation)
                return integral
            else:
                integral = np.zeros((len(frequencies), 4), dtype=np.complex128)
                for i, tune in enumerate(frequencies):
                    correlation = np.sum(dataw * np.exp(-2.0j * np.pi * tune * i_line)) / N
                    integral[i, :] = (
                    np.abs(correlation), np.real(correlation), np.imag(correlation), correlation)
                return integral

    correlation = compute_correlations_v2

    def get_projection(self, u, v):
        """
        Get inner product projection of two vectors - <u,v>/<u,u>
        """
        # <v,v> = 1 * len(v) because frequency vector has entries of magnitude 1
        return np.sum(np.conjugate(u) * v) / len(u)

    def get_current_window(self, N):
        return self.window(N, self.window_power)

    def window(self, N: int, power: int, kind: str = None, **kwargs):
        """
        Returns a windowing function.
        For cache, manual management is used (vs lru_cache), so as to provide extra options
        :param power:
        :param N:
        :param kind:
        :return:
        """
        kind = kind or self.window_type
        opts = kwargs or self.window_opts
        key = (kind, N, power)
        if key in self.window_cache:
            window = self.window_cache[key]
        else:
            if power == 0:
                window = np.ones(N)
            else:
                if kind == 'hann':
                    # Hanning window - this is 2x vs np.hanning
                    T = np.linspace(0, N - 1, num=N, endpoint=True) * 2.0 * np.pi - np.pi * (N - 1)
                    window = ((2 ** power * np.math.factorial(power) ** 2) /
                              np.math.factorial(2 * power)) * (1.0 + np.cos(T / (N - 1))) ** power
                    self.window_cache[key] = window
                elif kind == 'hann2':
                    window = np.hanning(N) ** power
                    self.window_cache[key] = window
                elif kind == 'halfhann':
                    # Half-window for impulse data windowing (preserving more at the start)
                    # Suitable for cases like fast decaying kicks
                    window = np.hanning(N * 2)[N:] ** power
                    self.window_cache[key] = window
                elif kind == 'tukey':
                    # Cosine taper, another option to preserve signal at ends (see obspy for inspiration)
                    import scipy.signal as ss
                    assert power == 1
                    assert len(opts) == 1 and 'alpha' in opts
                    window = ss.windows.tukey(N, **opts)
                    # dont cache for now
                else:
                    raise Exception(f'Unknown window type:{kind}')
        return window

    def reset_perf_counters(self):
        self.naff_runtimes = []
