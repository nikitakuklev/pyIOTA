__all__ = ['Routines', 'NIO', 'Invariants', 'Coordinates', 'Phase', 'Twiss', 'SVD', 'ICA', 'Interpolator']

import itertools
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from .tbt import Kick
from numba import jit
from scipy.optimize import curve_fit
from scipy.signal import hilbert

from .. import math as pmath
from sklearn import decomposition

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


class Physics:
    @staticmethod
    def magnetic_rigidity(pc):
        """ Computes magnetic rigidity in T*m for energy given in MeV - only valid for E>>rest energy """
        return pc / 300.0

    @staticmethod
    def beta_gamma(p_central_mev: float):
        import scipy.constants
        return np.sqrt(p_central_mev ** 2 / (scipy.constants.value('electron mass energy equivalent in MeV') ** 2) - 1)


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
    def compute_CS_2D_2BPM(x1, x2, beta1, beta2, alpha1, alpha2, dphase):
        """
         Compute Courant-Snyder invariant directly from unnormalized 2BPM data
         (this composes px calculation and normalization)
         Ref: Independent component analysis for beam measurements - https://doi.org/10.1063/1.3226858
         Ref: S.Y.Lee book
        """
        return (x1 ** 2 + (x2 * 1 / np.sin(dphase) * np.sqrt(beta1 / beta2) - x1 * 1 / np.tan(dphase)) ** 2) / beta1

    @staticmethod
    def compute_I1(x, px, y, py, alpha, c, normalized=True):
        """
        Compute first DN invariant in QI notation - (x^2+y^2+px^2+py^2)/2 + (x^4+y^4-6*y^2*x^2)*a/4
        :param x:
        :param px:
        :param y:
        :param py:
        :param alpha: standard constant
        :param c: an unused parameter
        :param normalized: placeholder
        :return: I1
        """
        # c is unused
        assert normalized
        assert c is None
        I = (x ** 2 + y ** 2 + px ** 2 + py ** 2) / 2 + alpha / 4 * (x ** 4 + y ** 4 - 6 * (y ** 2) * (x ** 2))
        return I

    @staticmethod
    def compute_I1_DN(x, px, y, py, t, c, normalized=True):
        xN = x / c
        yN = y / c
        sqrt = np.sqrt
        u = (sqrt((xN + 1.) ** 2 + yN ** 2) + sqrt((xN - 1.) ** 2 + yN ** 2)) / 2.
        v = (sqrt((xN + 1.) ** 2 + yN ** 2) - sqrt((xN - 1.) ** 2 + yN ** 2)) / 2.
        f2u = u * sqrt(u ** 2 - 1.) * np.arccosh(u)
        g2v = v * sqrt(1. - v ** 2) * (-np.pi / 2 + np.arccos(v))
        elliptic = (f2u + g2v) / (u ** 2 - v ** 2)
        quadratic = 0.5 * (px ** 2 + py ** 2) + 0.5 * (x ** 2 + y ** 2)
        I = quadratic + t * c ** 2 * elliptic
        return I

    @staticmethod
    def compute_I1_DN_CM(x, px, y, py, t, c, normalized=True):
        x = x / c;
        y = y / c;
        px = px / c;
        py = py / c;
        quadratic = 0.5 * (px ** 2 + py ** 2 + x ** 2 + y ** 2)
        z = (x + 1.0j * y)
        elliptic = t * np.real(z / np.sqrt(1 - z * z) * np.arcsin(z))
        return quadratic - elliptic

    @staticmethod
    def compute_I2_DN(x, px, y, py, t, c, normalized=True):
        sqrt = np.sqrt
        # Angular momentum
        ang_momentum = (x * py - y * px) ** 2
        lin_momentum = (c * px) ** 2

        # elliptic coordinates
        xN = x / c
        yN = y / c

        u = (sqrt((xN + 1) ** 2 + yN ** 2) + sqrt((xN - 1) ** 2 + yN ** 2)) / (2.)
        v = (sqrt((xN + 1) ** 2 + yN ** 2) - sqrt((xN - 1) ** 2 + yN ** 2)) / (2.)

        # harmonic part of the potential
        f1u = c ** 2 * u ** 2 * (u ** 2 - 1.)
        g1v = c ** 2 * v ** 2 * (1. - v ** 2)

        # elliptic part of the potential
        f2u = -t * c ** 2 * u * sqrt(u ** 2 - 1.) * np.arccosh(u)
        g2v = -t * c ** 2 * v * sqrt(1. - v ** 2) * (0.5 * np.pi - np.arccos(v))

        # combined potentials
        fu = (0.5 * f1u - f2u)
        gv = (0.5 * g1v + g2v)

        # putting it all together
        invariant = (ang_momentum + lin_momentum) + 2. * (c ** 2) * \
                    (fu * v ** 2 + gv * u ** 2) / (u ** 2 - v ** 2)
        return invariant

    @staticmethod
    def compute_I2_DN_CM(x, px, y, py, t, c, normalized=True):
        x = x / c;
        y = y / c;
        px = px / c;
        py = py / c;
        quadratic = ((x * py - y * px) ** 2) + px ** 2 + x ** 2
        z = (x + 1.0j * y)
        elliptic = t * np.real((z + np.conj(z)) / np.sqrt(1 - z * z) * np.arcsin(z))
        return quadratic - elliptic


class Coordinates:
    @staticmethod
    def matrix_Binv(beta: float, alpha: float) -> np.ndarray:
        """ Courant-Snyder normalization matrix B^-1 """
        return np.array([[1 / np.sqrt(beta), 0.0], [alpha / np.sqrt(beta), np.sqrt(beta)]])

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
    def calc_px_from_bpms(x1, x2, beta1, beta2, a1, a2, dphase) -> np.ndarray:
        """ Compute momentum at location 1 from position readings at locations 1 and 2 and local optics funcs """
        # px1 = x2 * (1 / np.sin(dphase)) * (1 / np.sqrt(beta1 * beta2)) - \
        #      x1 * (1 / np.tan(dphase)) * (1 / beta1) - x1 * a1 / beta1
        csc, cot = 1 / np.sin(dphase), 1 / np.tan(dphase)
        px1 = - x1 * cot / beta1 + - x1 * a1 / beta1 + x2 * csc / np.sqrt(beta1 * beta2)
        return px1

    @staticmethod
    def calc_px_from_bpms_v2(x1, x2, beta1, beta2, a1, a2, dphase) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute momentum at location 1 and 2 from position readings and local optics funcs
        See 'phasespace.nb', fixed Jun 2021
        """
        csc, cot = 1 / np.sin(dphase), 1 / np.tan(dphase)
        px1 = - x1 * cot / beta1 + - x1 * a1 / beta1 + x2 * csc / np.sqrt(beta1 * beta2)
        px2 = - x1 * csc / np.sqrt(beta1 * beta2) + x2 * cot / beta2 - x2 * a2 / beta2
        return px1, px2

    @staticmethod
    def calc_pxn_from_normalized_bpms(x1n: np.ndarray, x2n: np.ndarray, dphase: float) -> np.ndarray:
        """
        Compute normalized momentum at location 1 from normalized positions 1 and 2
        See https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.8.024001 and 'phasespace.nb' notebook
        """
        csc, cot = 1 / np.sin(dphase), 1 / np.tan(dphase)
        px1n = -x1n * cot + x2n * csc
        return px1n

    @staticmethod
    def calc_pxn_from_normalized_bpms_v2(x1n: np.ndarray, x2n: np.ndarray, dphase: float) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Compute normalized momentum at location 1 and 2 from normalized positions 1 and 2
        Same refs as above
        """
        csc, cot = 1 / np.sin(dphase), 1 / np.tan(dphase)
        px1n = -x1n * cot + x2n * csc
        px2n = -x1n * csc + x2n * cot
        return px1n, px2n

    @staticmethod
    def calc_px_from_bpms_mat(x1, x2, M11, M12):
        """
        Compute momentum at location 1 from position readings at locations 1 and 2 and transfer matrix
        Ref: https://doi.org/10.1103/PhysRevAccelBeams.23.052802
        """
        return (x2 - M11 * x1) / M12

    @staticmethod
    def slopes_to_momenta(xp, yp, delta):
        """ From elegant - convert (x, x') to canonical (x, px) """
        # ref - LS-356
        denom = np.sqrt(1 + xp ** 2 + yp ** 2)
        px = (1 + delta) * xp / denom
        py = (1 + delta) * yp / denom
        return px, py

    @staticmethod
    def slopes_to_canonical(x, xp, y, yp, cdt, delta):
        """ Same as above """
        factor = (1 + delta) / np.sqrt(1 + xp ** 2 + yp ** 2)
        px = xp * factor
        py = yp * factor
        return x, px, y, py, cdt, delta

    @staticmethod
    def canonical_to_slopes(x, px, y, py, cdt, delta):
        """ Convert canonical momenta to slopes """
        # LS-356
        factor = 1 / np.sqrt((1 + delta ** 2) - px ** 2 - py ** 2)
        xp = px * factor
        yp = py * factor
        return x, xp, y, yp, cdt, delta


class NBPM:
    def __init__(self, box, pairs, params):
        params_default = {'sig_x': 100e-6, 'sig_phi': 0.1, 'verbose': True}
        params_default.update(params)
        assert all(p[0] == pairs[0][0] for p in pairs) # looking at one bpm
        self.pairs = pairs
        self.params = params_default
        self.box = box
        self.bpms = list(set([b[0] for b in pairs] + [b[1] for b in pairs]))
        self.plane = 'H'
        self.verbose = False

    def import_kick(self, k: Kick, key, families=None):
        assert families is not None
        assert all(b in k.bpm_list for b in self.bpms)
        df_pxn = k.v2_momentum(families=families, key=key, out=None, pairs=self.pairs)
        self.pxn = df_pxn

    def compute(self, mean=False, variance=False, covariance_stat=False,
                   covariance_full=False, covariance_stat_tbtav=False, plane=None):
        if plane is not None:
            self.plane = plane
        if mean:
            px1n_comboM, error_finalM, weightsM = self.compute_momentum_mean()
            return px1n_comboM
        xna = self.params.get('var_xn_average', None)
        if variance:
            px1n_comboN, error_finalN, weightsN = self.compute_momentum_combo_variance(xn_avg = xna)
            return px1n_comboN
        if covariance_stat:
            px1n_comboCV, error_finalCV, weightsCV = self.compute_momentum_combo_v2()
            return px1n_comboCV
        xna = self.params.get('tbtav_xn_average',None)
        if covariance_stat_tbtav:
            px1n_comboT, error_finalT, weightsT = self.compute_momentum_combo_tbtav(xn_avg = xna)
            return px1n_comboT
        raise Exception

    def weights_df(self, mean=False, variance=False, covariance_stat=False,
                   covariance_full=False, covariance_stat_tbtav=False):
        """ Make a pretty dataframe with optics and weights """
        bpm_pairs = self.pairs

        def dist180(dp):
            dg = dp * 180 / np.pi % 180
            if dg > 0:
                dg = -dg if dg < 90 else 180 - dg
            else:
                dg = -dg if np.abs(dg) < 90 else -180 - dg
            return dg

        PI = np.pi
        PI2 = np.pi * 2

        if mean:
            px1n_comboM, error_finalM, weightsM = self.compute_momentum_mean()
        xna = self.params.get('var_xn_average', None)
        if variance:
            px1n_comboN, error_finalN, weightsN = self.compute_momentum_combo_variance(xn_avg = xna)
        if covariance_stat:
            px1n_comboCV, error_finalCV, weightsCV = self.compute_momentum_combo_v2()
        xna = self.params.get('tbtav_xn_average',None)
        if covariance_stat_tbtav:
            px1n_comboT, error_finalT, weightsT = self.compute_momentum_combo_tbtav(xn_avg = xna)

        dlist = []
        for i, pair in enumerate(bpm_pairs):
            b1, b2, a1, a2, dp, drel = self.get_opt(pair)
            row = {'BPM1': pair[0], 'BPM2': pair[1], 'beta1': b1, 'beta2': b2, 'delta(deg)': (dp % PI) * (180 / PI) - 90,
                   'dist180': dist180(dp), 'distcut': np.abs(dist180(dp)) - (0.05 * PI2) * 180 / PI}
            if mean:
                row['w_mean'] = weightsM.iloc[i, 0]
                row['e_mean'] = error_finalM
            if variance:
                row['w_var'] = weightsN.iloc[i, 0]
                row['e_var'] = error_finalN
            if covariance_stat:
                row['w_cvnaive'] = weightsCV.iloc[i, 0]
                row['e_cvnaive'] = error_finalCV
            if covariance_stat_tbtav:
                row['w_cvtbtav'] = weightsT.iloc[i, 0]
                row['e_cvtbtav'] = error_finalT
            #  'wt':weightsT.iloc[i,0]})
            dlist.append(row)
        return pd.DataFrame(data=dlist)

    def compute_momentum_combo_tbtav(self, plane=None, xn_avg = None):
        plane = plane or self.plane
        pws = self.pairs
        px1n_arr = self.pxn[plane].values
        pars = self.params.copy()
        pars['xn_average'] = xn_avg
        # print('TBT average calc')
        x = {b: np.nan for b in np.unique(list(itertools.chain.from_iterable(pws)))}
        V, VI, params = self.covar_final(pws, x, pars)
        weights = np.sum(VI, axis=1) / np.sum(VI)
        variances = np.diag(V)
        error_final = np.dot(weights, (np.dot(V, weights.T)))
        px1n_combo = np.sum(px1n_arr * weights[:, np.newaxis], axis=0)
        weights = pd.DataFrame(np.repeat(weights[:, np.newaxis], px1n_arr.shape[1], axis=1), index=pws)
        return px1n_combo, error_final, weights

    def compute_momentum_combo_tbt(self, plane=None):
        plane = plane or self.plane
        pws = self.pairs
        # print('TBT per-turn calc')
        raise Exception
        weights_list = []
        px1n_combo = np.zeros(px1n_arr.shape[1])
        for turn in range(px1n_arr.shape[1]):
            x = self.params['data_ideal'].loc[:, turn].to_dict()
            # x = data.loc[:,turn].to_dict()
            # print(x)
            V, VI, params = self.covar_final(pws, x, params)
            weights = np.sum(VI, axis=1) / np.sum(VI)
            variances = np.diag(V)
            error_final = np.dot(weights, (np.dot(V, weights.T)))
            px1n_combo[turn] = np.sum(px1n_arr[:, turn] * weights)
            weights_list.append(weights[:, np.newaxis])
            if self.verbose:
                print(f'Turn {turn} - {weights} | {variances}')
        weights = pd.DataFrame(np.hstack(weights_list), index=pws)
        return px1n_combo, error_final, weights

    def compute_momentum_combo_v2(self, plane=None):
        plane = plane or self.plane
        pws = self.pairs
        x = {b: 0.0 for b in np.unique(list(itertools.chain.from_iterable(pws)))}
        V, VI, params = self.covar_final(pws, x, self.params)
        weights = np.sum(VI, axis=1) / np.sum(VI)
        variances = np.diag(V)
        error_final = np.dot(weights, (np.dot(V, weights.T)))

        # if params['verbose']: bpm_summary(pws, weights, variances)
        px1n_arr = self.pxn[plane].values
        px1n_combo = np.sum(px1n_arr * weights[:, np.newaxis], axis=0)
        df = pd.DataFrame(np.repeat(weights[:, np.newaxis], px1n_arr.shape[1], axis=1), index=pws)
        return px1n_combo, error_final, df

    def compute_momentum_combo_variance(self, plane=None, xn_avg = None):
        plane = plane or self.plane
        pws = self.pairs
        pars = self.params
        if xn_avg is not None:
            pars = self.params.copy()
            pars['xn_average'] = xn_avg
        x = {b: 0.0 for b in self.bpms}
        V, VI, params = self.var_final(pws, x, pars)
        weights = np.sum(VI, axis=1) / np.sum(VI)
        variances = np.diag(V)
        error_final = np.dot(weights, (np.dot(V, weights.T)))

        # if params['verbose']: bpm_summary(pws, weights, variances)
        px1n_arr = self.pxn[plane].values
        px1n_combo = np.sum(px1n_arr * weights[:, np.newaxis], axis=0)
        df = pd.DataFrame(np.repeat(weights[:, np.newaxis], px1n_arr.shape[1], axis=1), index=pws)
        return px1n_combo, error_final, df

    def compute_momentum_single(self, pair=None, pair_idx=None, plane=None):
        plane = plane or self.plane
        pws = self.pairs
        px1n_arr = self.pxn[plane].values
        if pair is not None:
            pair_idx = pws.index(pair)
        if pair_idx:
            px1n_mean = px1n_arr[pair_idx, :]
            weights = np.zeros(px1n_arr.shape[0])
            weights[pair_idx] = 1.0
        else:
            px1n_mean = px1n_arr[0, :]
            weights = np.zeros(px1n_arr.shape[0])
            weights[0] = 1.0
        error_final = 0.0
        df = pd.DataFrame(np.repeat(weights[:, np.newaxis], px1n_arr.shape[1], axis=1), index=pws)
        return px1n_mean, error_final, df

    def compute_momentum_mean(self, plane='H'):
        pws = self.pairs
        px1n_arr = self.pxn[plane].values
        px1n_mean = np.mean(px1n_arr, axis=0)
        error_final = np.mean(np.std(px1n_arr, axis=0))
        weights = np.ones(px1n_arr.shape[0]) / px1n_arr.shape[0]
        df = pd.DataFrame(np.repeat(weights[:, np.newaxis], px1n_arr.shape[1], axis=1), index=pws)
        return px1n_mean, error_final, df

    def var_final(self, bpm_pairs, x, params):
        """ Compute variance only matrix """
        V, VI, params = self.covar_final(bpm_pairs, x, params)
        V2 = np.zeros_like(V)
        np.fill_diagonal(V2, np.diagonal(V))
        V = V2
        return V, np.linalg.pinv(V), params  # np.linalg.pinv(V, rcond=1.0e-14)

    def covar_final(self, bpm_pairs, x, params):
        """ Compute statistical covariance matrix """
        bpm_names = np.unique(list(itertools.chain.from_iterable(bpm_pairs)))
        bpm_pairs_map = {t: i for i, t in enumerate(bpm_pairs)}
        bpm_numbers_map = {bpm_name: i for i, bpm_name in enumerate(bpm_names)}
        bpm_pairs_numbers = [(bpm_numbers_map[b1], bpm_numbers_map[b2]) for b1, b2 in bpm_pairs]

        n_pairs = len(bpm_pairs)
        n_coordinates = 2 + (len(bpm_pairs) - 1)
        coordinates = [f'x_{bpm}' for bpm in bpm_names]
        n_phases = len(bpm_pairs)
        phases = [f'phi_{bpm1}_{bpm2}' for bpm1, bpm2 in bpm_pairs]
        n_variables = n_coordinates + n_phases
        variables = coordinates + phases

        params2 = {'bpm_pairs': bpm_pairs, 'bpm_pairs_map': bpm_pairs_map, 'bpm_numbers_map': bpm_numbers_map,
                   'variables': variables,
                   'n_pairs': n_pairs, 'n_coordinates': n_coordinates, 'n_phases': n_phases, 'n_variables': n_variables,
                   'x': x}
        params.update(params2)

        if self.verbose:
            print(bpm_pairs)
            print(bpm_pairs_numbers)
            print(f'Coords:{n_coordinates} | Phases:{n_phases} | Total:{n_variables}')

        J = self.pxn_jacobian(params)
        M = self.covar_variables(params['sig_x'], params['sig_phi'], params)
        V = J @ M @ J.T
        # V = J@M@np.linalg.pinv(J)
        params['J'] = J
        params['M'] = M
        return V, np.linalg.pinv(V), params  # np.linalg.pinv(V, rcond=1.0e-14)

    def pxn_partial_derivative(self, i, j, params):
        # Jacobian is D[pxn[i], var[j]] for i=[n_pairs] for j=[n_variables]
        assert 0 <= i < params['n_pairs'] and 0 <= j < params['n_variables']

        pair = params['bpm_pairs'][i]
        b1, b2, a1, a2, dp, drel = self.get_opt(pair)
        idx1, idx2 = params['bpm_numbers_map'][pair[0]], params['bpm_numbers_map'][pair[1]]
        idx3 = params['bpm_pairs_map'][pair]
        cotdp = 1.0 / np.tan(dp)
        cscdp = 1.0 / np.sin(dp)
        if j < params['n_coordinates']:
            # position derivative
            # Dpxn/dx1 = -(Cot[\[Phi]12]/Sqrt[b1])
            # Dpxn/dx2 = Csc[\[Phi]12]/Sqrt[b2]
            if j == idx1:
                val = -cotdp / np.sqrt(b1)
            elif j == idx2:
                val = cscdp / np.sqrt(b2)
            else:
                val = 0.0
        else:
            # phase derivative
            # Csc[\[Phi]12] (-((x2 Cot[\[Phi]12])/Sqrt[b2]) + (x1 Csc[\[Phi]12])/Sqrt[b1])
            x = params['x']
            x1, x2 = x[pair[0]], x[pair[1]]
            if j - params['n_coordinates'] == idx3:
                if 'xn_average' in params:
                    val = cscdp * params['xn_average'] * (-cotdp + cscdp)
                else:
                    val = cscdp * (- x2 * cotdp / np.sqrt(b2) + x1 * cscdp / np.sqrt(b1))
            else:
                val = 0.0
        if self.verbose: print(
            f'{i},{j} - partial {pair}({idx3})({idx1}-{idx2}) wrt {params["variables"][j]} = {val}')
        return val

    def get_opt(self, pair, plane='x'):
        odf = self.box.bpm_optics['model']
        bpm1, bpm2 = pair
        if plane == 'x':
            a1, a2 = odf.loc[bpm1, 'AX'], odf.loc[bpm2, 'AX']
            b1, b2 = odf.loc[bpm1, 'BX'], odf.loc[bpm2, 'BX']
            dp = odf.loc[bpm2, 'MUX'] - odf.loc[bpm1, 'MUX']  # if bpm1 < bpm2 else -phi_ex_abs[bpm2]+phi_ex_abs[bpm1]
        else:
            raise Exception
        drel = np.abs(dp) / (2 * np.pi) % 0.5
        return b1, b2, a1, a2, dp, drel

    def covar_variables(self, sig_x, sig_phi, params):
        return np.diag(np.array([sig_x ** 2] * params['n_coordinates'] + [sig_phi ** 2] * params['n_phases']))

    def pxn_jacobian(self, params):
        J = np.zeros((params['n_pairs'], params['n_variables']))
        for i in range(params['n_pairs']):
            for j in range(params['n_variables']):
                J[i, j] = self.pxn_partial_derivative(i, j, params)
        return J


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
        """
        phases_rel = phases.copy()
        phases_cum = np.zeros_like(phases)
        phases_rel = pmath.addsubtract_wrap(phases_rel, -phases_rel[0], -np.pi, np.pi)
        for i in range(1, len(phases)):
            phases_cum[i] = phases_cum[i - 1] + pmath.forward_distance(phases_rel[i - 1], phases_rel[i], -np.pi, np.pi)
        return phases_cum

    @staticmethod
    def relative_to_absolute_with_ref(phases: np.ndarray, phases_ref: np.ndarray) -> np.ndarray:
        """
        Converts relative phases to absolutes, assuming first phase is 0,
        Corrects >2pi phase hops with reference set
        """
        assert len(phases) == len(phases_ref)
        phases_rel = phases.copy()
        phases_cum = np.zeros_like(phases)
        phases_rel = pmath.addsubtract_wrap(phases_rel, -phases_rel[0], -np.pi, np.pi)
        for i in range(1, len(phases)):
            extra = 2 * np.pi * np.floor((phases_ref[i] - phases_ref[i - 1]) / (2 * np.pi))
            phases_cum[i] = phases_cum[i - 1] + pmath.forward_distance(phases_rel[i - 1], phases_rel[i], -np.pi, np.pi)
            phases_cum[i] += extra
        return phases_cum


class Twiss:
    """
    References:
    https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.18.031002
    https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.20.111002
    https://github.com/pylhc/omc3/blob/master/omc3/optics_measurements/beta_from_phase.py
    https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.11.084002 (phase correction, TBD if needed)
    """

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
        return np.sqrt(beta * emittance + (dispersion * deltaE) ** 2)


class Envelope:
    """
    Provides several models of signal amplitude envelopes, and associated fitting methods
    """

    def __init__(self, data_trim=None, output_trim=None):
        self.data_trim = data_trim
        self.output_trim = output_trim

    # The following F4 terms are from lee/nadolski 4D treatment
    @staticmethod
    def F4D_chroma(n, chrom_x, sigma_e, nu_s):
        theta = 2 * np.pi * chrom_x * sigma_e * np.sin(np.pi * n * nu_s) / nu_s
        F = np.exp(-theta * theta / 2)
        return F

    @staticmethod
    @jit(nopython=True)
    def F4D_xx(n: np.ndarray, j_x: float, sigma_x: float, k_xx: float):
        theta = 4 * np.pi * k_xx * sigma_x * sigma_x * n
        F = 1 / (1 + theta * theta) * np.exp(-(j_x * theta * theta) / (2 * sigma_x * sigma_x * (1 + theta * theta)))
        return F

    @staticmethod
    @jit(nopython=True)
    def F4D_xx_Z(n: np.ndarray, Z: float, mu: float):
        # Z = j_x * sigma_x * sigma_x    mu = k_xx / j_x
        # new Z = k_xx * sigma_x * sigma_x     mu = j_x * k_xx
        theta = 4 * np.pi * Z * n
        F = 1 / (1 + theta * theta) * np.exp(-(mu / Z * theta * theta) / (2 * (1 + theta * theta)))
        return F

    @staticmethod
    def F4D_xy(n, j_y, sigma_y, k_xy):
        theta = 4 * np.pi * k_xy * sigma_y * sigma_y * n
        F = 1 / (1 + theta * theta) * np.exp(-(j_y * theta * theta) / (2 * sigma_y * sigma_y * (1 + theta * theta)))
        return F

    @staticmethod
    def F4D_tune_lowtheta(k_xx, k_xy, j_x, j_y, sigma_x, sigma_y, ):
        dnu_x = k_xx * (j_x + 4 * sigma_x * sigma_x) + k_xy * (j_y + 4 * sigma_y * sigma_y)
        return dnu_x

    @staticmethod
    def F4D_tune(n, k_xx, k_xy, j_x, j_y, sigma_x, sigma_y):
        ex = sigma_x * sigma_x
        ey = sigma_y * sigma_y
        thetaxy = 4 * np.pi * k_xy * sigma_y * sigma_y * n
        thetaxx = 4 * np.pi * k_xx * sigma_x * sigma_x * n
        dnu_x = k_xx * ((4 * ex + j_x * (1 - thetaxx * thetaxx)) / (1 + thetaxx * thetaxx))
        dnu_x += k_xy * ((4 * ey + j_y * (1 - thetaxy * thetaxy)) / (1 + thetaxy * thetaxy))
        return dnu_x

    @staticmethod
    def lee_4D(x, chrom_x, sigma_e, nu_s, j_x, sigma_x, k_xx, j_y, sigma_y, k_xy):
        chroma = Envelope.F4D_chroma(x, chrom_x, sigma_e, nu_s, )
        xx = Envelope.F4D_xx(x, j_x, sigma_x, k_xx)
        xy = Envelope.F4D_xy(x, j_y, sigma_y, k_xy)
        return chroma, xx, xy

    # Fitting functions based on above 4D formulas
    @staticmethod
    def amp_fxx(x, pars):
        """ Lee Fxx only """
        A, offset, j_x, sigma_x, k_xx = pars
        Fxx = Envelope.F4D_xx(x, j_x=j_x, sigma_x=sigma_x, k_xx=k_xx)
        return A * Fxx * np.sqrt(j_x) + offset

    @staticmethod
    def fxx(x, data_x, data_y):
        envelope = Envelope.amp_fxx(data_x, x)
        lsq = np.sqrt(np.sum((data_y - envelope) ** 2))
        return lsq / len(data_x)

    @staticmethod
    def amp_fxx_nooffset(x, pars):
        """ Lee Fxx only with no y offset """
        A, j_x, sigma_x, k_xx = pars
        Fxx = Envelope.F4D_xx(x, j_x=j_x, sigma_x=sigma_x, k_xx=k_xx)
        return A * Fxx * np.sqrt(j_x)

    @staticmethod
    def fxx_nooffset(x, data_x, data_y):
        envelope = Envelope.amp_fxx_nooffset(data_x, x)
        lsq = np.sqrt(np.sum((data_y - envelope) ** 2))
        return lsq / len(data_x)

    @staticmethod
    def amp_fxx_nooffset_group(x, pars, i=None):
        """ Group fit version of fxx """
        j_x, sigma_x, k_xx = pars[-3:]
        amplitudes = pars[:-3]

        Fxx = Envelope.F4D_xx(x, j_x=j_x, sigma_x=sigma_x, k_xx=k_xx)
        if i is not None:
            return amplitudes[i] * Fxx * np.sqrt(j_x)
        else:
            return Fxx * np.sqrt(j_x)

    @staticmethod
    def fxx_nooffset_group(x, data_x, data_y):
        assert len(x) == data_y.shape[0] + 3
        envelope = Envelope.amp_fxx_nooffset_group(data_x, x, None)
        amplitudes = x[:data_y.shape[0]]
        envelope2D = amplitudes[:, np.newaxis] @ envelope[np.newaxis, :]
        assert envelope2D.shape == data_y.shape
        lsq = np.sqrt(np.sum((data_y - envelope2D) ** 2))
        return lsq

    @staticmethod
    def amp_fxxZ(x, pars):
        """ Lee Fxx reformulated in terms of new parameters Z and mu """
        A, offset, Z, mu = pars
        Fxx = Envelope.F4D_xx_Z(x, Z=Z, mu=mu)
        return A * Fxx + offset

    @staticmethod
    def fxxZ(x, data_x, data_y):
        envelope = Envelope.amp_fxxZ(data_x, x)
        lsq = np.sqrt(np.sum((data_y - envelope) ** 2))
        return lsq / len(data_x)

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
        raise Exception
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
        :return: U, S, V, vh
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
                          add_virtual_bpms: bool = True, families: List[str] = None,
                          n_components: int = 2):
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
                kick.bpms_add([f'SVD2D_{i}C_{family}' for i in range(n_components)], family='C' + family)
                for i in range(n_components):
                    kick.df[f'SVD2D_{i}C_{family}'] = [vh[i, :]]
                    # kick.df[f'SVD2D_{2}C_{family}'] = [vh[1, :]]

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

    def decompose2D(self, data: Kick, plane: str = None, n_components: int = 2):
        if isinstance(data, Kick):
            matrix = data.get_bpm_matrix(plane)
        elif isinstance(data, np.ndarray):
            matrix = data
        else:
            raise Exception(f'Unknown data type: {data}')

        if self.data_trim:
            matrix = matrix[:, self.data_trim]

        matrix -= np.mean(matrix, axis=1)[:, np.newaxis]

        ica = decomposition.FastICA(n_components=n_components, max_iter=2000, random_state=42, tol=1e-8)
        icadata = ica.fit_transform(matrix.T)

        # U, S, vh = np.linalg.svd(matrix, full_matrices=False)
        # V = vh.T  # transpose it back to conventional U @ S @ V.T
        return icadata.T, ica.mixing_


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
