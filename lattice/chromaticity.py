__all__ = ['chromaticity']

import copy
import functools
import numpy as np
import ocelot
from scipy import integrate
from typing import Dict
from ocelot import Twiss, MagneticLattice, Edge, twiss
from ocelot.cpbd.elements import SBend, RBend, Bend, Quadrupole, Multipole, Sextupole
from pyIOTA.tbt import NAFF

"""
Improved chromaticity calculations
"""


def chromaticity(lattice: MagneticLattice,
                 tws_0: Twiss = None,
                 tws_1: Twiss = None,
                 method: str = 'matrix',
                 method_kwargs: Dict = None,
                 debug: bool = False) -> np.ndarray:
    """
    Computes full chromaticity (elements+edges+sextupoles)

    :param lattice: Lattice sequence (with precomputed transfer maps)
    :param tws_0: Optional initial twiss (uses periodic solution if not provided)
    :param tws_1: Endpoint twiss (for channel mode)
    :param method: Computation method

        'matrix' - Recommended, uses second order matrices. Same as MADX or elegant when not tracking,
         i.e. without 'chrom' option.
         Parameters:
          None

        'integral_numeric' - Compute integral of [beta*k1] within focusing elements with discrete steps.
         Less accurate than matrix method. Does not account for feed down of misaligned sextupoles, octupoles, etc.
         Parameters:
          n_steps - number of steps inside each element

        'integral_analytic' - Compute integral of [beta*k1] within focusing elements by using analytic average
         beta function in each. Same caveats as 'integral_numeric', but should be more accurate and faster.
         Parameters:
          None

    :param method_kwargs: Method parameters as a dictionary.
        All methods also support:
        'n_superperiods' - optional number of superperiods that will scale final chromaticity
    :param debug: Toggle debug printing
    :return: (1x2) X/Y chromaticity array

    See https://mad8.web.cern.ch/mad8/doc/phys_guide.pdf
    See https://home.fnal.gov/~ostiguy/OptiM/OptimHelp/dipole_edge.html
    See doi:10.18429/JACoW-IPAC2018-TUPMK014
    See https://slac.stanford.edu/pubs/slacreports/reports01/slac-r-075.pdf page 117
    See LA-UR-11-10334
    See CERN-THESIS-2018-300
    See Fringe Effects in MAD PART I https://frs.web.cern.ch/frs/report/fringe_part_I.pdf
    """

    method_kwargs = method_kwargs or {}
    n_superperiods = method_kwargs.get('n_superperiods', 1)
    chrom_1period = None

    ring_mode = False
    # If no twiss, assume ring
    if tws_0 is None:
        ring_mode = True
        tws = twiss(lattice)
        tws_0 = tws[0]
        tws_1 = tws[-1]
    # Linac solution requires knowledge of twiss at end, if blank assume ring mode
    elif tws_1 is None or tws_1 == tws_0:
        ring_mode = True
        tws_1 = tws_0

    if method == 'track':
        # Track particle for tune
        navi = ocelot.Navigator(lattice)
        p_neg = ocelot.Particle(x=0.0, y=0.0, p=-1e-6, E=0.0)
        p_ref = ocelot.Particle(x=0.0, y=0.0, p=0.0, E=0.0)
        p_pos = ocelot.Particle(x=0.0, y=0.0, p=1e-6, E=0.0)
        p_list = [p_neg, p_ref, p_pos]
        t_list = [ocelot.Track_info(p) for p in p_list]
        tl_out = ocelot.track_nturns(lattice, 1024, t_list, print_progress=False)
        naff = NAFF(window_power=1, fft_pad_zeros_power=14)



    elif method == 'matrix':
        # Use second order matrix - this is same as MADX TWISS routine !without! chrom flag
        # Also same as in elegant for chromaticity correction (but elegant has up to 3rd order, only 2nd here)
        # See https://svn.aps.anl.gov/AOP/oag/trunk/apps/src/elegant/chrom.c
        # This is better than integration but can be inaccurate for small rings - use tracking for best results

        T1 = copy.deepcopy(lattice.T)
        if np.all(T1 == 0.):
            raise Exception('Second order matrix is empty - cannot use matrix chromaticity method')
        # OCELOT has unsymmetric T matrix opposite of elegant convention, need to flip diagonals to match elegant
        # TODO: change computation indexing
        for i in range(6):
            for j in range(6):
                for k in range(j, 6):
                    if j != k:
                        T1[i, k, j] = T1[i, j, k]
                        T1[i, j, k] = 0
        R, T = lattice.R, T1
        tws_params = [(tws_0.beta_x, tws_0.alpha_x, tws_1.beta_x, tws_1.alpha_x, tws_1.mux),
                      (tws_0.beta_y, tws_0.alpha_y, tws_1.beta_y, tws_1.alpha_y, tws_1.muy)]
        chromaticities = []
        for plane, (beta0, alpha0, beta1, alpha1, mu1) in zip([0, 2], tws_params):
            # Uncoupled computation, 1 plane at a time
            R11, R12, R22 = R[0 + plane, 0 + plane], R[0 + plane, 1 + plane], R[1 + plane, 1 + plane]
            if np.isclose(R12, 0.):
                # This will cause division by 0, but maybe we fail nicely with np.nan?
                raise Exception('R12 term is near 0 - something is wrong with optics')
            # Find linear map derivatives
            dR11 = __compute_R_derivative(0 + plane, 0 + plane, tws_1, T)
            dR12 = __compute_R_derivative(0 + plane, 1 + plane, tws_1, T)
            dR22 = __compute_R_derivative(1 + plane, 1 + plane, tws_1, T)
            # Compute dTune/dDelta
            c = __compute_chroma_from_dR(dR11, dR12, dR22, R11, R12, R22, beta0, alpha0, beta1, alpha1, mu1, ring_mode)
            chromaticities.append(c)
        chrom_1period = np.array(chromaticities)
        # This should contain all edge/sextupole contributions already, so we are done
    elif method == 'numeric':
        n_steps = method_kwargs.get('n_steps', 9)
        chrom_1period = natural_chromaticity_numeric(lattice, tws_0, n_steps, debug)
        chrom_1period += edge_chromaticity(lattice, tws_0, debug)
        chrom_1period += sextupole_chromaticity_numeric(lattice, tws_0, n_steps, debug)
    elif method == 'analytic':
        chrom_1period = natural_chromaticity_analytic(lattice, tws_0, debug)
        chrom_1period += edge_chromaticity(lattice, tws_0, debug)
        # TODO: Analytic dispersion to get analytic sextupole contributions
        chrom_1period += sextupole_chromaticity_numeric(lattice, tws_0, 9, debug)
    return chrom_1period * n_superperiods


def __compute_R_derivative(i: int, j: int, tws: Twiss, T: np.ndarray) -> float:
    """ Determine dRij/dDelta from second order matrix """
    eta = [tws.Dx, tws.Dxp, tws.Dy, tws.Dyp, 0, 1]
    result = 0.0
    for k in range(6):
        if k > j:
            result += eta[k] * T[i, k, j]
            # print(i, k, j, eta[k] * T[i][k][j], T[i][k][j], eta[k])
        elif k == j:
            result += eta[k] * T[i, k, j] * 2
            # print(i, k, j, eta[k] * T[i][k][j] * 2, T[i][k][j], eta[k])
        else:
            result += eta[k] * T[i, j, k]
            # print(i, j, k, eta[k] * T[i][j][k], T[i][j][k], eta[k])
    return result


def __compute_chroma_from_dR(dR11: float, dR12: float, dR22: float, R11: float, R12: float, R22: float, beta0: float,
                             alpha0: float, beta1: float, alpha1: float, phi1: float, ring_mode: bool) -> float:
    """ Use linear map first derivatives to get linear chromaticity """
    if ring_mode:
        return -(dR11 + dR22) / R12 * beta0 / (2 * 2 * np.pi)
    else:
        # From elegant, unverified
        return ((dR12 * (np.cos(phi1) + alpha0 * np.sin(phi1))) / np.sqrt(beta0 * beta1) -
                dR11 * np.sin(phi1) * np.sqrt(beta0 / beta1)) / (np.pi * 2)


def natural_chromaticity_analytic(lattice: MagneticLattice, tws_0: Twiss, debug: bool) -> np.ndarray:
    """
    Use twiss at entrance to compute analytic beta average inside, then sum betaavg*k1*l
    """
    tws_elem = copy.deepcopy(tws_0)
    integr_x = integr_y = 0.
    for el in lattice.sequence:
        if el.__class__ in [SBend, RBend, Bend, Quadrupole]:
            if el.l == 0:
                raise Exception(f'Thin focusing element ({el.id}) encountered?')
            # 1/rho
            h = el.angle / el.l if el.__class__ != Quadrupole and el.l != 0 else 0.0
            # This is 1/bendradius^2 dipole horizontal focusing, since arc = angle * radius
            k1x = el.k1 + h * h
            k1y = -el.k1

            # Memoize(cache) computation - lattices hopefully have <4096 unique combinations
            @functools.lru_cache(maxsize=4096)
            def get_avg_beta_parameters(k1, l):
                k1abs = np.abs(k1)
                tworootkl = 2 * np.sqrt(k1abs) * l
                if k1 > 0.0:
                    # Focusing
                    srtkl_over_rtkl = np.sin(tworootkl) / tworootkl
                    u0 = 0.5 * (1 + srtkl_over_rtkl)
                    u1 = (np.sin(np.sqrt(k1abs) * l) ** 2) / (k1abs * l)
                    u2 = (0.5 / k1abs) * (1 - srtkl_over_rtkl)
                elif k1 < 0.0:
                    # Defocusing
                    srtkl_over_rtkld = np.sinh(tworootkl) / tworootkl
                    u0 = 0.5 * (1 + srtkl_over_rtkld)
                    u1 = (np.sinh(np.sqrt(k1abs) * l) ** 2) / (k1abs * l)
                    u2 = (-0.5 / k1abs) * (1 - srtkl_over_rtkld)
                else:
                    # Limit as k -> 0 == drift, where beta(z) = beta0 - 2 * alpha0 *z + gamma0 * z^2
                    u0 = 1
                    u1 = l
                    u2 = l ** 2 / 3
                return u0, u1, u2

            u0x, u1x, u2x = get_avg_beta_parameters(k1x, el.l)
            u0y, u1y, u2y = get_avg_beta_parameters(k1y, el.l)
            twiss_z = tws_elem
            beta_x_avg = twiss_z.beta_x * u0x - twiss_z.alpha_x * u1x + twiss_z.gamma_x * u2x
            beta_y_avg = twiss_z.beta_y * u0y - twiss_z.alpha_y * u1y + twiss_z.gamma_y * u2y
            ix = beta_x_avg * (el.k1 + h * h) * el.l
            iy = beta_y_avg * -el.k1 * el.l
            integr_x += ix
            integr_y += iy
            if debug:
                print(f"{ix:<+8.3f} {iy:<+8.3f} {el.id:6s} k1:{getattr(el, 'k1', None):<+8.2f}"
                      f"\tbx:{np.mean(beta_x_avg):5.3f}\tby:{np.mean(beta_y_avg):5.3f}\th:{h:4.3f}")
                print(twiss_z.beta_x, tws_elem.beta_x, twiss_z.beta_y, tws_elem.beta_y)
        elif el.__class__ == Multipole:
            # Thin multipole
            twiss_z = el.transfer_map * tws_elem
            integr_x += twiss_z.beta_x * el.kn[1]
            integr_y -= twiss_z.beta_y * el.kn[1]
        # Move twiss forward
        tws_elem = el.transfer_map * tws_elem
    ksi_x = -integr_x / (4 * np.pi)
    ksi_y = -integr_y / (4 * np.pi)
    return np.array([ksi_x, ksi_y])


def natural_chromaticity_numeric(lattice: MagneticLattice, tws_0: Twiss, n_steps: int = 9,
                                 debug: bool = False) -> np.ndarray:
    """
    Computes chromaticity as integral of [beta*k1] within focusing elements in discrete steps
    """
    tws_elem = copy.deepcopy(tws_0)
    integr_x = integr_y = 0.
    # Declare now and refill for each element
    bx, by, k, h = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    for el in lattice.sequence:
        if el.__class__ in [SBend, RBend, Bend, Quadrupole]:
            if el.l == 0:
                raise Exception(f'Thin focusing element ({el.id}) encountered?')
            # Use twiss function steps to compute beta*k at several steps
            # Slightly optimized
            if el.__class__ != Quadrupole and el.l != 0:
                h.fill(el.angle / el.l)
            else:
                h.fill(0.0)
            Z = np.linspace(0, el.l, num=n_steps, endpoint=True)
            k.fill(el.k1)
            twiss_z = None
            for i, z_pos in enumerate(Z):
                twiss_z = el.transfer_map(z_pos) * tws_elem
                bx[i], by[i] = twiss_z.beta_x, twiss_z.beta_y
            X = bx * (k + h * h)
            Y = -by * k
            ix = integrate.simps(X, Z)
            iy = integrate.simps(Y, Z)
            integr_x += ix
            integr_y += iy
            if debug:
                print(f"{ix:<+8.3f} {iy:<+8.3f} {el.id:6s} k1:{getattr(el, 'k1', None):<+8.2f}"
                      f"\tbx:{np.mean(bx):5.3f}\tby:{np.mean(by):5.3f}\th:{h[0]:4.3f}")
                print(twiss_z.beta_x, tws_elem.beta_x, twiss_z.beta_y, tws_elem.beta_y)
        elif el.__class__ == Multipole:
            # Thin multipole
            twiss_z = el.transfer_map * tws_elem
            integr_x += twiss_z.beta_x * el.kn[1]
            integr_y -= twiss_z.beta_y * el.kn[1]
        # Move twiss forward
        tws_elem = el.transfer_map * tws_elem
    ksi_x = -integr_x / (4 * np.pi)
    ksi_y = -integr_y / (4 * np.pi)
    return np.array([ksi_x, ksi_y])


def sextupole_chromaticity_numeric(lattice: MagneticLattice, tws_0: Twiss, n_steps: int = 9,
                                   debug: bool = False) -> np.ndarray:
    """
    Computes chromaticity as integral of [Dx*beta*k2] within sextupoles
    """
    tws_elem = copy.deepcopy(tws_0)
    integr_x = integr_y = 0.
    # Declare now and refill for each element
    bx, by, dx = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    for el in lattice.sequence:
        if isinstance(el, Sextupole):
            if el.l == 0:
                raise Exception(f'Thin sextupole ({el.id}) encountered?')
            Z = np.linspace(0, el.l, num=n_steps, endpoint=True)
            twiss_z = None
            for i, z_pos in enumerate(Z):
                twiss_z = el.transfer_map(z_pos) * tws_elem
                bx[i], by[i] = twiss_z.beta_x, twiss_z.beta_y
                dx[i] = twiss_z.Dx
            X = bx * el.k2 * dx
            Y = -by * el.k2 * dx
            ix = integrate.simps(X, Z)
            iy = integrate.simps(Y, Z)
            integr_x += ix
            integr_y += iy
            if debug:
                print(f"{ix:<+8.3f} {iy:<+8.3f} {el.id:6s} k2:{getattr(el, 'k2', None):<+8.2f}"
                      f"\tbx:{np.mean(bx):5.3f}\tby:{np.mean(by):5.3f}\tDx:{dx[0]:4.3f}")
                print(twiss_z.beta_x, tws_elem.beta_x, twiss_z.beta_y, tws_elem.beta_y)
        elif el.__class__ == Multipole:
            # Thin multipole
            if len(el.kn) >= 3:
                twiss_z = el.transfer_map * tws_elem
                integr_x += twiss_z.beta_x * el.kn[2] * twiss_z.Dx
                integr_y -= twiss_z.beta_y * el.kn[2] * twiss_z.Dx
        # Move twiss forward
        tws_elem = el.transfer_map * tws_elem
    ksi_x = -integr_x / (4 * np.pi)
    ksi_y = -integr_y / (4 * np.pi)
    return np.array([ksi_x, ksi_y])


def edge_chromaticity(lattice: MagneticLattice, tws_0: Twiss, debug: bool = False):
    """
    Computes contribution of sector and rectangle bend edges to chromaticity analytically
    """
    # TODO: integrate with EdgeUtil conventions
    ksi_x_edge = 0.0
    ksi_y_edge = 0.0
    l = 0.0
    tws_elem = copy.deepcopy(tws_0)
    for i, el in enumerate(lattice.sequence):
        l += el.l
        tws_elem = el.transfer_map * tws_elem
        if isinstance(el, Edge):
            alpha = el.edge
            # Horizontal focusing only depends on pole angle
            ksi_x_focusing = tws_elem.beta_x * np.tan(alpha) * el.h
            ksi_x_edge += ksi_x_focusing

            # Vertical edge focusing has contributions from soft edge fringe fields
            # Same math as in OCELOT r_matrix.py
            # With tan approximation
            # ksi_y_fringe = tws_elem.beta_y * el.gap * el.fint * (1 + np.sin(alpha) ** 2) / np.cos(alpha) * (el.h ** 2)
            # ksi_y_focusing = -tws_elem.beta_y * np.tan(alpha) * el.h
            # ksi_y_edge += ksi_y_focusing + ksi_y_fringe
            # No tan approximation
            # phi = el.gap * el.fint * el.h * (1 + np.sin(alpha) ** 2) / np.cos(alpha)
            # ksi_y_edge += -tws_elem.beta_y * np.tan(alpha - phi) * el.h
            # Hmmmm - edge focusing doesnt contribute to chroma???
            ksi_y_focusing = -tws_elem.beta_y * np.tan(alpha) * el.h
            ksi_y_edge += ksi_y_focusing
            if debug:
                print("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.5f} {:.3f}".format(el.h, el.gap, el.fint, el.h ** 2, alpha,
                                                                                ksi_y_edge / tws_elem.beta_y,
                                                                                ksi_y_edge))
    return np.array([ksi_x_edge, ksi_y_edge])
