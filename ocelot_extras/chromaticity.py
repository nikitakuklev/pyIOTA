import copy, functools

from ocelot import Twiss, MagneticLattice, Edge
from ocelot.cpbd.elements import SBend, RBend, Bend, Quadrupole, Multipole
import numpy as np
from scipy import integrate

"""
Improved chromaticity calculations vs OCELOT defaults
"""


def natural_chromaticity(lattice: MagneticLattice,
                         tws_0: Twiss,
                         n_superperiods: int = 1,
                         n_steps: int = 5,
                         method='analytic',
                         debug=True) -> np.ndarray:
    """
    Computes chromaticity as integral of [beta*k1] within focusing elements
    Less accurate than tracking, but fast. Does not account for feed down of sextupoles, octupoles, etc.
    TODO: try Newton-Cotes higher orders

    :param lattice: Lattice sequence (with precomputed transfer maps)
    :param tws_0: Optional initial twiss (uses periodic ring solution if not provided)
    :param n_superperiods: Optional number of superperiods that will scale final chromaticity
    :param n_steps: Number of integration steps
    :param method: 'analytic' - use average beta where possible; 'numeric' - do discrete steps everywhere
    :return: (2,) sized array of x/y linear chromaticities
    """
    tws_elem = copy.deepcopy(tws_0)
    integr_x = 0.0
    integr_y = 0.0

    if method == 'numeric':
        # Declare now and refill for each element
        bx, by, k, h = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
        for el in lattice.sequence:
            if el.__class__ in [SBend, RBend, Bend, Quadrupole]:
                if el.l == 0:
                    raise Exception(f'Thin focusing element ({el.id}) encountered?')
                # Use actual tracking steps to compute beta*k at several steps
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
                    bx[i] = twiss_z.beta_x
                    by[i] = twiss_z.beta_y
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
                # print(k, h, bx)
            elif el.__class__ == Multipole:
                # Thin multipole
                twiss_z = el.transfer_map * tws_elem
                integr_x += twiss_z.beta_x * el.kn[1]
                integr_y -= twiss_z.beta_y * el.kn[1]
            # Move twiss forward
            tws_elem = el.transfer_map * tws_elem
    elif method == 'analytic':
        for el in lattice.sequence:
            if el.__class__ in [SBend, RBend, Bend, Quadrupole]:
                if el.l == 0:
                    raise Exception(f'Thin focusing element ({el.id}) encountered?')
                # Use beta at entrance to compute analytic average beta (see CERN-THESIS-2018-300)
                # 1/rho
                h = el.angle / el.l if el.__class__ != Quadrupole and el.l != 0 else 0.0
                # This is 1/bendradius^2 dipole horizontal focusing, since arc = angle * radius
                k1x = el.k1 + h * h
                k1y = -el.k1

                @functools.lru_cache(maxsize=512)
                def get_avg_beta_parameters(k1):
                    k1abs = np.abs(k1)
                    tworootkl = 2 * np.sqrt(k1abs) * el.l
                    if k1 > 0.0:
                        # Focusing
                        srtkl_over_rtkl = np.sin(tworootkl) / tworootkl
                        u0 = 0.5 * (1 + srtkl_over_rtkl)
                        u1 = (np.sin(np.sqrt(k1abs) * el.l) ** 2) / (k1abs * el.l)
                        u2 = (0.5 / k1abs) * (1 - srtkl_over_rtkl)
                    elif k1 < 0.0:
                        # Defocusing
                        srtkl_over_rtkld = np.sinh(tworootkl) / tworootkl
                        u0 = 0.5 * (1 + srtkl_over_rtkld)
                        u1 = (np.sinh(np.sqrt(k1abs) * el.l) ** 2) / (k1abs * el.l)
                        u2 = (-0.5 / k1abs) * (1 - srtkl_over_rtkld)
                    else:
                        # Limit as k -> 0 == drift, where beta(z) = beta0 - 2 * alpha0 *z + gamma0 * z^2
                        u0 = 1
                        u1 = el.l
                        u2 = el.l ** 2 / 3
                    return u0, u1, u2

                u0x, u1x, u2x = get_avg_beta_parameters(k1x)
                u0y, u1y, u2y = get_avg_beta_parameters(k1y)
                twiss_z = tws_elem
                beta_x_avg = twiss_z.beta_x * u0x - twiss_z.alpha_x * u1x + twiss_z.gamma_x * u2x
                beta_y_avg = twiss_z.beta_y * u0y - twiss_z.alpha_y * u1y + twiss_z.gamma_y * u2y
                ix = beta_x_avg * (el.k1 + h * h) * el.l
                iy = beta_y_avg * -el.k1 * el.l
                integr_x += ix
                integr_y += iy
                #print(tws_0.__dict__)
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
    else:
        raise Exception
    ksi_x = -integr_x / (4 * np.pi)
    ksi_y = -integr_y / (4 * np.pi)
    return np.array([ksi_x * n_superperiods, ksi_y * n_superperiods])


def edge_chromaticity(lattice, tws_0):
    """
    Computes contribution of sector and rectangle bend edges to chromaticity analytically
    # See https://home.fnal.gov/~ostiguy/OptiM/OptimHelp/dipole_edge.html
    # See doi:10.18429/JACoW-IPAC2018-TUPMK014
    # See LA-UR-11-10334
    """
    ksi_x_edge = 0.0
    ksi_y_edge = 0.0
    L = 0.0
    tws_elem = tws_0
    for i, el in enumerate(lattice.sequence):
        L += el.l
        tws_elem = el.transfer_map * tws_elem
        if isinstance(el, Edge):
            # TODO: integrate with EdgeUtil conventions
            alpha = el.edge
            # Horizontal focusing only depends on pole angle
            ksi_x_focusing = tws_elem.beta_x * np.tan(alpha) * el.h
            ksi_x_edge += ksi_x_focusing
            # Vertical edge focusing has contributions from soft edge fringe fields
            ksi_y_fringe = tws_elem.beta_y * el.gap * el.fint * (1 + np.sin(alpha) ** 2) / np.cos(alpha) * (el.h ** 2)
            ksi = el.gap * el.fint * el.h * (1 + np.sin(alpha) ** 2) / np.cos(alpha)
            #ksi_y_focusing = -tws_elem.beta_y * np.tan(alpha-ksi) * el.h
            #ksi_y_edge += ksi_y_focusing
            ksi_y_focusing = -tws_elem.beta_y * np.tan(alpha) * el.h
            ksi_y_edge += ksi_y_focusing + ksi_y_fringe
            print(el.h, el.gap, el.fint, el.h ** 2, ksi_y_fringe)
    return np.array([ksi_x_edge, ksi_y_edge])
