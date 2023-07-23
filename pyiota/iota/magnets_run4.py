import numpy as np
import scipy.constants

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sixdsim.io import KnobVariable


def calc_octupole_strengths(current: float, energy: float, mu0: float = 0.3, nn=17):
    """
    Calculated quasi-integrable insert strenth K3 distibution for given central current and beam energy
    :param current:
    :param energy:
    :return:
    """
    current = current / 1000  # Amps
    # ! phase advance over straight section
    # mu0 = 0.3  # moved to parameter
    # ! length of the straight section
    l0 = 1.8
    # ! number of nonlinear elements
    # nn = 17
    # ! (m^-1) strength parameter of octupole potential (for 1A/150MeV strength by default)
    alpha = 10000
    # ! dimentional parameter of nonlinear lens
    cn = 0.01
    # ! cut at multipole with power. (0) no cut, (1) quadrupole only, (3) quad+oct, (4) octupole only
    ncut = 4
    # ! type of magnet (0) thin, (1) thick, only works for octupoles (ncut=4)
    otype = 1
    # ! length of octupole for thick option. must be < l0/nn
    olen = 0.07
    # ! extra margin at the start and end of insert
    # margin = 0.0575;                      # ! extra margin at the start and end of insert
    margin = 0.0
    # ! equivalent strength of nonlinear lens, after matching expansion terms
    tn = olen * nn / l0 * 3 / 8 * (cn ** 2) * alpha
    musect = mu0 + 0.5
    f0 = l0 / 4.0 * (1.0 + 1.0 / np.tan(np.pi * mu0) ** 2)
    betae = l0 / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    alfae = l0 / 2.0 / f0 / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    betas = l0 * (1 - l0 / 4.0 / f0) / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    # beta_arr = np.zeros(nn)
    i = np.arange(1, nn + 1)
    sn = margin + (l0 - 2 * margin) / nn * (i - 0.5)
    bn = l0 * (1 - sn * (l0 - sn) / l0 / f0) / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    # beta_arr.append(bn)
    scaling_arr = 1 / (bn ** 3)
    # Apr 2020 - gradient change back to 0.7kG/cm^3 @1A
    # cal_factor means K3 in m^-4 for 1A current in central octupole
    cal_factor = (0.75 / 10 * 100 * 100 * 100) / (energy / (scipy.constants.c * 1e-6))
    # bn, scaling_arr
    currents = current * cal_factor * scaling_arr / max(scaling_arr)
    return currents, bn, scaling_arr / max(scaling_arr)


def get_dn_current(t, mu0=0.3, energy=150, GI=None):
    #if energy != 0:
        #print('BR recalc')
        #BR=335.26*1.5
    BR = 1000 * (energy / (scipy.constants.c * 1e-6))
    #
    # Constants
    l0 = 1.8  # Length of Straight Section`
    # cn = 0.01   #dimentional parameter
    cn = 0.008105461952
    nn = 18  # Number of Magnets
    L = 6.5  # Mag Length [cm]
    # L = 7.5  # Mag Length [cm]
    f0 = l0 / 4 * (1 + 1 / np.tan(np.pi * mu0) ** 2)  # IOTA Focus k
    beta_e = l0 / np.sqrt(1 - (1 - l0 / 2 / f0) ** 2)  # Beta at Entrance
    alfa_e = l0 / 2 / f0 / np.sqrt(1 - (1 - l0 / 2 / f0) ** 2)  # Alpha Function at entrance
    beta_s = l0 * (1 - l0 / 4 / f0) / np.sqrt(1 - (1 - l0 / 2 / f0) ** 2)
    # Array stuff
    i = np.arange(1, 19)  # Create 1-18 array
    sn = l0 / nn * (i - 0.5)  # Magnet distance
    bn = l0 * (1 - sn * (l0 - sn) / l0 / f0) / np.sqrt(1 - (1 - l0 / 2 / f0) ** 2)  # Beta at Magnet
    knn = l0 * t * cn ** 2 / nn / bn
    cnn = cn * np.sqrt(bn)
    k1 = 2 * knn / cnn ** 2
    if GI is None:
        GI = np.array([0.078015314, 0.08882103, 0.100544596, 0.114272161, 0.127323918, 0.142097491,
                       0.156935916, 0.168829748, 0.176310294, 0.176310294, 0.168829748, 0.156935916,
                       0.142097491, 0.127323918, 0.114272161, 0.100544596, 0.08882103, 0.078015314])
    I = k1 / GI / L * BR / 100
    return I, k1, bn


directions = np.array([[1, -1, 1, -1], [-1, 1, 1, -1], [1, 1, -1, -1]])


def calculate_current_margins(currents: np.ndarray, max_current: float = 1.9,
                              min_current: float = -1.9
                              ):
    margin_skew = np.abs(directions[0] * max_current - currents)
    margin_skew2 = np.abs(directions[0] * min_current - currents)

    margin_ch = np.abs(directions[1] * max_current - currents)
    margin_ch2 = np.abs(directions[1] * min_current - currents)

    margin_cv = np.abs(directions[2] * max_current - currents)
    margin_cv2 = np.abs(directions[2] * min_current - currents)

    return [(min(margin_skew), -min(margin_skew2)), (min(margin_ch), -min(margin_ch2)),
            (min(margin_cv), -min(margin_cv2))]


def get_combfun_coil_currents(skew: 'KnobVariable' = None,
                              ch: 'KnobVariable' = None,
                              cv: 'KnobVariable' = None
                              ):
    """
    Computes actual coil currents from respective virtual knob settings
    :param skew:
    :param ch:
    :param cv:
    :return:
    """
    currents = np.zeros(4)
    if skew:
        currents += np.array([1, -1, 1, -1]) * skew.value
    if ch:
        currents += np.array([-1, 1, 1, -1]) * ch.value
    if cv:
        currents += np.array([1, 1, -1, -1]) * cv.value
    return currents


def get_combfun_strengths(currents: np.ndarray):
    """
    Computes virtual H/V/skew from coil correctors
    :param currents:
    :return:
    """
    assert currents.ndim == 1 and len(currents) == 4
    skew = np.sum(currents * np.array([1, -1, 1, -1]))
    ch = np.sum(currents * np.array([-1, 1, 1, -1]))
    cv = np.sum(currents * np.array([1, 1, -1, -1]))
    return skew, ch, cv
