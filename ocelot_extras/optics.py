import numpy as np
from ocelot import *
from ocelot import transform_vec_ent, transform_vec_ext
from copy import copy


class NLKickTM(TransferMap):
    """
    Symplectic 2nd order integrator (aka drift-kick) through Danilov-Nagaitsev nonlinear potential.
    Can be considered as a more advanced version of KickTM.
    For NL lens:
     See general method in doi:10.1006/jcph.2000.6570
     See IMPACT-Z implementation - C. Mitchell et al., "Advances in Symplectic Tracking for Robust Modeling of Nonlinear Integrable Optics"
     See Synergia implementation - https://bitbucket.org/fnalacceleratormodeling/synergia2/src/devel3/src/synergia/libFF/ff_nllens.cc
    For exact drift:
     See CERN-THESIS-2013-248
     See http://pcwww.liv.ac.uk/~awolski/Teaching/Cockcroft/LinearDynamics/LinearDynamics-Lecture2.pdf
     See https://www.cockcroft.ac.uk/wp-content/uploads/2015/11/NonlinearDynamics-Part3.pdf
    """

    def __init__(self, knll, cnll, nkick=1):
        TransferMap.__init__(self)
        self.knll = knll
        self.cnll = cnll
        self.nkick = nkick

    def kick(self, X, l, knll, cnll, energy, nkick):
        kick_fraction = 1. / nkick
        l = l / nkick
        dl = l / 2

        # Relatistic gamma = Etotal/Erest
        gamma = energy / m_e_GeV
        if gamma != 0:
            gamma2 = gamma * gamma
            # KickTM has an approximation beta = 1. - 0.5 / gamma2 (from Taylor series), using exact here
            beta = np.sqrt(1.0 - 1.0 / gamma2)
            ibeta = 1. / beta
            gammabeta = np.sqrt(gamma2 - 1.0)
            igammabeta = 1. / gammabeta
            map_drift = self.map_drift_exact
            # Not an exact drift - it is linear first order only, but still symplectic
            # coef = 1.0 / (beta * beta * gamma2)
        else:
            # 4D - no energy
            ibeta = igammabeta = 0.
            map_drift = self.map_drift_4D

        for i in range(nkick):
            # Half-drift with applied offset
            map_drift(X, ibeta, igammabeta, dl)
            X[0] -= self.dx
            X[2] -= self.dy

            # Nonlinear kick
            self.map_nllens_thin(X, knll * kick_fraction, cnll)

            # Second half-drift and revert offset
            map_drift(X, ibeta, igammabeta, dl)
            X[0] += self.dx
            X[2] += self.dy

    def map_drift_exact(self, X, ibeta, igammabeta, l):
        # Time is cdt (earlier is negative), opposite of MADX and Sixtrack, same as ynergia)
        # deltaE/p same as MADX (NOT deltaP/p like in some refs, be careful about how delta is defined)
        x, px, y, py, cdt, deop = X

        # # If we had dpop, then:
        # # 1 + delta:
        # dp = dpop + uni
        # # 1/pz from thesis above:
        # inv_npz = 1.0 / np.sqrt(dp * dp - xp * xp - yp * yp)
        # D2 = lxpr * lxpr + l * l + lypr * lypr
        # # Actual momentum of particle
        # p = dp * vrm
        # E^2 = mom^2 + rest_mass^2
        # E2 = p * p + vm * vm;
        # # 1/beta^2 = 1/(1-1/gamma^2) = gamma^2/(gamma^2-1) = Etot^2/Erest^2 / (Etot^2/Erest^2 - 1) = Etot^2 / (Etot^2 - Erest^2) = Etot^2/Ekin^2 = E2/p^2
        # ibeta2 = E2 / (p * p);
        # sig = np.sign(l)
        # cdt = cdt + sig * sqrt(D2 * ibeta2) - reference_cdt;

        # But with deop have to do:
        ibeta_p_deop = ibeta + deop
        inv_npz = 1.0 / np.sqrt(ibeta_p_deop * ibeta_p_deop - px * px - py * py - igammabeta * igammabeta)

        # new x,y with added l*x',l*y' (px, py unchanged)
        X[0] += px * l * inv_npz
        X[2] += py * l * inv_npz
        # new cdt assuming reference time = design time
        # TODO: treat closed orbit
        X[4] -= (ibeta - ibeta_p_deop * inv_npz) * l

    def map_drift_4D(self, X, _, __, l):
        X[0] += X[1] * l
        X[2] += X[3] * l

    def map_nllens_thin(self, X, knll, cnll):
        icnll = 1.0 / cnll
        x = X[0] * icnll
        y = X[2] * icnll
        kick = -knll * icnll
        if y == 0.0 and abs(x) >= 1:
            raise Exception('Encountered branch cut in NLLENS transport')
        dF = self.Fderivative(x, y)
        dPx = kick * np.real(dF)
        dPy = -kick * np.imag(dF)
        X[1] += dPx
        X[3] += dPy

    def Fderivative(self, x, y):
        """
        Computes the derivative of the dimensionless complex potential for
        the IOTA nonlinear insert.
        Note Python-specific performance tuning
        """
        z = x + y * 1.0j
        denom = np.sqrt((1. + 0.j) - z * z)  # INLINE complex_root(z)
        fd = z / (denom * denom) + np.arcsin(z) / (denom * denom * denom)  # MULT faster than POW
        return fd

    # def complex_root(z):
    #     """ returns sqrt(1-z^2) """
    #     return np.sqrt((1.+0.j)-z**2)

    def kick_apply(self, X, l, knll, cnll, energy, nkick, dx, dy, tilt):
        """ Applies kicks and the entrance/exit coordinate transforms """
        if dx != 0 or dy != 0 or tilt != 0:
            X = transform_vec_ent(X, dx, dy, tilt)
        self.kick(X, l, knll, cnll, energy, nkick=nkick)
        if dx != 0 or dy != 0 or tilt != 0:
            X = transform_vec_ext(X, dx, dy, tilt)
        return X

    def __call__(self, s):
        m = copy(self)
        m.length = s
        m.R = lambda energy: m.R_z(s, energy)
        m.B = lambda energy: m.B_z(s, energy)
        m.delta_e = m.delta_e_z(s)
        m.map = lambda X, energy: m.kick_apply(X, s, m.knll, m.cnll, energy, m.nkick, m.dx, m.dy, m.tilt)
        return m