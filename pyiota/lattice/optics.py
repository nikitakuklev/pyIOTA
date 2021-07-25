import numpy as np
from ocelot import *
from ocelot import transform_vec_ent, transform_vec_ext
from copy import copy


class MultipoleKickTM(TransferMap):
    """ See NLKickTM for comments """

    def __init__(self, angle=0., k1=0., k2=0., k3=0., nkick=1, drift_type=1):
        TransferMap.__init__(self)
        assert angle == 0.0
        self.angle = angle
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.nkick = nkick
        self.drift_type = drift_type

    def kick(self, X, l, angle, k1, k2, k3, energy, nkick=1):
        kick_fraction = 1. / nkick
        l = l / nkick
        dl = l / 2

        gamma = energy / m_e_GeV
        if gamma != 0:
            gamma2 = gamma * gamma
            beta = np.sqrt(1.0 - 1.0 / gamma2)
            ibeta = 1. / beta
            gammabeta = np.sqrt(gamma2 - 1.0)
            igammabeta = 1. / gammabeta
            if self.drift_type == 1:
                map_drift = self.map_drift_6D_exact
            elif self.drift_type == 2:
                map_drift = self.map_drift_6D_paraxial
            else:
                map_drift = self.map_drift_6D_linear
        else:
            # 4D - no energy
            ibeta = igammabeta = 0.
            map_drift = self.map_drift_4D

        # Integrated strength
        k1 = k1 * l
        k2 = k2 * l / 2.
        k3 = k3 * l / 6.

        X[:, np.abs(X[0, :]) > 0.5] = np.nan
        X[:, np.abs(X[1, :]) > 0.5] = np.nan
        X[:, np.abs(X[2, :]) > 0.5] = np.nan
        X[:, np.abs(X[3, :]) > 0.5] = np.nan

        map_kick = self.map_mult_thin

        for i in range(nkick):
            # Half-drift with applied offset
            map_drift(X, ibeta, igammabeta, dl)
            #X[0] -= self.dx
            #X[2] -= self.dy

            # Kick
            map_kick(X, k1 * kick_fraction, k2 * kick_fraction, k3 * kick_fraction)

            # Second half-drift and revert offset
            map_drift(X, ibeta, igammabeta, dl)
            #X[0] += self.dx
            #X[2] += self.dy

    def map_mult_thin(self, X: np.ndarray, k1, k2, k3):
        xy1 = X[0, :] + 1j * X[2, :]
        xy2 = xy1 * xy1
        xy3 = xy2 * xy1
        kick = k1 * xy1 + k2 * xy2 + k3 * xy3
        X[1, :] += -np.real(kick)
        X[3, :] += np.imag(kick)

    def map_drift_6D_exact(self, X: np.ndarray, ibeta: float, igammabeta: float, l: float):
        ibeta_plus_deop = ibeta + X[5]
        inv_npz = 1.0 / np.sqrt(ibeta_plus_deop * ibeta_plus_deop - X[1] * X[1] - X[3] * X[3] - igammabeta * igammabeta)
        X[0] += X[1] * l * inv_npz
        X[2] += X[3] * l * inv_npz
        X[4] -= (ibeta - ibeta_plus_deop * inv_npz) * l

    def map_drift_6D_paraxial(self, X: np.ndarray, ibeta: float, igammabeta: float, l: float):
        ibeta_plus_deop = ibeta + X[5]
        inv_npz = 1.0 / np.sqrt(ibeta_plus_deop * ibeta_plus_deop - igammabeta * igammabeta)
        X[0] += X[1] * l * inv_npz
        X[2] += X[3] * l * inv_npz
        X[4] -= l * (
                ibeta - (1.0 + 0.5 * (X[1] * X[1] + X[3] * X[3]) * inv_npz * inv_npz) * (inv_npz * ibeta_plus_deop))

    def map_drift_6D_linear(self, X: np.ndarray, ibeta: float, igammabeta: float, l: float):
        X[0] += X[1] * l
        X[2] += X[3] * l
        X[4] -= X[5] * l * igammabeta * igammabeta

    def map_drift_4D(self, X: np.ndarray, ibeta: float, igammabeta: float, l: float):
        X[0] += X[1] * l
        X[2] += X[3] * l

    def kick_apply(self, X, l, angle, k1, k2, k3, energy, nkick, dx, dy, tilt):
        if dx != 0 or dy != 0 or tilt != 0:
            X = transform_vec_ent(X, dx, dy, tilt)
        self.kick(X, l, angle, k1, k2, k3, energy, nkick=nkick)
        if dx != 0 or dy != 0 or tilt != 0:
            X = transform_vec_ext(X, dx, dy, tilt)
        return X

    def __call__(self, s):
        m = copy(self)
        m.length = s
        m.R = lambda energy: m.R_z(s, energy)
        m.B = lambda energy: m.B_z(s, energy)
        m.delta_e = m.delta_e_z(s)
        m.map = lambda X, energy: m.kick_apply(X, s, m.angle, m.k1, m.k2, m.k3, energy, m.nkick, m.dx, m.dy, m.tilt)
        return m


class NLKickTM(TransferMap):
    """
    Symplectic 2nd order integrator (aka drift-kick) through Danilov-Nagaitsev nonlinear potential.
    Can be considered as a more advanced version of KickTM with exact drifts.

    For NL lens:
     See potential in V. Danilov and S. Nagaitsev, PRSTAB 13, 084002 (2010)
     See general method in doi:10.1006/jcph.2000.6570
     See IMPACT-Z implementation - C. Mitchell et al., "Advances in Symplectic Tracking for Robust Modeling of Nonlinear Integrable Optics"
     See Synergia implementation - https://bitbucket.org/fnalacceleratormodeling/synergia2/src/devel3/src/synergia/libFF/ff_nllens.cc

    For exact drift:
     See CERN-THESIS-2013-248
     See http://pcwww.liv.ac.uk/~awolski/Teaching/Cockcroft/LinearDynamics/LinearDynamics-Lecture2.pdf
     See https://www.cockcroft.ac.uk/wp-content/uploads/2015/11/NonlinearDynamics-Part3.pdf

    nkick - Number of drift-kick-drift sections
    drift_type - Type of drift to use (6D only)
    """

    def __init__(self, knll, cnll, nkick=1, drift_type=1):
        TransferMap.__init__(self)
        self.knll = knll
        self.cnll = cnll
        self.nkick = nkick
        self.drift_type = drift_type

    def kick(self, X: np.ndarray, l: float, knll: float, cnll: float, energy: float, nkick: int):
        # print(f'NLKickMap with knll ({knll}) and cnll ({cnll}), X vector size ({X.shape})')
        kick_fraction = 1. / nkick
        l = l / nkick
        dl = l / 2

        # gamma0 = Etotal/Erest
        gamma = energy / m_e_GeV
        if gamma != 0:
            gamma2 = gamma * gamma
            # KickTM has approximation beta = 1. - 0.5 / gamma2, using exact
            beta = np.sqrt(1.0 - 1.0 / gamma2)
            ibeta = 1. / beta
            gammabeta = np.sqrt(gamma2 - 1.0)
            igammabeta = 1. / gammabeta
            # Drift type in various approximations - linear is fastest
            if self.drift_type == 1:
                map_drift = self.map_drift_6D_exact
            elif self.drift_type == 2:
                map_drift = self.map_drift_6D_paraxial
            else:
                map_drift = self.map_drift_6D_linear
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

    def map_drift_6D_exact(self, X: np.ndarray, ibeta: float, igammabeta: float, l: float):
        """
        OCELOT units are (x, x', y, y', cdt, deop) = X
        also known as (x, px/p0, y, py/p0, cdt, pt/cp0)
        Time is cdt (earlier is negative), opposite of MADX and Sixtrack, same as Synergia)
        deltaE/p same as MADX (NOT deltaP/p like in some refs, be careful about how delta is defined)
        """
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
        ibeta_plus_deop = ibeta + X[5]
        inv_npz = 1.0 / np.sqrt(ibeta_plus_deop * ibeta_plus_deop - X[1] * X[1] - X[3] * X[3] - igammabeta * igammabeta)

        # new x,y with added l*x',l*y' (px, py unchanged)
        X[0] += X[1] * l * inv_npz
        X[2] += X[3] * l * inv_npz

        # new cdt assuming reference time = design time
        # TODO: treat distorted closed orbit reference particle
        X[4] -= (ibeta - ibeta_plus_deop * inv_npz) * l

    def map_drift_6D_paraxial(self, X: np.ndarray, ibeta: float, igammabeta: float, l: float):
        """ Expanded Hamiltonian to second order using (px, py << (1 + dpop)) """
        # (x, x', y, y', cdt, deop) = X
        # also known as (x, px/p0, y, py/p0, cdt, pt/cp0)

        # 'expanded' drift space case from above thesis
        ibeta_plus_deop = ibeta + X[5]
        inv_npz = 1.0 / np.sqrt(ibeta_plus_deop * ibeta_plus_deop - igammabeta * igammabeta)

        # new x,y with added l*x',l*y' (px, py unchanged)
        X[0] += X[1] * l * inv_npz
        X[2] += X[3] * l * inv_npz

        # inv_npz*(1/beta0 + deop) = 1/beta
        X[4] -= l * (
                ibeta - (1.0 + 0.5 * (X[1] * X[1] + X[3] * X[3]) * inv_npz * inv_npz) * (inv_npz * ibeta_plus_deop))

    def map_drift_6D_linear(self, X: np.ndarray, ibeta: float, igammabeta: float, l: float):
        """ This is what OCELOT uses in KickTM """
        X[0] += X[1] * l
        X[2] += X[3] * l
        X[4] -= X[5] * l * igammabeta * igammabeta

    def map_drift_4D(self, X: np.ndarray, ibeta: float, igammabeta: float, l: float):
        """ Linear 4D case """
        X[0] += X[1] * l
        X[2] += X[3] * l

    def map_nllens_thin(self, X: np.ndarray, knll: float, cnll: float):
        """ Single DN lens kick """
        icnll = 1.0 / cnll
        x = X[0, :] * icnll
        y = X[2, :] * icnll
        kick = -knll * icnll
        # print(x, y)
        if np.any(y == 0.0) and np.any(np.abs(x) >= 1):
            # TODO: recheck what actual domain is
            raise Exception('Encountered branch cut in NLLENS transport')
        dF = self.Fderivative(x, y)
        # dPx = kick * np.real(dF)
        # dPy = -kick * np.imag(dF)
        X[1] += kick * np.real(dF)
        X[3] += -kick * np.imag(dF)

    def Fderivative(self, x: float, y: float) -> np.complex128:
        """
        Computes the derivative of the dimensionless complex potential for
        the IOTA nonlinear insert. Note Python-specific performance tuning.
        """
        z = x + y * 1.0j
        denom = np.sqrt((1. + 0.j) - z * z)  # INLINE complex_root(z)
        fd = z / (denom * denom) + np.arcsin(z) / (denom * denom * denom)  # MULT faster than POW
        return fd

    # def complex_root(z):
    #     """ returns sqrt(1-z^2) """
    #     return np.sqrt((1.+0.j)-z**2)

    def kick_apply(self, X: np.ndarray, l: float, knll: float, cnll: float, energy: float, nkick: int,
                   dx: float, dy: float, tilt: float) -> np.ndarray:
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
