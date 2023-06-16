import functools
import logging
from copy import copy, deepcopy
from math import factorial

import numpy as np
import torch
from ocelot import CorrectorTM, Drift, Edge, Hcor, KickTM, Matrix, Monitor, Multipole, Particle, \
    ParticleArray, \
    Quadrupole, SecondOrderMult, Sextupole, TransferMap, \
    Twiss, Vcor, \
    m_e_GeV, t_nnn

logger = logging.getLogger(__name__)


class TorchTransferMap(TransferMap):
    def __init__(self):
        self.dx = 0.
        self.dy = 0.
        self.tilt = 0.
        self.length = 0
        self.hx = 0.
        self.delta_e = 0.0
        self.delta_e_z = lambda z: 0.0
        # 6x6 linear transfer matrix
        self.R = lambda energy: torch.eye(6)
        self.R_z = lambda z, energy: torch.zeros((6, 6))
        self.B_z = lambda z, energy: torch.dot((torch.eye(6) - self.R_z(z, energy)), torch.array(
                [[self.dx], [0.], [self.dy], [0.], [0.], [0.]]))
        self.B = lambda energy: self.B_z(self.length, energy)
        self.map = lambda u, energy: self.mul_p_array(u, energy=energy)

    def map_x_twiss(self, tws0):
        E = tws0.E
        M = self.R(E)
        zero_tol = 1.e-10
        if abs(self.delta_e) > zero_tol:
            Ei = tws0.E
            Ef = tws0.E + self.delta_e
            k = torch.sqrt(Ef / Ei)
            M[0, 0] = M[0, 0] * k
            M[0, 1] = M[0, 1] * k
            M[1, 0] = M[1, 0] * k
            M[1, 1] = M[1, 1] * k
            M[2, 2] = M[2, 2] * k
            M[2, 3] = M[2, 3] * k
            M[3, 2] = M[3, 2] * k
            M[3, 3] = M[3, 3] * k
            E = Ef

        m = tws0
        tws = Twiss(tws0)
        tws.E = E
        tws.p = m.p
        tws.beta_x = M[0, 0] * M[0, 0] * m.beta_x - 2 * M[0, 1] * M[0, 0] * m.alpha_x + M[
            0, 1] * M[0, 1] * m.gamma_x
        # tws.beta_x = ((M[0,0]*tws.beta_x - M[0,1]*m.alpha_x)**2 + M[0,1]*M[0,1])/m.beta_x
        tws.beta_y = M[2, 2] * M[2, 2] * m.beta_y - 2 * M[2, 3] * M[2, 2] * m.alpha_y + M[
            2, 3] * M[2, 3] * m.gamma_y
        # tws.beta_y = ((M[2,2]*tws.beta_y - M[2,3]*m.alpha_y)**2 + M[2,3]*M[2,3])/m.beta_y
        tws.alpha_x = -M[0, 0] * M[1, 0] * m.beta_x + (
                M[0, 1] * M[1, 0] + M[1, 1] * M[0, 0]) * m.alpha_x - M[0, 1] * M[
                          1, 1] * m.gamma_x
        tws.alpha_y = -M[2, 2] * M[3, 2] * m.beta_y + (
                M[2, 3] * M[3, 2] + M[3, 3] * M[2, 2]) * m.alpha_y - M[2, 3] * M[
                          3, 3] * m.gamma_y

        tws.gamma_x = (1. + tws.alpha_x * tws.alpha_x) / tws.beta_x
        tws.gamma_y = (1. + tws.alpha_y * tws.alpha_y) / tws.beta_y

        tws.Dx = M[0, 0] * m.Dx + M[0, 1] * m.Dxp + M[0, 5]
        tws.Dy = M[2, 2] * m.Dy + M[2, 3] * m.Dyp + M[2, 5]

        tws.Dxp = M[1, 0] * m.Dx + M[1, 1] * m.Dxp + M[1, 5]
        tws.Dyp = M[3, 2] * m.Dy + M[3, 3] * m.Dyp + M[3, 5]
        denom_x = M[0, 0] * m.beta_x - M[0, 1] * m.alpha_x
        if denom_x == 0.:
            d_mux = torch.pi / 2. * M[0, 1] / torch.abs(M[0, 1])
        else:
            d_mux = torch.arctan(M[0, 1] / denom_x)

        if d_mux < 0:
            d_mux += torch.pi
        tws.mux = m.mux + d_mux
        denom_y = M[2, 2] * m.beta_y - M[2, 3] * m.alpha_y
        if denom_y == 0.:
            d_muy = torch.pi / 2. * M[2, 3] / torch.abs(M[2, 3])
        else:
            d_muy = torch.arctan(M[2, 3] / denom_y)
        if d_muy < 0:
            d_muy += torch.pi
        tws.muy = m.muy + d_muy

        return tws

    def mul_p_array(self, rparticles, energy=0.):
        a = torch.add(torch.dot(self.R(energy), rparticles), self.B(energy))
        rparticles[:] = a[:]
        return rparticles

    def __mul__(self, m):
        """
        :param m: TransferMap, Particle or Twiss
        :return: TransferMap, Particle or Twiss
        Ma = {Ba, Ra, Ta}
        Mb = {Bb, Rb, Tb}
        X1 = R*(X0 - dX) + dX = R*X0 + B
        B = (E - R)*dX
        """

        if m.__class__ in [TransferMap]:
            m2 = TransferMap()
            m2.R = lambda energy: torch.dot(self.R(energy), m.R(energy))
            m2.B = lambda energy: torch.dot(self.R(energy), m.B(energy)) + self.B(energy)
            m2.length = m.length + self.length

            return m2

        elif m.__class__ == Particle:
            self.apply(m)
            return deepcopy(m)

        elif m.__class__ == Twiss:
            tws = self.map_x_twiss(m)
            tws.s = m.s + self.length
            return tws

        else:
            raise Exception(
                    " TransferMap.__mul__: unknown object in transfer map multiplication: " + str(
                            m.__class__.__name__))

    def apply(self, prcl_series):
        """
        :param prcl_series: can be list of Particles [Particle_1, Particle_2, ... ] or ParticleArray
        :return: None
        """
        if prcl_series.__class__ == ParticleArray:
            self.map(prcl_series.rparticles, energy=prcl_series.E)
            prcl_series.E += self.delta_e
            prcl_series.s += self.length

        elif prcl_series.__class__ == Particle:
            p = prcl_series
            p.x, p.px, p.y, p.py, p.tau, p.p = self.map(
                    torch.tensor([[p.x], [p.px], [p.y], [p.py], [p.tau], [p.p]]), p.E)[:, 0]
            p.s += self.length
            p.E += self.delta_e

        elif prcl_series.__class__ == list and prcl_series[0].__class__ == Particle:
            # If the energy is not the same (p.E) for all Particles in the list of Particles
            # in that case cycle is applied. For particles with the same energy p.E
            list_e = torch.tensor([p.E for p in prcl_series])
            if False in (list_e[:] == list_e[0]):
                for p in prcl_series:
                    self.map(torch.tensor([[p.x], [p.px], [p.y], [p.py], [p.tau], [p.p]]),
                             energy=p.E)
                    p.E += self.delta_e
                    p.s += self.length
            else:
                pa = ParticleArray()
                pa.list2array(prcl_series)
                pa.E = prcl_series[0].E
                self.map(pa.rparticles, energy=pa.E)
                pa.E += self.delta_e
                pa.s += self.length
                pa.array2ex_list(prcl_series)

        else:
            raise Exception(" TransferMap.apply(): Unknown type of Particle_series: " + str(
                    prcl_series.__class__.__name))

    def __call__(self, s):
        m = copy(self)
        m.length = s
        m.R = lambda energy: m.R_z(s, energy)
        m.B = lambda energy: m.B_z(s, energy)
        m.delta_e = m.delta_e_z(s)
        m.map = lambda u, energy: m.mul_p_array(u, energy=energy)
        return m


class TorchMethodTM:
    def __init__(self, params=None):
        if params is None:
            self.params = {'global': TransferMap}
        else:
            self.params = params

        if "global" in self.params:
            self.global_method = self.params['global']
        else:
            self.global_method = TransferMap
        self.sec_order_mult = SecondOrderMult()
        self.nkick = self.params['nkick'] if 'nkick' in self.params else 1
        self.edrift_method = self.params.get('edrift_method', None)

    def create_tm(self, element):
        if element.__class__ in self.params:
            transfer_map = self.set_tm(element, self.params[element.__class__])
        else:
            transfer_map = self.set_tm(element, self.global_method)
        return transfer_map

    def set_tm(self, element, method):
        dx = element.dx
        dy = element.dy
        tilt = element.dtilt + element.tilt
        if element.l == 0:
            hx = 0.
        else:
            hx = element.angle / element.l

        r_z_e = create_r_matrix(element)

        # global method
        if method == KickTM:
            try:
                k3 = element.k3
            except:
                k3 = 0.
            tm = KickTM(angle=element.angle, k1=element.k1, k2=element.k2, k3=k3, nkick=self.nkick)

        else:
            tm = TorchMethodTM()

        # if element.__class__ == Multipole:
        #     tm = MultipoleTM(kn=element.kn)

        if element.__class__ == Hcor:
            t_mat_z_e = lambda z, energy: t_nnn(z, 0, 0, 0, energy)
            tm = CorrectorTM(angle_x=element.angle, angle_y=0., r_z_no_tilt=r_z_e,
                             t_mat_z_e=t_mat_z_e)
            tm.multiplication = self.sec_order_mult.tmat_multip

        if element.__class__ == Vcor:
            t_mat_z_e = lambda z, energy: t_nnn(z, 0, 0, 0, energy)
            tm = CorrectorTM(angle_x=0, angle_y=element.angle, r_z_no_tilt=r_z_e,
                             t_mat_z_e=t_mat_z_e)
            tm.multiplication = self.sec_order_mult.tmat_multip

        # if element.__class__ == EDrift:
        #     if self.edrift_method:
        #         tm = EDriftTM(method=self.edrift_method)
        #     else:
        #         tm = EDriftTM(method=element.method)

        tm.length = element.l
        tm.dx = dx
        tm.dy = dy
        tm.tilt = tilt
        if tilt == 0.0:
            tm.R_z = lambda z, energy: r_z_e(z, energy)
        else:
            tm.R_z = lambda z, energy: torch.dot(torch.dot(rot_mtx(-tilt), r_z_e(z, energy)),
                                                 rot_mtx(tilt))
        tm.R = lambda energy: tm.R_z(element.l, energy)
        tm.R0 = r_z_e(element.l, 0.0)
        # tm.B_z = lambda z, energy: dot((eye(6) - tm.R_z(z, energy)), array([dx, 0., dy, 0., 0., 0.]))
        # tm.B = lambda energy: tm.B_z(element.l, energy)

        return tm


def rot6D(angle):
    if angle == 0.0:
        return torch.eye(6)
    cs = torch.cos(angle)
    sn = torch.sin(angle)
    return torch.tensor([[cs, 0., sn, 0., 0., 0.],
                         [0., cs, 0., sn, 0., 0.],
                         [-sn, 0., cs, 0., 0., 0.],
                         [0., -sn, 0., cs, 0., 0.],
                         [0., 0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 0., 1.]])


@functools.lru_cache(maxsize=None)
def uni_matrix(z, k1, hx, sum_tilts=0., energy=0.):
    """
    universal matrix. The function creates R-matrix from given parameters.
    r = element.l/element.angle
    +K - focusing lens, -K - defoc

    :param z: element length [m]
    :param k1: quadrupole strength [1/m**2]
    :param hx: the curvature (1/r) of the element [1/m]
    :param sum_tilts: rotation relative to longitudinal axis [rad]
    :param energy: the beam energy [GeV]
    :return: R-matrix [6, 6]
    """

    gamma = energy / m_e_GeV

    kx2 = (k1 + hx * hx)
    ky2 = -k1
    print(kx2, ky2, k1, hx)
    kx2c = torch.tensor(kx2 + 0.j)
    ky2c = torch.tensor(ky2 + 0.j)
    kx = torch.sqrt(kx2c)
    ky = torch.sqrt(ky2c)
    cx = torch.cos(z * kx).real
    cy = torch.cos(z * ky).real
    sy = (torch.sin(ky * z) / ky).real if ky != 0 else z

    igamma2 = torch.tensor(0.)

    if gamma != 0:
        igamma2 = 1. / (gamma * gamma)

    beta = torch.sqrt(1. - igamma2)

    if kx != 0:
        sx = (torch.sin(kx * z) / kx).real
        dx = hx / kx2 * (1. - cx)
        r56 = hx * hx * (z - sx) / kx2 / beta ** 2
    else:
        sx = z
        dx = z * z * hx / 2.
        r56 = hx * hx * z ** 3 / 6. / beta ** 2

    r56 -= z / (beta * beta) * igamma2

    u_matrix = torch.tensor([[cx, sx, 0., 0., 0., dx / beta],
                             [-kx2 * sx, cx, 0., 0., 0., sx * hx / beta],
                             [0., 0., cy, sy, 0., 0.],
                             [0., 0., -ky2 * sy, cy, 0., 0.],
                             [hx * sx / beta, dx / beta, 0., 0., 1., r56],
                             [0., 0., 0., 0., 0., 1.]])
    if sum_tilts != 0:
        u_matrix = torch.dot(torch.dot(rot_mtx(-sum_tilts), u_matrix), rot_mtx(sum_tilts))
    return u_matrix


def build_model(el, parameters: dict = None):
    z = el.l

    # r_z_e = lambda z, energy: uni_matrix(z, k1, hx=hx, sum_tilts=0, energy=energy)
    B = torch.zeros(4, 1)
    r_z_e = torch.eye(4)
    mp = []
    do_transforms = False
    if el.dx != 0 or el.dy != 0 or el.tilt != 0:
        do_transforms = True
    if el.__class__ == Quadrupole:
        # print(parameters)
        k1 = parameters.get(f'{el.id}.k1', el.k1)
        if z > 0:
            r_z_e = uni4D_noh(z, k1, sum_tilts=0)
        else:
            r_z_e = uni4Dthin(k1, sum_tilts=0)
        B = torch.zeros(4, 1)
    elif el.__class__ == Hcor:
        if z > 0:
            r_z_e = uni4D_noh(z, 0.0, sum_tilts=0)
        else:
            r_z_e = torch.eye(4) #uni4Dthin(0.0, sum_tilts=0)
        B = torch.zeros(4, 1)
        B[1, 0] = el.angle
    elif el.__class__ == Vcor:
        if z > 0:
            r_z_e = uni4D_noh(z, 0, sum_tilts=0)
        else:
            r_z_e = torch.eye(4)#uni4Dthin(0.0, sum_tilts=0)
        B = torch.zeros(4, 1)
        B[3, 0] = el.angle
    elif el.__class__ == Sextupole:
        assert z == 0
        maps = []

        if do_transforms:
            dx = parameters.get(f'{el.id}.dx', el.dx)
            dy = parameters.get(f'{el.id}.dy', el.dy)

            # maps.append([transform_vec_ent, dict(dx=dx, dy=dy, tilt=el.tilt)])
            # maps.append([kick4DthinK2, dict(k2=el.k2)])
            # maps.append([transform_vec_ext, dict(dx=dx, dy=dy, tilt=el.tilt)])
            maps.append(functools.partial(transform_vec_ent, dx=dx, dy=dy, tilt=el.tilt))
            maps.append(functools.partial(kick4DthinK2, k2=el.k2))
            maps.append(functools.partial(transform_vec_ext, dx=dx, dy=dy, tilt=el.tilt))
        else:
            #print('hi', parameters.get(f'{el.id}.dx', el.dx))
            #maps.append([kick4DthinK2, dict(k2=el.k2)])
            maps.append(functools.partial(kick4DthinK2, k2=el.k2))
        mp = maps
    # elif el.__class__ == Matrix:
    #     rm = torch.eye(6)
    #     rm = el.r
    #
    #     def r_matrix(z, l, rm):
    #         if z < l:
    #             r_z = uni4D_noh(z, 0)
    #         else:
    #             r_z = rm
    #         return r_z
    #
    #     r_z_e = lambda z, energy: r_matrix(z, el.l, rm)
    #     B = torch.zeros(4, 1)
    elif el.__class__ == Drift:
        r_z_e = uni4Ddrift(z)
        B = torch.zeros(4, 1)
    # elif el.__class__ == Multipole:
    #     raise
    #     r = torch.eye(6)
    #     r[1, 0] = -el.kn[1]
    #     r[3, 2] = el.kn[1]
    #     r[1, 5] = el.kn[0]
    #     r_z_e = lambda z, energy: r
    #     B = torch.zeros(4, 1)
    elif el.__class__ == Monitor:
        return
    else:
        raise Exception(f'Weird element {el}')
    return (r_z_e, B, mp)


def forward(particle, R, B):
    Pnew = torch.matmul(R, particle)
    Pnew2 = torch.add(Pnew, B)
    return Pnew2


def transform_vec_ent(X, dx, dy, tilt):
    # rotmat = rot_mtx(tilt)
    B = torch.zeros(4, 1)
    B[0, 0] = -dx
    B[2, 0] = -dy
    X2 = torch.add(X, B)
    # X[:] = np.dot(rotmat, x_add)[:]
    return X2


def transform_vec_ext(X, dx, dy, tilt):
    # rotmat = rot_mtx(-tilt)
    # x_tilt = np.dot(rotmat, X)
    B = torch.zeros(4, 1)
    B[0, 0] = dx
    B[2, 0] = dy
    # X[:] = np.add(x_tilt, np.array([[dx], [0.], [dy], [0.], [0.], [0.]]))[:]
    X2 = torch.add(X, B)
    return X2


# def kick4Dthin(self, X, kn):
#     p = -kn[0] * X[5] + 0j
#     for n in range(1, len(kn)):
#         p += kn[n] * (X[0] + 1j * X[2]) ** n / factorial(n)
#     X[1] = X[1] - np.real(p)
#     X[3] = X[3] + np.imag(p)
#     X[4] = X[4] - kn[0] * X[0]
#     return X

def kick4Dthin(X, kn):
    p = 0.0
    for n in range(1, len(kn)):
        p += kn[n] * (X[0] + 1j * X[2]) ** n / factorial(n)
    X[1] = X[1] - np.real(p)
    X[3] = X[3] + np.imag(p)
    return X


def kick4DthinK2(X, k2):
    # p = k3 * (X[0] + 1.0j * X[2]) ** 2
    # X[1] = X[1] - np.real(k3 * (X[0] + 1.0j * X[2])(X[0] + 1.0j * X[2]))
    # X[1] = X[1] - np.real(k3 * (X[0]*X[0] + 2*X[0]*X[2]*1.0j - X[2]*X[2]))
    B = torch.zeros(4, 1)
    B[0, 0] = - (X[0] * X[0] - X[2] * X[2])
    B[2, 0] = 2 * X[0] * X[2]
    X2 = torch.add(X, k2 * B)
    return X2

def kick4DthinK2combined(X, k2, dx, dy, tilt):
    B = torch.zeros(4, 1)
    B[0, 0] = -dx
    B[2, 0] = -dy
    X = torch.add(X, B)

    B = torch.zeros(4, 1)
    B[0, 0] = - k2 * (X[0] * X[0] - X[2] * X[2])
    B[2, 0] = (k2 * 2 * X[0] * X[2])
    X = torch.add(X, B)

    B = torch.zeros(4, 1)
    B[0, 0] = dx
    B[2, 0] = dy
    X = torch.add(X, B)
    return X

# def kick(self, X, l, angle, k1, k2, k3, energy, nkick=1):
#     """
#     does not work for dipole
#     """
#     gamma = energy / m_e_GeV
#     coef = 0.
#     beta = 1.
#     if gamma != 0:
#         gamma2 = gamma * gamma
#         beta = 1. - 0.5 / gamma2
#         coef = 1. / (beta * beta * gamma2)
#     l = l / nkick
#     angle = angle / nkick
#
#     dl = l / 2.
#     k1 = k1 * l
#     k2 = k2 * l/2.
#     k3 = k3 * l/6.
#
#     for i in range(nkick):
#         x = X[0] + X[1] * dl
#         y = X[2] + X[3] * dl
#
#         p = -angle * X[5] + 0j
#         xy1 = x + 1j * y
#         xy2 = xy1 * xy1
#         xy3 = xy2 * xy1
#         p += k1 * xy1 + k2 * xy2 + k3 * xy3
#         X[1] = X[1] - np.real(p)
#         X[3] = X[3] + np.imag(p)
#         X[4] = X[4] + np.real(angle * xy1)/beta - X[5] * dl * coef
#
#         X[0] = x + X[1] * dl
#         X[2] = y + X[3] * dl
#         #X[4] -= X[5] * dl * coef
#     return X

def rot4D(angle):
    if angle == 0.0:
        return torch.eye(4)
    cs = torch.cos(angle)
    sn = torch.sin(angle)
    return torch.tensor([[cs, 0., sn, 0.],
                         [0., cs, 0., sn],
                         [-sn, 0., cs, 0.],
                         [0., -sn, 0., cs], ])


def uni4Dthin(k1, sum_tilts=0.):
    u_matrix = torch.zeros(4, 4)
    u_matrix[0, 0] = 1.0
    u_matrix[1, 0] = -k1
    u_matrix[1, 1] = 1.0

    u_matrix[2, 2] = 1.0
    u_matrix[3, 2] = k1
    u_matrix[3, 3] = 1.0

    #if sum_tilts != 0:
    #    u_matrix = torch.dot(torch.dot(rot4D(-sum_tilts), u_matrix), rot_mtx(sum_tilts))
    return u_matrix


def uni4Ddrift(z):
    u_matrix = torch.eye(4)
    u_matrix[0, 1] = z
    u_matrix[2, 3] = z
    return u_matrix


def uni4D_noh(z, k1, sum_tilts=0.):
    """
    :param z: element length [m]
    :param k1: quadrupole strength [1/m**2]
    :param sum_tilts: rotation relative to longitudinal axis [rad]
    """
    kx2 = k1
    ky2 = -k1
    kx2c = kx2 + torch.tensor(0.j)
    ky2c = ky2 + torch.tensor(0.j)
    kx = torch.sqrt(kx2c)
    ky = torch.sqrt(ky2c)

    u_matrix = torch.zeros(4, 4)
    if z > 0:
        cx = torch.cos(z * kx).real
        cy = torch.cos(z * ky).real

        if ky != 0:
            sy = torch.sin(ky * z) / ky
            sy = sy.real
        else:
            sy = z

        if kx != 0:
            sx = (torch.sin(kx * z) / kx).real
        else:
            sx = z

        u_matrix[0, 0] = cx
        u_matrix[0, 1] = sx
        u_matrix[1, 0] = -kx2 * sx
        u_matrix[1, 1] = cx

        u_matrix[2, 2] = cy
        u_matrix[2, 3] = sy
        u_matrix[3, 2] = -ky2 * sy
        u_matrix[3, 3] = cy
    else:
        u_matrix[0, 0] = 1.0
        u_matrix[1, 0] = -k1
        u_matrix[1, 1] = 1.0

        u_matrix[2, 2] = 1.0
        u_matrix[3, 2] = k1
        u_matrix[3, 3] = 1.0

    if sum_tilts != 0:
        u_matrix = torch.dot(torch.dot(rot4D(-sum_tilts), u_matrix), rot_mtx(sum_tilts))
    return u_matrix


def uni4D_noh(z, k1, sum_tilts=0.):
    """
    :param z: element length [m]
    :param k1: quadrupole strength [1/m**2]
    :param sum_tilts: rotation relative to longitudinal axis [rad]
    """
    kx2 = k1
    ky2 = -k1
    kx2c = kx2 + torch.tensor(0.j)
    ky2c = ky2 + torch.tensor(0.j)
    kx = torch.sqrt(kx2c)
    ky = torch.sqrt(ky2c)

    u_matrix = torch.zeros(4, 4)
    if z > 0:
        cx = torch.cos(z * kx).real
        cy = torch.cos(z * ky).real

        if ky != 0:
            sy = torch.sin(ky * z) / ky
            sy = sy.real
        else:
            sy = z

        if kx != 0:
            sx = (torch.sin(kx * z) / kx).real
        else:
            sx = z

        u_matrix[0, 0] = cx
        u_matrix[0, 1] = sx
        u_matrix[1, 0] = -kx2 * sx
        u_matrix[1, 1] = cx

        u_matrix[2, 2] = cy
        u_matrix[2, 3] = sy
        u_matrix[3, 2] = -ky2 * sy
        u_matrix[3, 3] = cy
    else:
        u_matrix[0, 0] = 1.0
        u_matrix[1, 0] = -k1
        u_matrix[1, 1] = 1.0

        u_matrix[2, 2] = 1.0
        u_matrix[3, 2] = k1
        u_matrix[3, 3] = 1.0

    if sum_tilts != 0:
        u_matrix = torch.dot(torch.dot(rot4D(-sum_tilts), u_matrix), rot_mtx(sum_tilts))
    return u_matrix


def create_r4_noh(element):
    k1 = element.k1
    hx = 0.

    # r_z_e = lambda z, energy: uni_matrix(z, k1, hx=hx, sum_tilts=0, energy=energy)

    if element.__class__ == Quadrupole:
        r_z_e = lambda z, energy: uni4D_noh(z, k1, sum_tilts=0)

    elif element.__class__ == Edge:
        sec_e = 1. / torch.cos(element.edge)
        phi = element.fint * element.h * element.gap * sec_e * (1. + torch.sin(element.edge) ** 2)
        # phi = element.fint * element.h * element.gap * sec_e * (1. + torch.sin(2*element.edge) )
        r = torch.eye(6)
        r[1, 0] = element.h * torch.tan(element.edge)
        r[3, 2] = -element.h * torch.tan(element.edge - phi)
        r_z_e = lambda z, energy: r

    elif element.__class__ in [Hcor, Vcor]:
        r_z_e = lambda z, energy: uni4D_noh(z, 0, sum_tilts=0)

    elif element.__class__ == Matrix:
        rm = torch.eye(6)
        rm = element.r

        def r_matrix(z, l, rm):
            if z < l:
                r_z = uni4D_noh(z, 0)
            else:
                r_z = rm
            return r_z

        r_z_e = lambda z, energy: r_matrix(z, element.l, rm)

    elif element.__class__ == Multipole:
        r = torch.eye(6)
        r[1, 0] = -element.kn[1]
        r[3, 2] = element.kn[1]
        r[1, 5] = element.kn[0]
        r_z_e = lambda z, energy: r

    else:
        raise
    return r_z_e


def create_r_matrix(element):
    k1 = element.k1
    if element.l == 0:
        hx = 0.
    else:
        hx = element.angle / element.l

    # r_z_e = lambda z, energy: uni_matrix(z, k1, hx=hx, sum_tilts=0, energy=energy)

    if element.__class__ == Quadrupole:
        r_z_e = lambda z, energy: uni_matrix(z, k1, hx=hx, sum_tilts=0, energy=energy)

    elif element.__class__ == Edge:
        sec_e = 1. / torch.cos(element.edge)
        phi = element.fint * element.h * element.gap * sec_e * (1. + torch.sin(element.edge) ** 2)
        # phi = element.fint * element.h * element.gap * sec_e * (1. + torch.sin(2*element.edge) )
        r = torch.eye(6)
        r[1, 0] = element.h * torch.tan(element.edge)
        r[3, 2] = -element.h * torch.tan(element.edge - phi)
        r_z_e = lambda z, energy: r

    elif element.__class__ in [Hcor, Vcor]:
        r_z_e = lambda z, energy: uni_matrix(z, 0, hx=0, sum_tilts=0, energy=energy)

    elif element.__class__ == Matrix:
        rm = torch.eye(6)
        rm = element.r

        def r_matrix(z, l, rm):
            if z < l:
                r_z = uni_matrix(z, 0, hx=0)
            else:
                r_z = rm
            return r_z

        r_z_e = lambda z, energy: r_matrix(z, element.l, rm)

    elif element.__class__ == Multipole:
        r = torch.eye(6)
        r[1, 0] = -element.kn[1]
        r[3, 2] = element.kn[1]
        r[1, 5] = element.kn[0]
        r_z_e = lambda z, energy: r

    else:
        raise
    return r_z_e
