from ocelot.cpbd.elements import Element
import numpy as np


class NLLens(Element):
    """
    Danilov-Nagaitsev thick nonlinear lens. When l==0, matches MAD-X NLLENS element.
    Supports linear transport by deriving k1 from nonlinear potential.
    Tracking should use NLKickTM that does symplectic drift-kick integration.
    l - length of lens in [m]
    knll - integrated strength of lens [m]. The strength is parametrized so that the
     quadrupole term of the multipole expansion is k1=2*knll/cnll^2.
    cnll - dimensional parameter of lens [m]. The singularities of the potential are located at X=-cnll,+cnll and Y=0.
    tilt - tilt of lens in [rad]
    """
    def __init__(self, l=0., knll=0., cnll=0., tilt=0., eid=None):
        Element.__init__(self, eid)
        self.l = l
        self.knll = knll
        if cnll == 0.:
            raise Exception('Dimensional parameter of NLLens must be non-zero!')
        self.cnll = cnll
        self.tilt = tilt
        self.k1 = 2*knll/(cnll*cnll)
        self.k2 = 0.  # Potential has no sextupolar component
        #self.k3 = 0. # (knll/(cn*sqrt(bn))^2)/cn^2/bn*16 # TODO: implement calculation, there is octupolar field. Could be useful for KickTM.

    def __str__(self):
        s = 'NLLens : '
        s += 'id = ' + str(self.id) + '\n'
        s += 'l    =%8.4f m\n' % self.l
        s += 'knll     =%8.3f m\n' % self.knll
        s += 'cnll     =%8.3f m\n' % self.cnll
        s += 'k1 (calc)=%8.3f 1/m^2\n' % self.k1
        s += 'tilt =%8.2f deg\n' % (self.tilt * 180.0 / np.pi)
        return s
