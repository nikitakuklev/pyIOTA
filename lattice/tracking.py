"""
Convenience functions for tracking
"""

from ocelot import *
from pyIOTA.lattice import LatticeContainer


def track_nturns_with_bpms(box: LatticeContainer, turns: int, p: Particle):
    """
    Tracks box in ring mode, collecting data at all BPMS. Does not combine matrices
    since it is expected that some elements will not be matrix-based (i.e. NL insert)
    :param box:
    :param p:
    :param turns:
    :return:
    """
    bpms = box.get_elements(Monitor)
    datax = {bpm.id + 'H': [] for bpm in bpms}
    datay = {bpm.id + 'V': [] for bpm in bpms}
    datap = {bpm.id: [] for bpm in bpms}
    datax['OH'] = []
    datay['OV'] = []
    datap['O'] = []
    for t in np.arange(turns):
        navi = Navigator(box.lattice)
        L = 0.
        for bpm in bpms:
            dz = bpm.s_mid - L
            tracking_step(box.lattice, [p], dz, navi)
            datax[bpm.id + 'H'].append(p.x)
            datay[bpm.id + 'V'].append(p.y)
            datap[bpm.id].append(p.p)
            # bpm.y = p.y
            # bpm.E = p.E
            L = bpm.s_mid
            # particles.append(copy.copy(p))
        # Finish turn (extra length to ensure so)
        tracking_step(box.lattice, [p], box.lattice.totalLen - L + 100, navi)
        datax['OH'].append(p.x)
        datay['OV'].append(p.y)
        datap['O'].append(p.p)
        if not np.isclose(navi.z0, box.lattice.totalLen, atol=1e-10):
            print(f'End of turn navi length ({navi.z0}) not equal to lattice length ({box.lattice.totalLen})')
    return datax, datay, datap
