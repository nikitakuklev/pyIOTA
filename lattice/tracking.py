"""
Convenience functions for tracking
"""
import time
from typing import List

import ocelot
from ocelot import *
from ocelot import Track_info, merge_maps, SecondOrderMult
from pyIOTA.lattice import LatticeContainer
import copy


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


def track_nturns_fast(lat: MagneticLattice, nturns: int, track_list: List[Track_info], nsuperperiods: int = 1,
                      save_track: bool = True, print_progress: bool = False):
    """
    Modified OCELOT track_nturns method that tries to speed up tracking
    1 - It uses numpy, since numexpr is bad for few particles
    2 - It merges matrix elements, while leaving everything else
    """
    # Do not merge anything that is not a matrix
    exclusions = [el.__class__ for el in lat.sequence if not isinstance(el.transfer_map, (TransferMap, SecondTM))]
    new_lat = merger(lat, remaining_types=exclusions)
    assert lat.totalLen == new_lat.totalLen
    print(f'Merged lattice from ({len(lat.sequence)}) to ({len(new_lat.sequence)}) elements')

    navi = Navigator(new_lat)
    t_maps = get_map(new_lat, new_lat.totalLen, navi)
    mult = SecondOrderMult()
    for tm in t_maps:
        if isinstance(tm, SecondTM):
            tm.multiplication = mult.numpy_apply
    track_list_const = copy.copy(track_list)
    p_array = ParticleArray()
    p_list = [p.particle for p in track_list]
    p_array.list2array(p_list)

    start = time.time()
    for i in range(nturns):
        if print_progress: print(i)
        for n in range(nsuperperiods):
            for tm in t_maps:
                tm.apply(p_array)
        for n, pxy in enumerate(track_list):
            pxy.turn = i
            if save_track:
                # IMPORTANT TO COPY AND NOT STORE A VIEW (even though I'd expect column slices to copy in 'C' order)
                pxy.p_list.append(p_array.rparticles[:, n].copy())
    print(f'Perf: ({time.time()-start:.4e})s - ({(time.time()-start)/nturns:.4e})s per turn')
    return track_list_const

