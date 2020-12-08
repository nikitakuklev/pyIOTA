"""
Convenience functions for tracking
"""
import time
from typing import List

# import ocelot
from ocelot import *
import numpy as np
from ocelot import Track_info, merge_maps, SecondOrderMult
from pyIOTA.lattice import LatticeContainer
import copy


def track_nturns_with_bpms(box: LatticeContainer, n_turns: int, p: Particle, store_at_start: bool = True):
    """
    Tracks box in ring mode, collecting data at all BPMS. Does not combine matrices
    since it is expected that some elements will not be matrix-based (i.e. NL insert)
    :param box: box
    :param p: particle object
    :param n_turns: number of turns
    :param store_at_start: also store coordinates at s=0 (under 'O' name)
    :return: dictionaries for 6 planes
    """
    bpms = box.get_elements(Monitor)
    datax = {bpm.id + 'H': [] for bpm in bpms}
    datapx = {bpm.id: [] for bpm in bpms}
    datay = {bpm.id + 'V': [] for bpm in bpms}
    datapy = {bpm.id: [] for bpm in bpms}
    datap = {bpm.id: [] for bpm in bpms}
    if store_at_start:
        datax['OH'] = []
        datay['OV'] = []
        datap['O'] = []
    for t in np.arange(n_turns):
        navi = Navigator(box.lattice)
        L = 0.
        for bpm in bpms:
            dz = bpm.s_mid - L
            tracking_step_element(box.lattice, [p], Monitor, navi)
            datax[bpm.id + 'H'].append(p.x)
            datay[bpm.id + 'V'].append(p.y)
            datap[bpm.id].append(p.p)
            datapx[bpm.id].append(p.px)
            datapy[bpm.id].append(p.py)
            # bpm.y = p.y
            # bpm.E = p.E
            L = bpm.s_mid
            # particles.append(copy.copy(p))
        # Finish turn
        #tracking_step(box.lattice, [p], box.lattice.totalLen - L, navi)
        tracking_step_element(box.lattice, [p], Monitor, navi)
        if store_at_start:
            datax['OH'].append(p.x)
            datay['OV'].append(p.y)
            datap['O'].append(p.p)
        if not np.isclose(navi.z0, box.lattice.totalLen, atol=1e-10):
            print(f'End of turn navi length ({navi.z0}) not equal to lattice length ({box.lattice.totalLen})')
    return datax, datapx, datay, datapy, datap


def tracking_step(lat, particle_list, dz, navi):
    """
    tracking for a fixed step dz
    :param lat: Magnetic Lattice
    :param particle_list: ParticleArray or Particle list
    :param dz: step in [m]
    :param navi: Navigator
    """
    if navi.z0 + dz > lat.totalLen:
        dz = lat.totalLen - navi.z0
    t_maps = get_map(lat, dz, navi)
    for tm in t_maps:
        tm.apply(particle_list)


def tracking_step_element(lat, particle_list, el_type, navi):
    """ Tracking step until next element of given class or end of lattice """
    if navi.n_elem == len(lat.sequence)-1:
        raise Exception("Tracking step called while already at end of lattice?")
    t_maps = get_map_element(lat, el_type, navi)
    for tm in t_maps:
        tm.apply(particle_list)


def get_map_element(lattice, el_type, navi):
    """ Gets maps until next element of given class or end of lattice """
    nelems = len(lattice.sequence)
    TM = []
    i = navi.n_elem
    dl = 0.0
    while True:
        elem = lattice.sequence[i]
        TM.append(elem.transfer_map(elem.l))
        dl += elem.l
        i += 1
        if isinstance(elem, el_type) or i >= nelems:
            break

    navi.z0 += dl
    navi.sum_lengths += dl
    navi.n_elem = i
    return TM


def get_map(lattice, dz, navi):
    nelems = len(lattice.sequence)
    TM = []
    i = navi.n_elem
    z1 = navi.z0 + dz
    elem = lattice.sequence[i]
    # navi.sum_lengths = np.sum([elem.l for elem in lattice.sequence[:i]])
    L = navi.sum_lengths + elem.l
    while z1 + 1e-10 > L:

        dl = L - navi.z0
        TM.append(elem.transfer_map(dl))

        navi.z0 = L
        dz -= dl
        if i >= nelems - 1:
            break

        i += 1
        elem = lattice.sequence[i]
        L += elem.l
        #if i in navi.proc_kick_elems:
        #    break
    if abs(dz) > 1e-10:
        TM.append(elem.transfer_map(dz))
    navi.z0 += dz
    navi.sum_lengths = L - elem.l
    navi.n_elem = i
    return TM


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
    print(f'Perf: ({time.time() - start:.4e})s - ({(time.time() - start) / nturns:.4e})s per turn')
    return track_list_const
