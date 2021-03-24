from typing import Callable

import ocelot
from ocelot.cpbd.elements import *

import pyIOTA.lattice as lat
import pyIOTA.iota
import pyIOTA.sixdsim as sixdsim


def load_iota(method, file_name: str = None):
    """
    Loads IOTA lattice with standard options
    :param method: ocelot method object
    :param file_name: lattice path
    :return:
    """
    file_name = file_name or 'IOTA_1NL_100MeV_v8.6.1.4.6ds'
    lattice_list, correctors, monitors, info, variables = sixdsim.parse_lattice(file_name, verbose=False)
    box = lat.LatticeContainer(name='IOTA', lattice=lattice_list, correctors=correctors,
                               monitors=monitors, info=info, variables=variables, silent=False, method=method)
    box.correctors = [c for c in box.correctors if
                      not c.end_turn and not isinstance(c.ref_el, Drift) and not 'M' in c.id]
    box.monitors = [el for el in box.monitors if any([el.id in bpm for bpm in pyIOTA.iota.run2.BPMS.ALL])]
    box.pc = 100.0
    box.remove_elements(box.get_all(CouplerKick))
    box.transmute_elements(box.get_all(Cavity), Drift)
    box.transmute_elements(box.filter_elements('oq.*', Quadrupole), Drift)
    box.transmute_elements(box.filter_elements('nl.*', Quadrupole), Drift)
    box.insert_monitors()
    box.update_element_positions()
    box.lattice.update_transfer_maps()
    return box


def insert_qi(nn: int, tn: int, empty_space, ref_detuning):
    """ Generates QI insert with thick elements (octupoles) """
    olen = (1.8 - empty_space) / nn
    qi = lat.OctupoleInsert(nn=nn, olen=olen, tn=tn)
    dq = qi.compute_relative_detuning()
    oqK = ref_detuning / dq if dq != 0.0 else 0.0
    print(f'{nn} - detuning correction factor: {ref_detuning}/{dq} = {oqK}')
    qi.configure(nn=nn, olen=olen, tn=tn, oqK=oqK)
    return qi, [el for el in qi.seq if isinstance(el, Octupole)]


def insert_qi_thin(nn: int, tn: int, empty_space, ref_detuning):
    """ Generates QI insert with thin elements (multipoles) """
    qi = lat.OctupoleInsert(nn=nn, olen=0.0, tn=tn, otype=0)
    dq = qi.compute_relative_detuning()
    oqK = ref_detuning / dq if dq != 0.0 else 0.0
    print(f'{nn} - detuning correction factor: {ref_detuning}/{dq} = {oqK}')
    qi.configure(nn=nn, olen=0.0, tn=tn, otype=0, oqK=oqK)
    return qi, [el for el in qi.seq if isinstance(el, Multipole)]


def insert_flat(nn: int, tn: int, empty_space, ref_detuning):
    """ Generates QI insert with flat strength profile and thick elements """
    olen = (1.8 - empty_space) / nn
    qi = lat.OctupoleInsert(nn=nn, olen=olen, tn=tn)
    ocps = [el for el in qi.seq if isinstance(el, Octupole)]
    for ocp in ocps:
        ocp.k3 = ocps[0].k3
    dq = qi.compute_relative_detuning()
    oqK = ref_detuning / dq if dq != 0.0 else 0.0
    print(f'{nn} - detuning correction factor: {ref_detuning}/{dq} = {oqK}')
    qi.configure(nn=nn, olen=olen, tn=tn, oqK=oqK)
    ocps = [el for el in qi.seq if isinstance(el, Octupole)]
    for ocp in ocps:
        ocp.k3 = ocps[0].k3
    return qi, [el for el in qi.seq if isinstance(el, Octupole)]


def build_lattice(lattice_style: str, insert_style: str, nn: int, tn: int, empty_space: float,
                  ref_detuning: float, method, drift_cnt: int = None):
    insert_styles = {'flat': insert_flat, 'qi': insert_qi, 'qi_thin': insert_qi_thin}
    assert insert_style in insert_styles

    qi, ocps = insert_styles[insert_style](nn, tn, empty_space, ref_detuning)
    f0, betae, alfae, betas = qi.calculate_optics_parameters()
    mat = Matrix(l=0.0, eid='TInsert', r11=1, r22=1, r33=1, r44=1, r55=1, r66=1, r21=-1 / f0, r43=-1 / f0)
    qibox = lat.LatticeContainer('ideal', qi.to_sequence() + [mat], reset_elements_to_defaults=False, method=method)

    if lattice_style == 'iota':
        # full lattice
        box = load_iota(method)
        e1, e2 = box['NLMLDOWN_2'], box['NLMLUP_1']
        i1, i2 = box.sequence.index(e1), box.sequence.index(e2)
        l_orig = box.totallen
        seq = box.sequence[:i1 + 1] + qi.to_sequence() + box.sequence[i2:]
        box.sequence = seq
        box.update_element_positions()
        assert np.isclose(l_orig, box.totallen)

        # els_center = box.at(box['NLMLDOWN_2'].s_end + 0.9)
        # if isinstance(els_center[0], Octupole):
        #     splits = box.split_elements(els_center[0], n_parts=2, return_new_elements=True)
        #     box.rotate_lattice(splits[0][1])
        # elif isinstance(els_center[0], Drift):
        #     assert len(els_center) == 1
        #     box.rotate_lattice(els_center[0])

        box.update_element_positions()
        box.lattice.update_transfer_maps()
        assert np.isclose(l_orig, box.totallen)
    elif lattice_style == 'ideal':
        # just insert
        seq = []
        [seq.extend([Drift(l=1.8)] + [Monitor(l=0.0, eid=f'BPM{i}')] + [mat]) for i in range(drift_cnt // 2)]
        seq += qi.to_sequence() + [mat]
        [seq.extend([Drift(l=1.8)] + [Monitor(l=0.0, eid=f'BPM{i + drift_cnt // 2}')] + [mat]) for i in
         range(drift_cnt // 2 - 1)]
        seq = seq[:-2] + [seq[-1]] + [Monitor(l=0.0, eid='BPMBB1R'), Drift(l=1.8), Monitor(l=0.0, eid='BPMBB2R'), mat]

        box = lat.LatticeContainer('ideal', seq, reset_elements_to_defaults=False, method=method)
        box.pc = 100.0
        box.lattice.update_transfer_maps()
        box.update_element_positions()

        els_center = box.at(0.9)
        if isinstance(els_center[0], Octupole):
            splits = box.split_elements(els_center[0], n_parts=2, return_new_elements=True)
            box.rotate_lattice(splits[0][1])
        elif isinstance(els_center[0], Drift):
            assert len(els_center) == 1
            splits = box.split_elements(els_center[0], n_parts=2, return_new_elements=True)
            box.rotate_lattice(splits[0][1])
            # box.rotate_lattice(els_center[0])

        box.update_element_positions()
        box.lattice.update_transfer_maps()
        # assert np.isclose(box.get_first('TInsert').s_mid, 0.9)
        assert np.isclose(box.totallen, 1.8 * drift_cnt + 1.8)

        box.lattice.sequence = [ocelot.Marker(eid='S')] + box.lattice.sequence
        box.update_element_positions()
        box.lattice.update_transfer_maps()
    else:
        raise AttributeError("Unknown lattice style requested!")
    return qibox, box, ocps
