from typing import Callable, List, Literal, Optional
import logging
import ocelot
from ocelot import MethodTM
from ocelot.cpbd.elements import *
import numpy as np

from .. import lattice as lat
import pyiota.iota

logger = logging.getLogger(__name__)


def load_iota(method=None, file_name: str = None, insert_monitors: bool = True,
              parse_kwargs=None):
    """
    Loads IOTA lattice with standard options
    :param method: ocelot method object
    :param file_name: lattice path
    :return:
    """
    from .. import sixdsim
    method = method or ocelot.MethodTM()
    parse_kwargs = parse_kwargs or {}
    assert file_name is not None
    # file_name = file_name or 'IOTA_1NL_100MeV_v8.6.1.4.6ds'
    lattice_list, correctors, monitors, info, variables = sixdsim.parse_lattice(file_name,
                                                                                verbose=False,
                                                                                **parse_kwargs)
    box = lat.LatticeContainer(name='IOTA', lattice=lattice_list, correctors=correctors,
                               monitors=monitors, info=info, variables=variables, silent=False, method=method)
    box.correctors = [c for c in box.correctors if
                      not c.end_turn and not isinstance(c.ref_el, Drift) and not 'M' in c.id]
    box.monitors = [el for el in box.monitors if any([el.id in bpm for bpm in pyiota.iota.run2.BPMS.ALL])]
    # box.pc = 100.0
    if 'pc' in info:
        box.pc = info['pc']
        if box.pc not in [100.0, 150.0]:
            logger.warning(f'Unusual energy {box.pc} parsed, proceeding anyways')
    else:
        logger.warning(f'No energy found, using 100MeV')
        box.pc = 100.0
    box.remove_elements(box.get_all(CouplerKick))
    # box.transmute_elements(box.get_all(Cavity), Drift)
    box.transmute_elements(box.filter_elements('oq.*', Quadrupole), Drift)
    box.transmute_elements(box.filter_elements('nl.*', Quadrupole), Drift)
    if insert_monitors:
        box.insert_monitors()
    box.update_element_positions()
    box.lattice.update_transfer_maps()
    return box


def insert_qi(nn: int, tn: int, empty_space: float, ref_detuning: float = None, **kwargs):
    """ Generates QI insert with thick elements (octupoles) """
    if ref_detuning == 0.0:
        qi = lat.OctupoleInsert(nn=nn, tn=tn, olen=None, ospacing=empty_space / nn, run=None, oqK=0.0, **kwargs)
        logger.info(f'QI thick ({nn}/{tn}) - TURNED OFF')
    else:
        qi = lat.OctupoleInsert(nn=nn, tn=tn, olen=None, ospacing=empty_space / nn, run=None, **kwargs)
        dq = qi.compute_relative_detuning()
        if ref_detuning is not None:
            oqK = ref_detuning / dq if dq != 0.0 else 0.0
            logger.info(f'QI thick ({nn}/{tn}) - scaled by {ref_detuning:.3f}/{dq:.3f} = {oqK:.3f}')
            qi.configure(nn=nn, tn=tn, olen=None, ospacing=empty_space / nn, run=None, oqK=oqK, **kwargs)
        else:
            logger.info(f'QI thick ({nn}/{tn}) - no scaling, dq = {dq:.3f}')
    return qi, [el for el in qi.seq if isinstance(el, Octupole)]


def insert_qi_phase(nn: int, tn: int, empty_space: float, ref_detuning: float = None, **kwargs):
    """ Generates QI insert with thick elements (octupoles) spaced in equal phase"""
    olen = (1.8 - empty_space) / nn
    if ref_detuning == 0.0:
        qi = lat.OctupoleInsert(nn=nn, tn=tn, olen=olen, spacing_mode='phase', run=None, oqK=0.0, **kwargs)
        logger.info(f'QI thick phase ({nn}/{tn}) - TURNED OFF')
    else:
        qi = lat.OctupoleInsert(nn=nn, tn=tn, olen=olen, spacing_mode='phase', run=None, **kwargs)
        dq = qi.compute_relative_detuning()
        if ref_detuning is not None:
            oqK = ref_detuning / dq if dq != 0.0 else 0.0
            logger.info(f'QI thick phase ({nn}/{tn}) - scaled by {ref_detuning:.3f}/{dq:.3f} = {oqK:.3f}')
            qi.configure(nn=nn, tn=tn, olen=olen, spacing_mode='phase', run=None, oqK=oqK, **kwargs)
        else:
            logger.info(f'QI thick ({nn}/{tn}) - no scaling, dq = {dq:.3f}')
    return qi, [el for el in qi.seq if isinstance(el, Octupole)]


def insert_dn(nn: int, tn: int, empty_space: float, ref_detuning: float = None, **kwargs):
    """ Generates DN insert with thick elements (octupoles) """
    if ref_detuning == 0.0:
        qi = lat.NLInsert(nn=nn, tn=tn, olen=None, ospacing=empty_space / nn, run=None, oqK=0.0, **kwargs)
        print(f'DN thick ({nn}/{tn}) - TURNED OFF')
    else:
        qi = lat.NLInsert(nn=nn, tn=tn, olen=None, ospacing=empty_space / nn, run=None, **kwargs)
        # qi = lat.OctupoleInsert(nn=nn, tn=tn, olen=None, ospacing=empty_space/nn, run=None, **kwargs)
        # dq = qi.compute_relative_detuning()
        # if ref_detuning is not None:
        #    oqK = ref_detuning / dq if dq != 0.0 else 0.0
        #    print(f'QI thick ({nn}/{tn}) - scaled by {ref_detuning:.3f}/{dq:.3f} = {oqK:.3f}')
        #    qi.configure(nn=nn, tn=tn, olen=None, ospacing=empty_space/nn, run=None, oqK=oqK, **kwargs)
        # else:
        print(f'DN thick ({nn}/{tn}) - no scaling!')
    return qi, [el for el in qi.seq if isinstance(el, lat.NLLens)]


def insert_qi_thin(nn: int, tn: int, empty_space: float, ref_detuning):
    """ Generates QI insert with thin elements (multipoles) """
    qi = lat.OctupoleInsert(nn=nn, olen=0.0, tn=tn, otype=0)
    dq = qi.compute_relative_detuning()
    oqK = ref_detuning / dq if dq != 0.0 else 0.0
    print(f'{nn} - detuning correction factor: {ref_detuning}/{dq} = {oqK}')
    qi.configure(nn=nn, olen=0.0, tn=tn, otype=0, oqK=oqK)
    return qi, [el for el in qi.seq if isinstance(el, Multipole)]


def insert_flat(nn: int, tn: int, empty_space: float, ref_detuning):
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


def insert_noop(*args, **kwargs):
    return None, None


def build_lattice(lattice_style: Literal['iota', 'iota_nio_ior', 'iota_ior_nio_ior', 'dn',
'iota_ior', 'iota_iol', 'ideal', 'ideal_bare'],
                  insert_style: Optional[Literal['flat', 'qi', 'qi_thin', 'dn', 'qi_phase']],
                  nn: int = None, tn: int = None,
                  empty_space: float = None,
                  olen: float = None,
                  ref_detuning: float = None,
                  method: MethodTM = None,
                  drift_cnt: int = None,
                  drop_empty_octupole_drifts: bool = False,
                  replace_disabled_elements: bool = False,
                  rotate_to_nio_center: bool = True,
                  insert_monitors: bool = True,
                  lattice_load_kwargs: dict = None,
                  **kwargs):
    method = method or ocelot.MethodTM()
    ll_kwargs = lattice_load_kwargs or None
    if insert_style is None:
        qibox = ocps = None
    else:
        if nn is None or tn is None:
            raise Exception
        if empty_space is None and olen is None:
            raise Exception
        else:
            if empty_space is not None and olen is not None:
                raise Exception
            else:
                if empty_space is not None:
                    pass
                elif olen is not None:
                    empty_space = (1.8-olen*nn)
                else:
                    raise Exception
        insert_styles = {'flat': insert_flat, 'qi': insert_qi,
                         'qi_thin': insert_qi_thin, 'dn': insert_dn,
                         'qi_phase': insert_qi_phase}
        assert insert_style in insert_styles

        kwargs = {'drop_empty_drifts': drop_empty_octupole_drifts,
                  'replace_zero_strength_octupoles': replace_disabled_elements, **kwargs}
        qi, ocps = insert_styles[insert_style](nn, tn, empty_space, ref_detuning=ref_detuning, **kwargs)
        f0, betae, alfae, betas = qi.calculate_optics_parameters()
        mat = Matrix(l=0.0, eid='TInsert', r11=1, r22=1, r33=1, r44=1, r55=1, r66=1, r21=-1 / f0, r43=-1 / f0)
        qibox = lat.LatticeContainer('ideal', qi.to_sequence() + [mat], reset_elements_to_defaults=False, method=method)
        qibox.qi = qi
    logger.info('QI built, making lattice')

    if lattice_style == 'iota':
        # full lattice
        box = load_iota(method, insert_monitors=insert_monitors, **ll_kwargs)
        # insert qi
        if insert_style is not None:
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
        # assert np.isclose(l_orig, box.totallen)
        box.update()
    elif lattice_style == 'iota_nio_ior':
        box = load_iota(method, insert_monitors=insert_monitors)
        l_orig = box.totallen
        if insert_style is not None:
            e1, e2 = box['NLMRUP_2'], box['NLMRDOWN_1']
            i1, i2 = box.sequence.index(e1), box.sequence.index(e2)
            qiseq = qi.to_sequence()
            assert nn % 2 == 0
            assert len(qiseq) % 2 == 0
            m = Marker(eid='IOR')
            qiseq = qiseq[:len(qiseq) // 2] + [m] + qiseq[len(qiseq) // 2:]
            seq = box.sequence[:i1 + 1] + qiseq + box.sequence[i2:]
            box.sequence = seq
            box.update_element_positions()
            assert np.isclose(l_orig, box.totallen)
        box.sequence = [ocelot.Marker(eid='START')] + box.sequence + [ocelot.Marker(eid='END')]
        box.update()
        assert np.isclose(l_orig, box.totallen)
    elif lattice_style == 'iota_ior_nio_ior':
        box = load_iota(method, insert_monitors=insert_monitors, **ll_kwargs)
        l_orig = box.totallen
        if insert_style is not None:
            e1, e2 = box['NLMRUP_2'], box['NLMRDOWN_1']
            i1, i2 = box.sequence.index(e1), box.sequence.index(e2)
            qiseq = qi.to_sequence()
            assert nn % 2 == 0
            assert len(qiseq) % 2 == 0
            m = Marker(eid='IOR')
            qiseq = qiseq[:len(qiseq) // 2] + [m] + qiseq[len(qiseq) // 2:]
            seq = box.sequence[:i1 + 1] + qiseq + box.sequence[i2:]
            box.sequence = seq
            box.update_element_positions()
            box.rotate_lattice(m)
            box.update_element_positions()
            assert np.isclose(l_orig, box.totallen)
        box.sequence = [ocelot.Marker(eid='START')] + box.sequence + [ocelot.Marker(eid='END')]
        box.update()
        assert np.isclose(l_orig, box.totallen)
    elif lattice_style == 'iota_ior':
        # full lattice at ior
        box = load_iota(method, insert_monitors=insert_monitors, **ll_kwargs)
        l_orig = box.totallen
        if insert_style is not None:
            e1, e2 = box['NLMLDOWN_2'], box['NLMLUP_1']
            i1, i2 = box.sequence.index(e1), box.sequence.index(e2)
            seq = box.sequence[:i1 + 1] + qi.to_sequence() + box.sequence[i2:]
            box.sequence = seq
            box.update_element_positions()
            assert np.isclose(l_orig, box.totallen)

        els_center = box['NLLC']
        assert isinstance(els_center, Drift)
        splits = box.split_elements(els_center, n_parts=2, return_new_elements=True)
        box.rotate_lattice(splits[0][1])
        box.sequence = [ocelot.Marker(eid='START')] + box.sequence + [ocelot.Marker(eid='END')]
        box.update()
        assert np.isclose(l_orig, box.totallen)
    elif lattice_style == 'iota_iol':
        # full lattice at iol
        box = load_iota(method, insert_monitors=insert_monitors, **ll_kwargs)
        l_orig = box.totallen
        if insert_style is not None:
            e1, e2 = box['NLMLDOWN_2'], box['NLMLUP_1']
            i1, i2 = box.sequence.index(e1), box.sequence.index(e2)
            qiseq = qi.to_sequence()
            els_center = qibox.at(0.9)
            assert len(els_center) == 1
            if isinstance(els_center[0], Octupole):
                assert nn % 2 == 1
                splits = qibox.split_elements(els_center[0], n_parts=2, return_new_elements=True)
                last_el = splits[0][1]
            elif isinstance(els_center[0], Drift):
                assert nn % 2 == 0
                assert len(qiseq) % 2 == 0
                last_el = els_center[0]
            m = Marker(eid='IOL')
            qibox.insert_elements(m, before=last_el)
            qiseq = qibox.sequence[:-1]
            #qiseq = qiseq[:len(qiseq) // 2] + [m] + qiseq[len(qiseq) // 2:]
            seq = box.sequence[:i1 + 1] + qiseq + box.sequence[i2:]
            box.sequence = seq
            box.update_element_positions()
            box.rotate_lattice(m)
            box.update_element_positions()
            assert np.isclose(l_orig, box.totallen)
        else:
            els_center = box.get_one('OQ09', exact=True)
            assert isinstance(els_center, Drift)
            splits = box.split_elements(els_center, n_parts=2, return_new_elements=True)
            box.rotate_lattice(splits[0][1])
        box.sequence = [ocelot.Marker(eid='START')] + box.sequence + [ocelot.Marker(eid='END')]
        box.update()
        assert np.isclose(l_orig, box.totallen)
    elif lattice_style == 'ideal':
        # just insert
        seq = []
        for i in range(drift_cnt // 2):
            seq.extend([Drift(l=1.8)] + [Monitor(l=0.0, eid=f'BPM{i}')] + [mat])
        seq += qi.to_sequence() + [mat]
        for i in range(drift_cnt // 2 - 1):
            seq.extend([Drift(l=1.8)] + [Monitor(l=0.0, eid=f'BPM{i + drift_cnt // 2}')] + [mat])
        seq = seq[:-2] + [seq[-1]] + [Monitor(l=0.0, eid='BPMBB1R'), Drift(l=1.8), Monitor(l=0.0, eid='BPMBB2R'), mat]

        box = lat.LatticeContainer('ideal', seq, reset_elements_to_defaults=False, method=method)
        box.pc = 150.0
        box.update()

        els_center = box.at(0.9)
        if isinstance(els_center[0], Octupole):
            splits = box.split_elements(els_center[0], n_parts=2, return_new_elements=True)
            box.rotate_lattice(splits[0][1])
        elif isinstance(els_center[0], Drift):
            assert len(els_center) == 1
            splits = box.split_elements(els_center[0], n_parts=2, return_new_elements=True)
            box.rotate_lattice(splits[0][1])
            # box.rotate_lattice(els_center[0])

        box.update()
        # assert np.isclose(box.get_first('TInsert').s_mid, 0.9)
        assert np.isclose(box.totallen, 1.8 * drift_cnt + 1.8)

        box.lattice.sequence = [ocelot.Marker(eid='S')] + box.lattice.sequence
        box.update()
    elif lattice_style == 'ideal_bare':
        #if nn % 2 != 0:
        #    raise AttributeError('Even lattices only')
        seq: List[Element] = qi.to_sequence() + [mat]
        box = lat.LatticeContainer('ideal_bare', seq, reset_elements_to_defaults=False, method=method)
        box.pc = 150.0
        box.update()

        els_center = box.at(0.9)
        assert len(els_center) == 1
        if isinstance(els_center[0], Octupole):
            splits = box.split_elements(els_center[0], n_parts=2, return_new_elements=True)
        elif isinstance(els_center[0], Drift):
            assert isinstance(els_center[0], Drift), f'{els_center=}'
            splits = box.split_elements(els_center[0], n_parts=2, return_new_elements=True)
        else:
            raise Exception
        box.update()
        assert np.isclose(box.totallen, 1.8)

        els_center = box.at(0.9)
        assert len(els_center) == 1
        logger.info(f'{els_center=}')
        assert np.isclose(els_center[0].s_start, 0.9, atol=1e-10, rtol=0), f'{els_center[0].s_start=}'
        m = ocelot.Marker(eid='IOL')
        box.insert_elements(m, before=els_center[-1])
        if rotate_to_nio_center:
            box.rotate_lattice(m)
        box.sequence = [ocelot.Marker(eid='START')] + box.sequence + [ocelot.Marker(eid='END')]
        box.update()
        assert np.isclose(box.totallen, 1.8)
    else:
        raise AttributeError(f"Unknown lattice style {lattice_style} requested!")
    return qibox, box, ocps
