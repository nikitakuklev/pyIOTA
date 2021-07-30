"""
Convenience functions for tracking
You will notice a huge number of almost-duplicates - this is for performance reasons to cater to different studies
"""
import time
import logging
import re
import copy
from pathlib import PurePath
from typing import List, Type, Dict

import numpy as np
import pandas as pd

from ..lattice import LatticeContainer
from ocelot import Particle, Monitor, ParticleArray, Navigator, MagneticLattice, \
    merger, Element, SecondTM, Marker, TransferMap, Track_info, merge_maps, SecondOrderMult

logger = logging.getLogger(__name__)


def track_nturns_with_bpms(box: LatticeContainer, p: Particle, n_turns: int, store_at_start: bool = True):
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
        # tracking_step(box.lattice, [p], box.lattice.totalLen - L, navi)
        tracking_step_element(box.lattice, [p], Monitor, navi)
        if store_at_start:
            datax['OH'].append(p.x)
            datay['OV'].append(p.y)
            datap['O'].append(p.p)
        if not np.isclose(navi.z0, box.lattice.totalLen, atol=1e-10):
            print(f'End of turn navi length ({navi.z0}) not equal to lattice length ({box.lattice.totalLen})')
    return datax, datapx, datay, datapy, datap


def _elegant_check_length(data, box):
    # Make sure elegant length matched - sanity check
    try:
        matches = [re.match("^length of beamline IOTA per pass: (-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?) m$", s)
                   for s in data.stdout.splitlines()]
        hits = [m for m in matches if m is not None]
        assert len(hits) == 1
        len_elegant = float(hits[0].group(1))
        assert np.isclose(len_elegant, box.totallen)
    except:
        print('Elegant job failure!')
        print(data.stdout)


def _elegant_check_state(box):
    for el in box.sequence:
        if getattr(el, 'elegant_temporary', False):
            raise AttributeError(f'Element ({el.id}) has temporary elegant flag - state inconsistent')


def track_nturns(box: LatticeContainer, p: Particle, n_turns: int):
    """
    Tracks box in ring mode without storing data. Does not combine matrices.
    :param box: box
    :param n_turns: number of turns
    :param p: particle object
    :return: tracked particle
    """
    i = 0
    # navi = Navigator(box.lattice)
    # t_maps = get_map(box.lattice, box.lattice.totalLen, navi)
    # if not np.isclose(navi.z0, box.lattice.totalLen, atol=1e-10):
    #     raise ValueError(f'Navi length ({navi.z0}) not equal to lattice ({box.lattice.totalLen})')
    # # TODO: add profiling
    # for turn in np.arange(n_turns):
    #     for tm in t_maps:
    #         tm.apply(p)
    #         i += 1

    # Not checked, but particle array seems faster even for single particle
    t_maps = [el.transfer_map(el.l) for el in box.lattice.sequence]
    pa = ParticleArray()
    pa.list2array([p])
    for turn in np.arange(n_turns):
        for tm in t_maps:
            tm.apply(pa)
            i += 1
    pa.array2ex_list([p])

    assert i == n_turns * len(box.lattice.sequence)
    return p


def track_nturns_store_particles(box: LatticeContainer, p: Particle, n_turns: int):
    """
    Same as track_nturns but stores particles after each element
    :param box: box
    :param n_turns: number of turns
    :param p: initial particle
    :return: List of particles after each element, including copy of initial one
    """
    p_list = [copy.copy(p)]
    navi = Navigator(box.lattice)
    t_maps = get_map(box.lattice, box.lattice.totalLen, navi)
    if not np.isclose(navi.z0, box.lattice.totalLen, atol=1e-10):
        raise ValueError(f'Navi length ({navi.z0}) not equal to lattice ({box.lattice.totalLen})')
    for turn in np.arange(n_turns):
        for tm in t_maps:
            tm.apply(p)
            p_list.append(copy.copy(p))
    assert len(p_list) == n_turns * len(box.lattice.sequence) + 1
    return p_list


def frequency_map_analysis(box: LatticeContainer, n_turns: int = 1, backend='ocelot', dry_run: bool = False,
                           xmin=1e-5, xmax=0.1, ymin=1e-5, ymax=0.1, nx=21, ny=21,
                           mpi: int = 0,
                           lattice_options_extra: dict = None,
                           task_options: dict = None,
                           timeout: int = 60):
    """
    Perform FMA simulation.

    Elegant - uses frequency_map, so custom distributions not supported. Use track methods + NAFF if needed.
    """
    if backend == 'ocelot':
        raise Exception

    elif backend == 'elegant' or 'elegant_async':
        from ..sim import DaskClient, STATE
        from ..elegant import Task, routines
        from ..util import config as cfg

        for el in box.sequence:
            if getattr(el, 'elegant_temporary', False):
                raise AttributeError(f'Element ({el.id}) has temporary elegant flag - state inconsistent')

        # Create elegant task
        lattice_options = {'sr': 0, 'isr': 0, 'dip_kicks': 64, 'quad_kicks': 32, 'sext_kicks': 16, 'oct_kicks': 16}
        if lattice_options_extra:
            lattice_options.update(lattice_options_extra)
        task_options = task_options or {}
        params = {'label': 'auto_freqmap'}

        sj = routines.standard_sim_job(work_folder=cfg.DASK_DEFAULT_WORK_FOLDER,
                                       lattice_options=lattice_options,
                                       task_options=task_options,
                                       add_random_id=True,
                                       parameter_file_map=None,
                                       **params)

        sj.lattice_file_contents = box.to_elegant(lattice_options=lattice_options, silent=True)

        for el in box.sequence:
            if getattr(el, 'elegant_temporary', False):
                raise AttributeError(f'Element ({el.id}) has temporary elegant flag - state inconsistent')

        # Elegant taskfile
        t = Task(relative_mode=True, run_folder=sj.run_subfolder, lattice_path=sj.lattice_file_abs_path)
        routines.template_task_frequency_map(box, t, n_turns=n_turns, nx=nx, ny=ny,
                                             xmax=xmax, ymax=ymax, xmin=xmin, ymin=ymin, **task_options)
        sj.task_file_contents = t.compile()

        if backend == 'elegant_async':
            sj.params['n_turns'] = n_turns
            return sj

        dc = DaskClient()
        futures = dc.submit_to_elegant([sj], dry_run=dry_run, mpi=mpi, pure=False)
        future = futures[0]
        try:
            (data, etaskresp) = future.result(timeout)
        except Exception as e:
            import traceback
            print(traceback.format_tb(future.traceback()))
            raise e
        if etaskresp.state != STATE.ENDED:
            logger.error(f'Sim failed: {etaskresp}')
            raise Exception

        if data.returncode != 0:
            logger.error(data)
            raise ValueError(f'Job returned with error code {data.returncode}!')

        futures = dc.read_out([etaskresp], dry_run=dry_run)
        future = futures[0]
        try:
            (data2, etaskresp2) = future.result(60)
        except Exception as e:
            import traceback
            print(traceback.format_tb(future.traceback()))
            print(data.stdout)
            print(data.stderr)
            print(etaskresp)
            raise e
        if etaskresp2.state != STATE.ENDED_READ:
            logger.error(f'Read failed: {etaskresp2}')
            raise Exception

        if dry_run:
            df_fma = pd.DataFrame(data=[{'x': 0.0, 'y': 0.0, 'nux': 0.0, 'nuy': 0.0, 'diffusion': 0.0}])
            return df_fma, (data, data2, etaskresp, etaskresp2, sj)

        _elegant_check_length(data, box)

        df_fma = data2['fma'].df()
        result_dict = {'sim_data': data, 'read_data': data2, 'sim_task': etaskresp, 'read_task': etaskresp2, 'job': sj}
        return df_fma, result_dict  # (data, data2, etaskresp, etaskresp2, sj)
    else:
        raise AttributeError(f'Unknown backend ({backend})')


def process_frequency_map_analysis(box, job_list, dry_run: bool, mpi, timeout=60):
    logger.info(f'Processing {len(job_list)} jobs')
    jobs_todo = [j for j in job_list if not hasattr(j, 'sim_results')]
    logger.info(f'Submitting {len(jobs_todo)} jobs')
    j_out = process_elegant_jobs(jobs_todo, dry_run=dry_run, mpi=mpi, timeout=timeout)
    j_notnone = [j for j in j_out if j.sim_results is not None]
    logger.info(f'Succeded {len(j_out)}/{len(jobs_todo)} jobs')
    if len(j_notnone) < len(j_out):
        raise Exception(f'Failed sims {len(j_out)-len(j_notnone)}/{len(j_notnone)}')

    jobs_toread = [j for j in job_list if not hasattr(j, 'read_results')]
    j_out2 = read_elegant_jobs(jobs_toread, dry_run=dry_run, mpi=mpi, timeout=timeout)
    j_notnone2 = [j for j in j_out2 if j.read_results is not None]
    logger.info(f'Read {len(j_out2)}/{len(job_list)} jobs')
    if len(j_notnone2) < len(j_out2):
        raise Exception(f'Failed sims {len(j_out2)-len(j_notnone2)}/{len(j_notnone2)}')

    results = []
    for sj in job_list:
        (data, etaskresp) = sj.sim_results
        (data2, etaskresp2) = sj.read_results

        result_dict = {'sim_data': data, 'read_data': data2,
                       'sim_task': etaskresp, 'read_task': etaskresp2, 'job': sj}

        if dry_run:
            df_fma = pd.DataFrame(data=[{'x': 0.0, 'y': 0.0, 'nux': 0.0, 'nuy': 0.0, 'diffusion': 0.0}])
            return df_fma, (data, data2, etaskresp, etaskresp2, sj)

        _elegant_check_length(data, box)

        df_fma = data2['fma'].df()
        results.append((df_fma, result_dict))
    return results


def track_single_particle_nturns_store_at_tags(box: LatticeContainer, p: Particle,
                                               n_turns: int = 1, backend='ocelot',
                                               dry_run: bool = False):
    """
    Track single particle n turns, storing coordinates at selected elements only
    Positions will be recorded before elements tagged with '.coordinate_watchpoint = True'

    In OCELOT, this is trivial - implemented as usual

    In elegant, watchpoints will be inserted before each tagged element, so use sparingly

    Returned dataframe has one row per watchpoint with arrays of data for each coordinate
    """
    assert isinstance(p, Particle)

    if backend == 'ocelot':
        navi = Navigator(box.lattice)
        t_maps = get_map(box.lattice, box.lattice.totalLen, navi)
        is_watchpoint = [getattr(el, 'coordinate_watchpoint', False) for el in box.sequence]
        data_stores = []
        for tm, el in zip(t_maps, box.sequence):
            if getattr(el, 'coordinate_watchpoint', False):
                tm.data_store = np.zeros((n_turns, 6))
                data_stores.append(tm.data_store)
            else:
                tm.data_store = None
        assert len(t_maps) == len(is_watchpoint)
        if not np.isclose(navi.z0, box.lattice.totalLen, atol=1e-10):
            raise ValueError(f'Navi length ({navi.z0}) not equal to lattice ({box.lattice.totalLen})')

        i = j = 0
        for turn in np.arange(n_turns):
            for tm in t_maps:
                if tm.data_store is not None:
                    tm.data_store[turn, :] = np.array([[p.x, p.px, p.y, p.py, p.tau, p.p]])
                    j += 1
                tm.apply(p)
                i += 1
        assert j == n_turns * len(data_stores)
        assert i == n_turns * len(t_maps)

        df_ocelot = box.df(1)
        df_ocelot['wp'] = is_watchpoint
        df_ocelot = df_ocelot[df_ocelot.wp].reset_index(drop=True)
        df_ocelot['x'] = [np.squeeze(arr[:, 0]) for arr in data_stores]
        df_ocelot['px'] = [np.squeeze(arr[:, 1]) for arr in data_stores]
        df_ocelot['y'] = [np.squeeze(arr[:, 2]) for arr in data_stores]
        df_ocelot['py'] = [np.squeeze(arr[:, 3]) for arr in data_stores]

        # Wipe to save memory
        for tm in t_maps:
            del tm.data_store
        return df_ocelot, None

    elif backend == 'elegant':
        from ..sim import DaskClient, STATE
        from ..elegant import Task, routines
        from ..util import config as cfg

        for el in box.sequence:
            if getattr(el, 'elegant_temporary', False):
                raise AttributeError(f'Element ({el.id}) has temporary elegant flag - state inconsistent')

        # Create elegant task
        lattice_options = {'sr': 0, 'isr': 0, 'dip_kicks': 64, 'quad_kicks': 32,
                           'sext_kicks': 16, 'oct_kicks': 16}
        params = {'label': 'auto_track_nturn_wp'}

        # Build SDDS beam file
        betagamma = box.pc / 0.51099906  # elegant uses 1988 Particle Properties Data Booklet values
        df = pd.DataFrame(columns=['x', 'xp', 'y', 'yp', 't', 'p'], index=[1],
                          data=np.array([p.x, p.px, p.y, p.py, p.tau, betagamma])[np.newaxis, :])
        df.attrs = {'units': ['m', '', 'm', '', 's', 'm$be$nc']}
        sdds_file_name = 'bunch_in.sddsinput'
        parameter_file_map = {PurePath(sdds_file_name): df}

        sj = routines.standard_sim_job(work_folder=cfg.DASK_DEFAULT_WORK_FOLDER,
                                       lattice_options=lattice_options,
                                       add_random_id=True,
                                       parameter_file_map=parameter_file_map,
                                       **params)

        # Insert temporary watch elements where asked, writer will convert to elements
        wp_shared_props = {'mode': 'coordinates', 'start_pass': '0', 'end_pass': '-1'}
        for el in copy.copy(box.sequence):
            if getattr(el, 'coordinate_watchpoint', False):
                wp = Marker(eid=f'__TEMPWPCOORD__{el.id}')
                props = wp_shared_props.copy()
                label = el.id
                seq_num = 0
                props.update({'filename': f'"{sj.run_subfolder.as_posix()}/%s-{label}-{seq_num:02d}.track"'})
                wp.elegant_temporary = True
                wp.elegant_watchpoint = True
                wp.elegant_watchpoint_props = props
                logging.disable(logging.CRITICAL)
                box.insert_elements(wp, before=el)
                logging.disable(logging.NOTSET)

        sj.lattice_file_contents = box.to_elegant(lattice_options=lattice_options)

        temps = [el for el in box.sequence if isinstance(el, Marker) and getattr(el, 'elegant_temporary', False)]
        box.remove_elements(temps)
        for el in box.sequence:
            if getattr(el, 'elegant_temporary', False):
                raise AttributeError(f'Element ({el.id}) has temporary elegant flag - state inconsistent')

        # Elegant taskfile
        t = Task(relative_mode=True, run_folder=sj.run_subfolder, lattice_path=sj.lattice_file_abs_path)
        routines.template_task_track_watchpoint(box, t, n_turns=n_turns, sdds_beam=sdds_file_name,
                                                orbit='reference')
        sj.task_file_contents = t.compile()

        dc = DaskClient()
        futures = dc.submit_to_elegant([sj], dry_run=dry_run, pure=False)
        future = futures[0]
        try:
            (data, etaskresp) = future.result(30)
        except Exception as e:
            import traceback
            print(traceback.format_tb(future.traceback()))
            raise e
        assert etaskresp.state == STATE.ENDED
        if data.returncode != 0:
            logger.error(data)
            raise ValueError(f'Job returned with error code {data.returncode}!')

        futures = dc.read_out([etaskresp], dry_run=dry_run)
        future = futures[0]
        try:
            (data2, etaskresp2) = future.result(30)
        except Exception as e:
            import traceback
            print(traceback.format_tb(future.traceback()))
            print(data.stdout)
            print(data.stderr)
            print(etaskresp)
            raise e
        assert etaskresp2.state == STATE.ENDED_READ

        df_ocelot = box.df()
        df_ocelot['wp'] = [getattr(el, 'coordinate_watchpoint', False) for el in box.sequence]
        df_ocelot = df_ocelot[df_ocelot.wp].reset_index(drop=True)

        if dry_run:
            df_ocelot['x'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
            df_ocelot['px'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
            df_ocelot['y'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
            df_ocelot['py'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
            return df_ocelot, (data, data2, etaskresp, etaskresp2, sj)

        # Make sure elegant length matched - sanity check
        matches = [re.match("^length of beamline IOTA per pass: (-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?) m$", s)
                   for s in data.stdout.splitlines()]
        hits = [m for m in matches if m is not None]
        assert len(hits) == 1
        len_elegant = float(hits[0].group(1))
        assert np.isclose(len_elegant, box.totallen)

        data_dict = {}
        for sdt in data2['track']:
            df = sdt.df
            path = PurePath(sdt.path)
            name = path.name
            eid = name.split('-', 1)[1].rsplit('-', 1)[0]
            assert box.df().id.str.match(eid).sum() == 1
            assert len(df) == 1
            # Particle frame index is particle ID, 1-based
            assert df.loc[1, 'N'] == n_turns  # Full turns for now
            assert df.loc[1, 'P'] == 1
            df['id'] = eid
            data_dict[eid] = df

        assert len(data_dict) == len(data2['track'])
        df_elegant = pd.concat(data_dict.values(), axis=0)
        # print(df_elegant.dtypes)
        # print(df_ocelot.dtypes)
        df_merged = df_ocelot.merge(df_elegant, left_on='id', right_on='id')
        # df_elegant = cen.df()
        # df_elegant = df_elegant.drop(0).reset_index(drop=True)
        # assert np.all(df_elegant['Particles'] == 1)
        # df_elegant.drop(columns=['Charge', 'ElementOccurence', 'Particles'], inplace=True)
        #
        # assert len(df_elegant) == len(df_ocelot)
        # assert np.all(df_ocelot['id'].str.upper() == df_elegant['ElementName'].str.upper())
        # assert np.all(np.isclose(df_ocelot['s_end'], df_elegant['s']))
        # df_merged = df_ocelot.join(df_elegant)
        # df_merged.rename(columns={'Cx': 'x', 'Cy': 'y', 'Cxp': 'px', 'Cyp': 'py'}, inplace=True)
        df_merged.index = df_merged.id
        df_merged.rename_axis(index=None, inplace=True)
        # At high energy, slopes are pretty much canonical vars
        # TODO: Add transformation
        df_merged.rename(columns={'xp': 'px', 'yp': 'py', 'xpi': 'pxi', 'ypi': 'pyi'}, inplace=True)
        return df_merged, (data, data2, etaskresp, etaskresp2, sj)
    else:
        raise AttributeError(f'Unknown backend ({backend})')


def track_bunch_nturns_store_centroid(box: LatticeContainer, n_turns: int = 1, backend='ocelot', dry_run: bool = False,
                                      mpi: int = 0,
                                      bunch_type: str = None,
                                      bunch_dict: Dict = None,
                                      lattice_options_extra: dict = None,
                                      task_options: dict = None,
                                      timeout: int = 60,
                                      wp_mode: str = None,
                                      ):
    """
    Track particle bunch n turns and store centroid
    Positions will be recorded before elements tagged with '.centroid_watchpoint = True'

    In OCELOT, this is trivial - implemented as usual

    In elegant, watchpoints will be inserted before each tagged element, so use sparingly

    Returned dataframe has one row per watchpoint with arrays of data for each coordinate
    """
    assert bunch_type is not None

    if backend == 'elegant' or 'elegant_async':
        from ..sim import DaskClient, STATE
        from ..elegant import Task, routines
        from ..util import config as cfg

        _elegant_check_state(box)

        # Options
        lattice_options = {'sr': 1, 'isr': 1, 'dip_kicks': 64, 'quad_kicks': 32, 'sext_kicks': 16, 'oct_kicks': 16}
        if lattice_options_extra:
            lattice_options.update(lattice_options_extra)
        task_options = task_options or {}
        params = {'label': 'auto_track_bunch_nturn_wp'}

        # If we are doing custom bunch
        if bunch_type == 'custom':
            particles = bunch_dict['particles']
            betagamma = box.pc / 0.51099906  # elegant uses 1988 Particle Properties Data Booklet values
            df_data = np.array([[p.x, p.px, p.y, p.py, p.tau, betagamma] for p in particles])[np.newaxis, :]
            df = pd.DataFrame(columns=['x', 'xp', 'y', 'yp', 't', 'p'], index=np.arange(1, len(df_data) + 1),
                              data=df_data)
            df.attrs = {'units': ['m', '', 'm', '', 's', 'm$be$nc']}
            sdds_file_name = 'bunch_in.sddsinput'
            parameter_file_map = {PurePath(sdds_file_name): df}
        elif bunch_type == 'matched':
            assert isinstance(bunch_dict, dict)
            parameter_file_map = None
        else:
            raise ValueError(f'Unknown bunch type {bunch_type}')

        sj = routines.standard_sim_job(work_folder=cfg.DASK_DEFAULT_WORK_FOLDER,
                                       lattice_options=lattice_options,
                                       task_options=task_options,
                                       add_random_id=True,
                                       parameter_file_map=parameter_file_map,
                                       **params)

        # Insert temporary watch elements where asked, writer will convert to elements
        assert wp_mode in ['centroid', 'parameters']
        wp_shared_props = {'mode': wp_mode, 'start_pass': '0', 'end_pass': '-1'}
        wp_cnt = 0
        for el in copy.copy(box.sequence):
            if getattr(el, f'{wp_mode}_watchpoint', False):
                wp = Marker(eid=f'__TEMPWP{wp_mode.upper()}__{el.id}')
                props = wp_shared_props.copy()
                label = el.id
                seq_num = 0
                props.update({'filename': f'"{sj.run_subfolder.as_posix()}/%s-{label}-{seq_num:02d}.ctrack"'})
                props['FLUSH_INTERVAL'] = '1000'
                wp.elegant_temporary = True
                wp.elegant_watchpoint = True
                props.update(getattr(el, f'{wp_mode}_watchpoint_props', {}))
                wp.elegant_watchpoint_props = props
                logging.disable(logging.CRITICAL)
                box.insert_elements(wp, before=el)
                logging.disable(logging.NOTSET)
                wp_cnt += 1
        if wp_cnt == 0:
            raise Exception('No watchpoints found - check mode?')
        sj.lattice_file_contents = box.to_elegant(lattice_options=lattice_options, silent=True)

        temps = [el for el in box.sequence if isinstance(el, Marker) and getattr(el, 'elegant_temporary', False)]
        box.remove_elements(temps)
        _elegant_check_state(box)

        # Elegant taskfile
        t = Task(relative_mode=True, run_folder=sj.run_subfolder, lattice_path=sj.lattice_file_abs_path)
        if bunch_type == 'custom':
            routines.template_task_track_watchpoint(box, t, n_turns=n_turns,
                                                    sdds_beam=sdds_file_name,
                                                    orbit='reference', **task_options)
        elif bunch_type == 'matched':
            routines.template_task_track_watchpoint(box, t, n_turns=n_turns,
                                                    sdds_beam=None,
                                                    bunch_dict=bunch_dict,
                                                    orbit='closed_offset_no_momentum',
                                                    create_file=True, **task_options)
        else:
            raise Exception
        sj.task_file_contents = t.compile()

        if backend == 'elegant_async':
            sj.params['wp_cnt'] = wp_cnt
            sj.params['n_turns'] = n_turns
            return sj
        else:
            dc = DaskClient()
            futures = dc.submit_to_elegant([sj], dry_run=dry_run, mpi=mpi, pure=False)
            future = futures[0]
            try:
                (data, etaskresp) = future.result(timeout)
            except Exception as e:
                import traceback
                logger.error(f'Sim FAIL: {future} {e}')
                logger.error(traceback.format_tb(future.traceback()))
                raise e
            assert etaskresp.state == STATE.ENDED
            if data.returncode != 0:
                logger.error(data)
                raise ValueError(f'Job returned with error code {data.returncode}!')

            futures = dc.read_out([etaskresp], dry_run=dry_run)
            future = futures[0]
            try:
                (data2, etaskresp2) = future.result(60)
            except Exception as e:
                import traceback
                print(traceback.format_tb(future.traceback()))
                print(future)
                print(e)
            assert etaskresp2.state == STATE.ENDED_READ

            df_ocelot = box.df()
            df_ocelot['wp'] = [getattr(el, 'centroid_watchpoint', False) for el in box.sequence]
            df_ocelot = df_ocelot[df_ocelot.wp].reset_index(drop=True)

            result_dict = {'sim_data': data, 'read_data': data2, 'sim_task': etaskresp, 'read_task': etaskresp2,
                           'job': sj}

            if dry_run:
                df_ocelot['x'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
                df_ocelot['px'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
                df_ocelot['y'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
                df_ocelot['py'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
                return df_ocelot, result_dict

            _elegant_check_length(data, box)

            data_dict = {}
            assert len(data2['ctrack']) == wp_cnt
            for sdt in data2['ctrack']:
                df = sdt.df()
                # if not np.isclose(box.totallen, df.loc[1,'Cs']):
                #    print(df.loc[1,'Cs']) # change_t messes with this
                # assert len(df) == n_turns
                path = PurePath(sdt.path)
                name = path.name
                eid = name.split('-', 1)[1].rsplit('-', 1)[0]
                assert box.df().id.str.match(eid).sum() == 1
                df.rename(columns={'Cx': 'x', 'Cy': 'y', 'Cxp': 'px', 'Cyp': 'py'}, inplace=True)
                # assert df.iloc[-1, df.columns.get_loc('Pass')] == n_turns-1
                df['id'] = eid
                data_dict[eid] = [df]

            assert len(data_dict) == len(data2['ctrack'])
            # df_elegant = pd.concat(data_dict.values(), axis=0)
            # df_merged = df_ocelot.merge(df_elegant, left_on='id', right_on='id')
            df_data = pd.DataFrame.from_dict(data=data_dict, orient='index')
            result_dict['processed'] = df_data
            # df_elegant.drop(columns=['Charge', 'ElementOccurence', 'Particles'], inplace=True)
            # df_merged.index = df_merged.id
            # df_merged.rename_axis(index=None, inplace=True)
            # df_merged.reset_index(drop=True)
            # df_merged.rename(columns={'xp': 'px', 'yp': 'py', 'xpi': 'pxi', 'ypi': 'pyi'}, inplace=True)
            return df_data, result_dict
    else:
        raise AttributeError(f'Unknown backend ({backend})')


def process_elegant_jobs(job_list, dry_run: bool, mpi: int = None, timeout: float = 60):
    from ..sim import DaskClient, STATE, ElegantSimJob
    import dask.distributed
    assert all(type(t).__name__ == ElegantSimJob.__name__ for t in job_list)

    dc = DaskClient()
    futures = dc.submit_to_elegant(job_list, dry_run=dry_run, mpi=mpi, pure=False)
    t1 = time.perf_counter()

    cnt = 0
    for future in dask.distributed.as_completed(futures):
        try:
            (data, etaskresp) = future.result(timeout)
            tasks = [t for t in job_list if t.label == etaskresp.label]
            assert len(tasks) == 1
            j = tasks[0]
            if data.returncode != 0:
                logger.error(data)
            j.sim_results = (data, etaskresp)
        except Exception as e:
            import traceback
            logger.error(f'Sim FAIL: {future} {e}')
            logger.error(traceback.format_tb(future.traceback()))
            #j.sim_results = None
        cnt += 1
        tpj = (time.perf_counter() - t1) / cnt
        logger.info(
            f'Processed {cnt}/{len(job_list)} - {tpj:.2f}s/job, {tpj * (len(job_list) - cnt) / 60 / 60 :.3f} hrs left ')
    for j in job_list:
        if not hasattr(j,'sim_results'):
            j.sim_results = None
    return job_list


def read_elegant_jobs(job_list, dry_run: bool, mpi: int = None, timeout: float = 60):
    from ..sim import DaskClient, STATE, ElegantSimJob
    import dask.distributed
    assert all(type(t).__name__ == ElegantSimJob.__name__ for t in job_list)

    dc = DaskClient()
    read_futures = dc.read_out(job_list, dry_run=dry_run, pure=False)

    cnt = 0
    for future in dask.distributed.as_completed(read_futures):
        t2 = time.perf_counter()
        try:
            (data2, etaskresp2) = future.result(timeout=120)
            tasks = [t for t in job_list if t.label == etaskresp2.label]
            assert len(tasks) == 1
            j = tasks[0]
            assert etaskresp2.state.value == STATE.ENDED_READ.value
            j.read_results = (data2, etaskresp2)
        except Exception as e:
            import traceback
            print(traceback.format_tb(future.traceback()))
            print(future)
            print(e)
            #read_results.append(None)
            #raise e
        cnt += 1
        tpj = (time.perf_counter() - t2) / cnt
        logger.info(f'Read {cnt}/{len(job_list)} - {tpj:.2f}s/job')
    for j in job_list:
        if not hasattr(j,'read_results'):
            j.read_results = None
    return job_list




def process_track_bunch_nturns_store_centroid(box, job_list, dry_run: bool, mpi, timeout=60):
    logger.info(f'Processing {len(job_list)} jobs')
    jobs_todo = [j for j in job_list if not hasattr(j, 'sim_results')]
    logger.info(f'Submitting {len(jobs_todo)} jobs')
    j_out = process_elegant_jobs(jobs_todo, dry_run=dry_run, mpi=mpi, timeout=timeout)
    j_notnone = [j for j in j_out if j.sim_results is not None]
    logger.info(f'Ran {len(j_out)}/{len(job_list)} jobs')
    if len(j_notnone) < len(j_out):
        raise Exception(f'Failed sims {len(j_out)-len(j_notnone)}/{len(j_notnone)}')

    jobs_toread = [j for j in job_list if not hasattr(j, 'read_results')]
    j_out2 = read_elegant_jobs(jobs_toread, dry_run=dry_run, mpi=mpi, timeout=timeout)
    j_notnone2 = [j for j in j_out2 if j.read_results is not None]
    logger.info(f'Read {len(j_out2)}/{len(job_list)} jobs')
    if len(j_notnone2) < len(j_out2):
        raise Exception(f'Failed sims {len(j_out2)-len(j_notnone2)}/{len(j_notnone2)}')

    df_ocelot = box.df()
    df_ocelot['wp'] = [getattr(el, 'centroid_watchpoint', False) for el in box.sequence]
    df_ocelot = df_ocelot[df_ocelot.wp].reset_index(drop=True)

    results = []
    for sj in job_list:
        (data, etaskresp) = sj.sim_results
        (data2, etaskresp2) = sj.read_results
        n_turns = sj.params['n_turns']

        result_dict = {'sim_data': data, 'read_data': data2,
                       'sim_task': etaskresp, 'read_task': etaskresp2, 'job': sj}

        if dry_run:
            df_ocelot['x'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
            df_ocelot['px'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
            df_ocelot['y'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
            df_ocelot['py'] = [np.zeros(n_turns) * np.nan for _ in range(len(df_ocelot))]
            return df_ocelot, result_dict

        _elegant_check_length(data, box)

        data_dict = {}
        assert len(data2['ctrack']) == sj.params['wp_cnt']
        for sdt in data2['ctrack']:
            df = sdt.df()
            # assert len(df) == n_turns
            path = PurePath(sdt.path)
            name = path.name
            eid = name.split('-', 1)[1].rsplit('-', 1)[0]
            assert box.df().id.str.match(eid).sum() == 1
            df.rename(columns={'Cx': 'x', 'Cy': 'y', 'Cxp': 'px', 'Cyp': 'py'}, inplace=True)
            # assert df.iloc[-1, df.columns.get_loc('Pass')] == n_turns-1
            df['id'] = eid
            data_dict[eid] = [df]

        assert len(data_dict) == len(data2['ctrack'])
        df_data = pd.DataFrame.from_dict(data=data_dict, orient='index')
        result_dict['processed'] = df_data
        results.append([df_data, result_dict])
    return results


def track_single_particle_1turn_store_particles(box: LatticeContainer, p: Particle,
                                                n_turns: int = 1, backend='ocelot',
                                                dry_run: bool = False):
    """
    Track single particle 1 turn - supports elegant backend
    Works because we are 'abusing' centroid file
    """
    assert isinstance(p, Particle)
    assert n_turns == 1

    if backend == 'ocelot':
        p_list = track_nturns_store_particles(box, p, n_turns)
        df_ocelot = box.df(n_turns)
        df_ocelot['x'] = [p.x for p in p_list[1:]]
        df_ocelot['y'] = [p.y for p in p_list[1:]]
        df_ocelot['px'] = [p.px for p in p_list[1:]]
        df_ocelot['py'] = [p.py for p in p_list[1:]]
        df_ocelot = df_ocelot[df_ocelot.loc[:, 'class'] != 'Edge'].reset_index(drop=True)
        return df_ocelot, None

    elif backend == 'elegant':
        from .. import sim
        from .. import elegant
        from ..util import config as cfg

        # Create elegant task
        lattice_options = {'sr': 0, 'isr': 0, 'dip_kicks': 64, 'quad_kicks': 32,
                           'sext_kicks': 16, 'oct_kicks': 16}
        params = {'label': 'auto_track_1turn'}

        # Build SDDS beam file
        # from scipy import constants
        # betagamma = box.pc / constants.value('electron mass energy equivalent in MeV')  # check
        betagamma = box.pc / 0.51099906  # elegant uses 1988 Particle Properties Data Booklet values
        df = pd.DataFrame(columns=['x', 'xp', 'y', 'yp', 't', 'p'], index=[1],
                          data=np.array([p.x, p.px, p.y, p.py, p.tau, betagamma])[np.newaxis, :])
        df.attrs = {'units': ['m', '', 'm', '', 's', 'm$be$nc']}
        sdds_file_name = 'bunch_in.sddsinput'
        parameter_file_map = {PurePath(sdds_file_name): df}

        et = elegant.routines.standard_sim_job(work_folder=cfg.DASK_DEFAULT_WORK_FOLDER,
                                               lattice_options=lattice_options,
                                               add_random_id=True,
                                               parameter_file_map=parameter_file_map,
                                               **params)
        et.lattice_file_contents = box.to_elegant(lattice_options=lattice_options)

        # Elegant taskfile
        t = elegant.Task(relative_mode=True, run_folder=et.run_subfolder, lattice_path=et.lattice_file_abs_path)
        elegant.routines.template_task_track_single(box, t, n_turns=n_turns, sdds_beam=sdds_file_name,
                                                    orbit='reference')
        et.task_file_contents = t.compile()

        dc = sim.DaskClient()
        futures = dc.submit_to_elegant([et], dry_run=dry_run, pure=False)
        future = futures[0]
        try:
            (data, etaskresp) = future.result(30)
        except Exception as e:
            import traceback
            print(traceback.format_tb(future.traceback()))
            raise e
        assert etaskresp.state == sim.STATE.ENDED
        if data.returncode != 0:
            logger.error(data)
            raise ValueError(f'Job returned with error code {data.returncode}!')

        futures = dc.read_out([etaskresp], dry_run=dry_run)
        future = futures[0]
        (data2, etaskresp2) = future.result(30)
        assert etaskresp2.state == sim.STATE.ENDED_READ

        if dry_run:
            return None, None

        # Make sure elegant length matched - sanity check
        matches = [re.match("^length of beamline IOTA per pass: (-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?) m$", s)
                   for s in data.stdout.splitlines()]
        hits = [m for m in matches if m is not None]
        assert len(hits) == 1
        len_elegant = float(hits[0].group(1))
        assert np.isclose(len_elegant, box.totallen)

        cen = data2['cen']
        df_ocelot = box.df()
        df_elegant = cen.df()
        df_elegant = df_elegant.drop(0).reset_index(drop=True)
        assert np.all(df_elegant['Particles'] == 1)
        df_elegant.drop(columns=['Charge', 'ElementOccurence', 'Particles'], inplace=True)
        df_ocelot = df_ocelot[df_ocelot.loc[:, 'class'] != 'Edge'].reset_index(drop=True)
        assert len(df_elegant) == len(df_ocelot)
        assert np.all(df_ocelot['id'].str.upper() == df_elegant['ElementName'].str.upper())
        assert np.all(np.isclose(df_ocelot['s_end'], df_elegant['s']))
        df_merged = df_ocelot.join(df_elegant)
        df_merged.rename(columns={'Cx': 'x', 'Cy': 'y', 'Cxp': 'px', 'Cyp': 'py'}, inplace=True)
        return df_merged, (data, data2, etaskresp, etaskresp2)
    else:
        raise AttributeError(f'Unknown backend ({backend})')


# Helper methods


def tracking_step(lat: MagneticLattice, particle_list, dz: float, navi: Navigator):
    """
    Tracking for a fixed step dz
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


def tracking_step_element(lat: MagneticLattice, particle_list, el_type: Type[Element], navi: Navigator):
    """ Tracking step until next element of given class or end of lattice """
    if navi.n_elem == len(lat.sequence):
        raise Exception("Tracking step called while already at end of lattice?")
    t_maps = get_map_element(lat, el_type, navi)
    for tm in t_maps:
        tm.apply(particle_list)


def get_map_element(lattice: MagneticLattice, el_type: Type[Element], navi: Navigator):
    """ Gets maps until next element of given class or end of lattice """
    nelems = len(lattice.sequence)
    t_maps = []
    i = navi.n_elem
    dl = 0.0
    while True:
        elem = lattice.sequence[i]
        t_maps.append(elem.transfer_map(elem.l))
        dl += elem.l
        i += 1
        if isinstance(elem, el_type) or i >= nelems:
            break

    navi.z0 += dl
    navi.sum_lengths += dl
    navi.n_elem = i
    return t_maps


def get_map(lattice, dz, navi):
    """ From OCELOT """
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
        # if i in navi.proc_kick_elems:
        #    break
    if abs(dz) > 1e-10:
        TM.append(elem.transfer_map(dz))
    navi.z0 += dz
    navi.sum_lengths = L - elem.l
    navi.n_elem = i
    return TM


def track_nturns_fast(lat: MagneticLattice, nturns: int, track_list: List[Track_info],
                      nsuperperiods: int = 1,
                      save_track: bool = True, print_progress: bool = False,
                      merge:bool = True):
    """
    Modified OCELOT track_nturns method that tries to speed up tracking
    1 - It uses numpy, since numexpr is bad for few particles
    2 - It merges matrix elements, while leaving everything else
    """
    if merge:
        # Do not merge anything that is not a matrix
        exclusions = [el for el in lat.sequence if not isinstance(el.transfer_map, SecondTM) and not type(el.transfer_map) == TransferMap]
        new_lat = merger(lat, remaining_elems=exclusions)
        assert lat.totalLen == new_lat.totalLen
        print(f'Merged lattice from ({len(lat.sequence)}) to ({len(new_lat.sequence)}) elements')
    else:
        new_lat = lat
    #print(lat.sequence[1].transfer_map)

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
