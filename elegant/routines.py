__all__ = ['template_fma_setup', 'template_task_closed_orbit',
           'template_task_track_single', 'standard_sim_job']

from typing import Dict
import numpy as np
from .taskfile import Task
from ..lattice import LatticeContainer
from ..util import id_generator
from ..sim import ElegantSimJob
import pathlib
from pathlib import Path, PurePath, PurePosixPath
from ocelot import Marker


def standard_sim_job(work_folder: PurePath, lattice_options: Dict = None, add_random_id: bool = True,
                     parameter_file_map: Dict = None, **kwargs):
    assert isinstance(work_folder, PurePath)

    defaults = {'label': 'test_label', 'lat_label': 'lat_label'}
    defaults.update(kwargs)

    if add_random_id:
        defaults['label'] += '.' + id_generator(8)

    label = defaults['label']
    lattice_options = lattice_options or {'sr': 1, 'isr': 1, 'dip_kicks': 64, 'quad_kicks': 32,
                                          'sext_kicks': 16, 'oct_kicks': 16}
    lattice_tag = 'v86_{sr:d}_{isr:d}_{dip_kicks:02d}_{quad_kicks:02d}_{sext_kicks:02d}_{oct_kicks:02d}'.format_map(
        lattice_options)

    work_folder = work_folder or PurePosixPath('/home/nkuklev/studies/dask_tasks/')
    run_subfolder = pathlib.PurePosixPath(label + '/')
    lattice_file = pathlib.PurePosixPath(lattice_tag + '.' + defaults['lat_label'] + '.lte')

    etask = ElegantSimJob(label=label,
                          lattice_options=lattice_options,
                          work_folder=work_folder,
                          data_folder=None,
                          run_subfolder=run_subfolder,
                          task_file_name=defaults['label'] + '.ele',
                          lattice_file_name=lattice_file,
                          parameter_file_map=parameter_file_map)
    return etask


def template_task_track_single(box: LatticeContainer, t: Task, **kwargs):
    """ Task template for tracking single particle """
    assert len(kwargs) <= 3
    orbit = kwargs.get('orbit', 'closed_offset')
    # n_turns should be 1, else centroid makes no sense, but it wont cause any errors
    n_turns = kwargs.get('n_turns', 1)
    sdds_beam = kwargs.get('sdds_beam', None)
    assert n_turns == 1

    # Coordinates will be extracted from centroid file, no need for watchpoints
    t.setup_run_setup(p=box.pc, beamline='iota', rootname='test', centroid=True)
    # t.action_insert_watch(name='COORDS_START', label='t-START', mode='coordinates',
    #                      start_pass=0, end_pass=-1, loc_name=box.lattice.sequence[0].id)
    t.setup_run_control(n_passes=n_turns)
    if sdds_beam:
        # Input single particle via sdds
        t.setup_sdds_beam(path=sdds_beam)
    else:
        # Just make one on reference orbit
        t.setup_bunched_beam()
    t.action_twiss(create_file=True, output_at_each_step=1)
    t.setup_closed_orbit(verbosity=1)
    t.action_track(orbit=orbit)


def template_task_track_watchpoint(box: LatticeContainer, t: Task, **kwargs):
    """ Task template for tracking single particle at origin watchpoint over many turns """
    assert len(kwargs) <= 3
    orbit = kwargs.get('orbit', 'closed_offset')
    n_turns = kwargs.get('n_turns', 1)
    sdds_beam = kwargs.get('sdds_beam', None)
    assert n_turns >= 1

    # Coordinates will be extracted from centroid file, no need for watchpoints
    t.setup_run_setup(p=box.pc, beamline='iota', rootname='test')
    # assert isinstance(box.sequence[0], Marker)
    # eid = box.lattice.sequence[0].id
    # t.action_insert_watch(name=f'COORDS_{eid}', label=eid, mode='coordinates',
    #                       start_pass=0, end_pass=-1, loc_name=eid)
    t.setup_run_control(n_passes=n_turns)
    if sdds_beam:
        # Input single particle via sdds
        t.setup_sdds_beam(path=sdds_beam)
    else:
        # Just make one on reference orbit
        t.setup_bunched_beam()
    t.action_twiss(create_file=True, output_at_each_step=1)
    t.setup_closed_orbit(verbosity=1)
    t.action_track(orbit=orbit)


def template_task_closed_orbit(box: LatticeContainer, t: Task, **kwargs):
    """ Task template for determining closed orbit - also computes twiss to check stability """
    # Expect no extra parameters
    assert len(kwargs) == 0
    # t.setup_subprocess_mkdir()
    t.setup_run_setup(p=box.pc, beamline='iota', rootname='test')
    t.setup_run_control()
    t.action_twiss(create_file=True, output_at_each_step=1)
    t.setup_bunched_beam()
    t.setup_closed_orbit(verbosity=1, create_file=True)
    t.action_track(orbit='default')
    return t


def template_fma_setup(box: LatticeContainer,
                       t: Task,
                       **kwargs):
    assert 'oc' in kwargs and kwargs['oc']
    keys = set(kwargs.keys())
    critical_args = {'beamline', 'turns', 'rf_mode', 'errors'}
    missing = critical_args.difference(keys)
    if missing:
        raise Exception(f'Missing critical keys: ({missing})')
    rf_mode = kwargs['rf_mode']
    errors = kwargs['errors']
    turns = kwargs['turns']

    t.setup_global_settings()
    t.setup_subprocess_mkdir()
    t.setup_run_setup(p=box.pc, beamline=kwargs['beamline'], rootname='test', parameters=True)

    if errors is None:
        pass
    else:
        assert isinstance(errors, Path)
        assert str(errors).split('.')[-1] == 'erl'
        t.action_load_parameters(path=errors)

    # if errors == 'noerrors' or errors is None:
    #     pass
    # elif errors == 'pregen':
    #     t.action_load_parameters(path='errors.erl')
    # else:
    #     raise Exception(f'Unknown error mode: ({errors})')

    if 'oc' in kwargs:
        t.action_load_parameters(path=kwargs['oc'])

    #     # Lattice optics knob
    #     if quads:
    #         action_load_parameters(f, lattice, iota, subfolder, name='quad_parameters/ml/'+quads)

    if 'chrom' in kwargs:
        chrom = kwargs['chrom']
        if not np.isnan(chrom[1]):
            t.setup_chromaticity(families=chrom[0], dnux_dp=chrom[1], dnuy_dp=chrom[2],
                                 change_defined_values=1, n_iterations=10, exit_on_failure=1)

    t.setup_run_control(n_passes=turns, n_steps=1)

    # Twiss on clean modified lattice
    t.action_twiss(create_file=True, full=True)
    t.action_twiss(create_file=True, ext='twi2', output_at_each_step=1, full=True)

    # RF
    if rf_mode == 'rf':
        t.setup_rf(total_voltage=300.0, each_step=True)
    elif rf_mode is None:
        pass
    else:
        raise Exception('RF mode MUST be specified')

    t.setup_bunched_beam()


def fma_read_results():
    pass
