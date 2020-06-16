__all__ = ['template_fma_setup']

import numpy as np
from .taskfile import Task
from ..lattice import LatticeContainer
from pathlib import Path


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