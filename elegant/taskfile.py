import sys
import io
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace
import functools
import inspect
import datetime
import hashlib

import numpy as np


def task(name=None):
    """
    Decorator for all actions, to save on formatting pain. It also adds any kwargs to command
    that are not consumed by the wrapped function signature. Implementation...is not trivial.
    :param name:
    :return:
    """
    assert name

    def decorator(func):
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            fullargspec = inspect.getfullargspec(func)

            # print('Function', func)
            # print('KWargs', fullargspec.kwonlyargs)
            # print('Default kwargs', fullargspec.kwonlydefaults)
            # print('Passed kwargs', kwargs)
            # print('Pos args', fullargspec.args)

            # This ensures we can identify any 'extra' keywords that are passed, and write them appropriately
            assert fullargspec.args == ['self']
            assert fullargspec.varargs == 'args'
            extra_kwargs = {k: v for k, v in kwargs.items() if k not in fullargspec.kwonlyargs} if \
                (kwargs and fullargspec.kwonlyargs) else (kwargs or {})
            #print('Extra kwargs', extra_kwargs)
            lines = func(self, *args, **kwargs)
            lines = ['&' + name] + [' ' + ln + ',' for ln in lines] + \
                    [f' {k} = {v},' for k, v in extra_kwargs.items()] + ['&end\n']
            self._add_lines(lines)

        return inner

    return decorator


class Task:
    def __init__(self, lattice: str, run_folder: str):
        self.f = io.StringIO('', newline='\n')
        self.lattice = lattice
        self.rf = run_folder
        self._add_lines(self._header())
        # self.iota = iota

    def reset(self):
        self.f = io.StringIO('', newline='\n')
        self._add_lines(self._header())

    def compile(self):
        data = self.f.getvalue()
        h = hashlib.md5(data.encode('ascii')).hexdigest()
        data += '! CHECKSUM:MD5 ' + ':'.join(h[i:i + 2] for i in range(0, len(h), 2))
        return data

    def _add_lines(self, lines):
        self.f.writelines("%s\n" % s for s in lines)

    def _header(self):
        strings = ['! This file is auto-generated by pyIOTA',
                   '! Module: {} v0.4'.format(Task.__module__ + '.' + Task.__qualname__),
                   '! Send bug reports to nkuklev@uchicago.edu',
                   '',
                   f'! TIME: {datetime.datetime.now().isoformat()}',
                   '']
        return strings

    def custom_command(self, cmd):
        self._add_lines([cmd])

    @task(name='run_setup')
    def setup_run_setup(self, *args, p, beamline='iota', rootname='test',
                        acceptance=False, final=False, output=False, parameters=False, sigma=False,
                        load_balance=False, silent=False, **kwargs):
        if silent:
            acceptance = final = output = parameters = sigma = False
        strings = [f'lattice = {self.lattice}',
                   f'use_beamline = {beamline}',
                   f'tracking_updates = 1',
                   f'p_central_mev = {p}',  # iota.header["PC"] * 1000
                   f'default_order = 3',
                   ]
        if rootname != '':
            strings.append(f'rootname = {rootname}')
        if load_balance:
            strings.append(f'load_balancing_on = 1')
        sf = self.rf
        if sigma:
            strings.append(f'sigma        = {sf}/%s.sig')
        if output:
            strings.append(f'output       = {sf}/%s.out')
        if final:
            strings.append(f'final        = {sf}/%s.fin')
        if acceptance:
            strings.append(f'acceptance   = {sf}/%s.acc')
        if parameters:
            strings.append(f'parameters   = {sf}/%s.param')
        return strings

    @task(name='run_control')
    def setup_run_control(self, *args, n_passes=1, n_steps=1, **kwargs):
        assert not (np.isnan(n_passes) or np.isnan(n_steps))
        assert 1 <= n_passes <= 1e7 and 1 <= n_steps <= 1e6
        strings = [f'n_passes = {n_passes}',
                   f'n_steps = {n_steps}']
        return strings

    @task(name='global_settings')
    def setup_global_settings(self, *args):
        strings = ['mpi_io_read_buffer_size = 128000000',
                   'mpi_io_write_buffer_size = 128000000',
                   'inhibit_fsync = 1',
                   # ' mpi_randomization_mode = 3',
                   ]
        return strings

    @task(name='semaphores')
    def setup_semaphores(self, *args):
        return []

    @task(name='subprocess')
    def setup_subprocess_mkdir(self, *args):
        return ['command = "mkdir {}"'.format(self.rf)]

    ###

    @task(name='twiss_output')
    def action_twiss(self, *args, full=False, create_file=False, ext=None, **kwargs):
        strings = ['matched = 1']
        if create_file:
            if ext:
                strings.append(f'filename = {self.rf}/%s.{ext}')
            else:
                strings.append(f'filename = {self.rf}/%s.twi')
        if full:
            strings.extend(['statistics = 1', 'radiation_integrals = 1', 'compute_driving_terms = 1'])
        return strings

    @task(name='bunched_beam')
    def setup_bunched_beam(self, *args, mode=None, create_file=False, gridspec=None, **kwargs):
        if not mode:
            return []
        strings = []
        if create_file:
            strings.append(f'bunch = {self.rf}/%s.bun')
        if mode in ['DA_upperhalf', 'DA_upperright']:
            assert gridspec
            x, nx, y, ny = gridspec
            # For DA distribution, elegant forms uniform grid between +- sqrt(emittance*beta)
            # with no slope, so we backtrack the values to fit bounds.
            if mode == 'DA_upperright':
                x = x / 2
                strings.append(
                    f'centroid[0] = {x + 2 * x / nx:10e}, 0, {y / ny:10e}, 0, 0, 0, !shift to upper right quadrant and off zeroes')
            strings.extend([f'n_particles_per_bunch = {nx * nx}',
                            f'beta_x = 1.0',
                            f'emit_x = {x ** 2:10e}',
                            f'beta_y = 1.0',
                            f'emit_y = {y ** 2:10e}',
                            f'distribution_type[0] = "dynamic-aperture","dynamic-aperture","hard-edge"',
                            f'distribution_cutoff[0] = {nx}, {ny}, 1'])

        return strings

    @task(name='track')
    def action_track(self, *args, orbit='closed', **kwargs):
        strings = ['soft_failure=0']
        if orbit == 'closed_center':
            strings.extend(
                ['center_on_orbit = 1', 'center_momentum_also = 1', 'offset_by_orbit = 0', 'offset_momentum_also = 0'])
        elif orbit == 'closed_offset_no_momentum':
            strings.extend(
                ['center_on_orbit = 0', 'center_momentum_also = 1', 'offset_by_orbit = 1', 'offset_momentum_also = 0'])
        elif orbit == 'closed_offset':
            strings.extend(
                ['center_on_orbit = 0', 'center_momentum_also = 1', 'offset_by_orbit = 1', 'offset_momentum_also = 1'])
        elif orbit == 'reference':
            strings.extend(
                ['center_on_orbit = 0', 'center_momentum_also = 0', 'offset_by_orbit = 0', 'offset_momentum_also = 0'])
        elif orbit == 'default':
            pass
        else:
            raise ValueError('Orbit mode must be specified!')
        return strings

    @task(name='chromaticity')
    def setup_chromaticity(self, *args, families: list = [], dnux_dp: float = np.nan, dnuy_dp: float = np.nan, **kwargs):
        assert not (np.isnan(dnux_dp) or np.isnan(dnuy_dp))
        assert -20 < dnux_dp < 20 and -20 < dnuy_dp < 20
        strings = [f'sextupoles = "{families}"',
                   f'dnux_dp = {dnux_dp}',
                   f'dnuy_dp = {dnuy_dp}']
        return strings

    @task(name='rf_setup')
    def setup_rf(self, *args, harmonic=4, total_voltage=0.0, create_file=False, each_step=False, **kwargs):
        # iota.GetElementsOfType('RFCAVITY')[0]['VOLT'] * 1e6)
        strings = [f'harmonic = {harmonic}',
                   f'total_voltage = {total_voltage}']
        if create_file:
            strings.append(f'filename = {self.rf}/%s.rf')
        if each_step:
            strings.append('set_for_each_step = 1')
        return strings

    @task(name='closed_orbit')
    def setup_closed_orbit(self, *args, create_file=False, centroid_start=True, **kwargs):
        strings = [f'closed_orbit_iterations = 10000',
                   f'iteration_fraction = 0.5']
        if create_file:
            strings.append(f'output = {self.rf}/%s.clo')
        if not centroid_start:
            strings.append('start_from_centroid = 0')
        return strings
    ###

    @task(name='load_parameters')
    def action_load_parameters(self, *args, path, change_values=1):
        strings = [f'filename = {path}',
                   f'change_defined_values = {change_values}',
                   f'verbose = 1']
        return strings

    @task(name='insert_elements')
    def action_insert(self, *args, definition, location_name=None, location_type=None):
        assert location_type or location_name
        strings = [f'element_def = "{definition}"',
                   f'verbose = 1']
        strings.append(f'type = {location_type}') if location_type else strings.append(f'type = "*"')
        strings.append(f'name = {location_name}') if location_name else strings.append(f'name = "*"')
        return strings

    @task(name='insert_elements')
    def action_insert_watch(self, *args, name, label, mode, seq_num=0, start_pass=0, end_pass=-1, loc_name=None,
                            loc_type=None):
        assert loc_type or loc_name
        strings = [
            f'element_def = "{name}: watch, filename="{self.rf}/%s-{label}-{seq_num:02d}.sdds",mode="{mode}",start_pass={start_pass},end_pass={end_pass}"',
            f'verbose = 1']
        strings.append(f'type = {loc_type}') if loc_type else strings.append(f'type = "*"')
        strings.append(f'name = {loc_name}') if loc_name else strings.append(f'name = "*"')
        return strings
