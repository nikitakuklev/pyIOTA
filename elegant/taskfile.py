__all__ = ['Task', 'Optimizer', 'IOTAOptimizer']

import io
import functools
import inspect
import datetime
import hashlib
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pyIOTA
from pyIOTA.lattice.elements import LatticeContainer
from ocelot import Hcor, Vcor, Quadrupole, SBend, Marker, Monitor


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
            # print('Extra kwargs', extra_kwargs)
            lines = func(self, *args, **kwargs)
            lines = ['&' + name] + [' ' + ln + ',' for ln in lines] + \
                    [f' {k} = {v},' for k, v in extra_kwargs.items()] + ['&end\n']
            self._add_lines(lines)

        return inner

    return decorator


class Task:
    def __init__(self, lattice: str, run_folder: str, auto_paths: bool = True, knobs_dir: Path = None):
        """
        General elegant taskfile generator
        :param lattice:
        :param run_folder: The root folder for all data of the run
        :param auto_paths: Automatically prepend paths for some commands, like parameter load
        :param knobs_dir: Parameter subdirectory
        """
        self.f = io.StringIO('', newline='\n')
        self.lattice = lattice
        self.rf = run_folder
        self.auto_paths = auto_paths
        if not knobs_dir:
            knobs_dir = Path('parameters/')
        self.knobs_dir = knobs_dir
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
                   f"! Module: {Task.__module__ + '.' + Task.__qualname__} v{pyIOTA.__version__}",
                   '! Source: ?',
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
                   #f'tracking_updates = 1',
                   f'p_central_mev = {p}',  # iota.header["PC"] * 1000
                   #f'default_order = 3',
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
    def setup_bunched_beam(self, *args, mode=None, create_file=False, gridspec=None, latspec=None, **kwargs):
        """
        Bunched beam generation in elegant.
        :param args:
        :param mode:
        :param create_file:
        :param gridspec: maxima and number of points
        :param latspec: optics functions for non-DA distributions to enable computing real dimensions
        :param kwargs:
        :return:
        """
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
        elif mode in ['HE_upperright', 'SH_upperright']:
            assert gridspec, latspec
            if mode.startswith('HE'):
                dist = 'hard-edge'
            elif mode.startswith('SH'):
                dist = 'shell'
            else:
                raise Exception(f'Bad mode {mode}')
            x, nx, y, ny = gridspec
            beta_x, beta_y = latspec
            x = x / 2
            y = y / 2
            strings.append(f'centroid[0] = {x + 2 * x / nx:10e}, 0, {y + 2 * y / ny:10e}, 0, 0, 0, !shift to upper right quadrant and off zeroes')
            # xmax^2 = emittance*beta
            strings.extend([f'n_particles_per_bunch = {nx * ny}',
                            f'beta_x = 1.0',
                            f'emit_x = {x ** 2 / beta_x:10e}',
                            f'beta_y = 1.0',
                            f'emit_y = {y ** 2 / beta_y:10e}',
                            f'distribution_type[0] = "{dist}","{dist}","hard-edge"',
                            f'distribution_cutoff[0] = 1, 1, 1'])
        else:
            raise Exception

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

    @task(name='frequency_map')
    def action_frequency_map(self, *args, silent=False, quadratic_spacing=False, full_grid_output=True,
                             include_changes=False, x=None, nX=None, y=None, nY=None, d=None, nD=None, **kwargs):
        strings = [f'output = {self.rf}/%s.fma']
        # If in kwargs, they will get auto-placed by annotating method
        if 'xmin' not in kwargs:
            strings.append('xmin = 1e-6')
        if 'ymin' not in kwargs:
            strings.append('ymin = 1e-6')

        if silent:  # default 1
            strings.append('verbosity = 0')
        if quadratic_spacing:
            strings.append('quadratic_spacing = 1')
        if full_grid_output:
            strings.append('full_grid_output = 1')
        if include_changes:
            strings.append('include_changes = 1')
        if x:
            strings.append(f'xmax = {x}')
            strings.append(f'nx = {nX}')
        if y:
            strings.append(f'ymax = {y}')
            strings.append(f'ny = {nY}')
        if d:
            strings.append(f'delta_max = {d}')
            strings.append(f'delta_min = {-d}')
            strings.append(f'ndelta = {nD}')
        return strings

    @task(name='chromaticity')
    def setup_chromaticity(self, *args, families: list = [], dnux_dp: float = np.nan, dnuy_dp: float = np.nan,
                           **kwargs):
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
        strings = []
        if create_file:
            strings.append(f'output = {self.rf}/%s.clo')
        if not centroid_start:
            strings.append('start_from_centroid = 0')
        return strings

    ###

    @task(name='load_parameters')
    def action_load_parameters(self, *args, path: Union[str, Path], change_values: int = 1, auto_path: bool = None):
        auto_path = auto_path or self.auto_paths
        if auto_path:
            assert isinstance(path, str) or isinstance(path, Path)
            path = str(self.rf / self.knobs_dir / path)
        else:
            assert isinstance(path, str)
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
    def action_insert_watch(self, *args, name, label, mode, seq_num=0, start_pass=0, end_pass=-1,
                            loc_name=None, loc_type=None):
        strings = [
            f'element_def = "{name}: watch, filename="{self.rf}/%s-{label}-{seq_num:02d}.sdds",mode="{mode}",start_pass={start_pass},end_pass={end_pass}"',
            f'verbose = 1']
        if not loc_name and not loc_type:
            raise Exception('Elegant requires name or type to be given')
        strings.append(f'type = {loc_type}') if loc_type else strings.append(f'type = "*"')
        strings.append(f'name = {loc_name}') if loc_name else strings.append(f'name = "*"')
        return strings

    # Errors

    @task(name='error_control')
    def setup_error_control(self, *args, clean_step=False,):
        strings = [' clear_error_settings = 1',
                   ' no_errors_for_first_step = {}'.format(clean_step),
                   f' error_log = {self.rf}/%s.erl']
        return strings

    @task(name='error_control')
    def setup_error_control_end(self):
        strings = ['summarize_error_settings = 1']
        return strings

    @task(name='error_element')
    def setup_error_element(self, *args, name, elname, item, amplitude):
        strings = [f' name = "{name}"',
                   f' element_type = "{elname}"',
                   f' item = "{item}"',
                   f' amplitude={amplitude:10e}',
                   ' cutoff = 2.0',
                   ' bind = 0',
                   ' fractional = 0']
        return strings

    # Optimizer

    @task(name='optimize')
    def optimizer_run(self, *args, **kwargs):
        strings = ['summarize_setup = 1']
        return strings

    @task(name="optimization_setup")
    def optimizer_setup(self, *args, n_evaluations=1500, n_passes=2, n_restarts=1, target=1e-10, term_log='%s.terms',
                        sparsing_factor=100, **kwargs):
        strings = [f'n_evaluations = {n_evaluations}',
                   f'n_passes = {n_passes}',
                   f'n_restarts = {n_restarts}',
                   # f'tolerance = 1e-16',
                   f'target =  {target}',
                   f'soft_failure = 0',
                   # f'log_file = /dev/tty',
                   f'term_log_file = {self.rf}/{term_log}',
                   f'output_sparsing_factor = {sparsing_factor}',
                   f'verbose = 1']
        return strings


class Optimizer:
    """
    Elegant optimization specification generator. Includes shorthand commands for common tune and orbit knob derivation.
    """

    def __init__(self, name: str = 'optimizer', box: LatticeContainer = None):
        self.name = name
        self.strings = []
        self.box = box

    def add_term(self, term: str, weight=1.0):
        strings = f'&optimization_term term = "{term}", weight = {weight}, verbose = 0 &end'
        self.strings.append(strings)

    def add_variable(self, name: str, item: str, step_size: Union[int, float] = 1.0, lower_limit: float = 0.0,
                     upper_limit: float = 0.0):
        strings = f'&optimization_variable name = {name}, item = {item}, step_size = {step_size}'
        if lower_limit: strings += f', lower_limit = {lower_limit}'
        if upper_limit: strings += f', upper_limit = {upper_limit}'
        strings += ' &end'
        self.strings.append(strings)

    def add_link(self, source: str, target: str, parameter: str):
        strings = f'&link_elements target = {target}, source={source}, item={parameter}, equation={parameter} &end'
        self.strings.append(strings)

    def add_comment(self, comment: str):
        self.strings.append(comment)

    def sene(self, var1: Union[str, float, int], var2: Union[str, float, int], eps: float = 1e-4):
        strings = "{} {} {} sene".format(var1, var2, eps)
        # self.strings.append(strings)
        return strings

    def dump(self):
        print(f'Dumping ({len(self.strings)}) lines of optimizer setup')
        print('\n'.join(self.strings) + '\n')

    def compile(self):
        return '\n'.join(self.strings) + '\n'


class IOTAOptimizer(Optimizer):
    """
    Elegant optimizer with additional meta-directives that enforce IOTA-specific constraints for integrable optics
    """

    def set_integrable_phase_advances(self, phases):
        self.add_comment('!Phases')
        (dxi, dxo, dyi, dyo) = phases
        # nux = xi+xo; dnux=5.3-nux; dxo=5.0-xo
        # nuy = yi+yo; dnuy=5.3-nuy; dyo=5.0-yo
        # total
        self.add_term(f"nux 5.3 {dxi + dxo:+0.3f} + 0.00001 sene")
        self.add_term(f"nuy 5.3 {dyi + dyo:+0.3f} + 0.00001 sene")
        # outside
        self.add_term(f"NLR2#1.nux NLR1#1.nux - abs 5 {dxo:+0.3f} + 0.00001 sene")
        self.add_term(f"NLR2#1.nuy NLR1#1.nuy - abs 5 {dyo:+0.3f} + 0.00001 sene")

    def enforce_zero_dispersion(self, tol: float = 1e-2):
        self.add_comment('!Dispersion')
        self.add_term(f"NLR2#1.etax 0 {tol} sene")
        self.add_term(f"NLR2#1.etay 0 {tol} sene")
        self.add_term(f"NLR1#1.etax 0 {tol} sene")
        self.add_term(f"NLR1#1.etay 0 {tol} sene")

        self.add_term(f"IOR#1.etax 0 {tol} sene")
        self.add_term(f"IOR#1.etay 0 {tol} sene")

    def enforce_insert_symmetry(self):
        self.add_comment('!Symmetry')
        self.add_term("NLR2#1.betax NLR1#1.betax - 0 0.001 sene")
        self.add_term("NLR2#1.betay NLR1#1.betay - 0 0.001 sene")
        self.add_term("NLL2#1.betax NLL1#1.betax - 0 0.001 sene")
        self.add_term("NLL2#1.betay NLL1#1.betay - 0 0.001 sene")

    def enforce_LR_optics_symmetry(self):
        self.add_comment('!LR Symmetry')
        self.add_term("NLL1#1.betax NLR2#1.betax - 0 0.001 sene")
        self.add_term("NLL1#1.betay NLR2#1.betay - 0 0.001 sene")

    def link_LR_quads(self, box: LatticeContainer, parameter: str = 'K1'):
        box = box or self.box
        quads = [q.id for q in box.get_elements(Quadrupole) if q.id.startswith('Q') and q.id.endswith('R')]
        linked = []
        swap_dict = {'L': 'R', 'R': 'L'}
        for el_name in quads:
            if el_name not in linked:
                opposite_name = el_name[:-1] + swap_dict[el_name[-1]]
                if opposite_name not in quads:
                    raise Exception(f'Quad ({el_name}) is missing the symmetric partner!')
                self.add_link(target=opposite_name, source=el_name, parameter=parameter)
                linked.append(el_name)
                linked.append(opposite_name)
        print(f'Linked ({len(linked) // 2}) quads')

    ### adding variables

    def add_corrector_variables(self, box: LatticeContainer = None, correctors: list = None, limits: dict = None,
                                step_size: float = 0.1, verbose: bool = False):
        """
        Adds a kick strength variable for all correctors. Assumes they are all integrated with dipoles/skew quads.
        :param step_size:
        :param correctors:
        :param box:
        :return:
        """
        box = box or self.box
        correctors = correctors or box.correctors
        if not limits: limits = {}
        for c in correctors:
            if c in limits:
                vmax, vmin = limits[c]
            else:
                vmax = vmin = 0.0
            if isinstance(c, Vcor):
                item = 'VKICK'
            elif isinstance(c, Hcor):
                item = 'HKICK'
            else:
                continue
            if isinstance(c.ref_el, SBend):
                if item != 'HKICK':
                    raise Exception(
                        f'Corrector ({c.id}) can only be horizontal, since its in a dipole ({c.ref_el.id})!')
                item = 'FSE_DIPOLE'
                self.add_variable(name=c.ref_el.id, item=item, step_size=step_size / 100, lower_limit=vmin,
                                  upper_limit=vmax)
                if verbose: print(f'Added corrector ({c.id}) - variable ({c.ref_el.id}-{item})')
            else:
                self.add_variable(name=c.ref_el.id, item=item, step_size=step_size, lower_limit=vmin, upper_limit=vmax)
                if verbose: print(f'Added corrector ({c.id}) - variable ({c.ref_el.id}-{item})')

    def add_main_quad_variables(self, box: LatticeContainer = None, parameter: str = 'K1',
                                side: Optional[str] = 'R', verbose: bool = False):
        """
        Adds a variable for all main quadrupoles
        :param box:
        :param parameter: which parameter to add, by default quad strength
        :param side: restrict list to a particular side (and any central elements like QE3)
        :return:
        """
        box = box or self.box
        swap_dict = {'L': 'R', 'R': 'L'}
        if side:
            # To account for QE3, which is in the middle and so has no L/R designator
            quads = [q for q in box.get_elements(Quadrupole) if
                     q.id.startswith('Q') and not q.id.endswith(swap_dict[side])]
        else:
            quads = [q for q in box.get_elements(Quadrupole) if q.id.startswith('Q')]
        for el_name in quads:
            self.add_variable(name=el_name, item=parameter, step_size=1)
        if verbose: print(f'Added {len(quads)} variables of parameter {parameter} for main quads')

    def add_corrector_constraints(self, box: LatticeContainer = None, max_kicks: list = None, min_kicks: list = None,
                                  tol: float = 1e-3, kicks_dict: dict = None, verbose: bool = False):
        """
        Adds strength constraints on correctors - typically this is dictated by current limits
        :param verbose:
        :param kicks_dict:
        :param box:
        :param max_kicks: List of maximum positive kicks. Must either match corrector count or be a singleton
        :param min_kicks: List of maximum negative kicks. Must either match corrector count or be a singleton.
         If none, -max is used.
        :param tol:
        :return:
        """
        box = box or self.box
        if kicks_dict:
            for c, (vmin, vmax) in kicks_dict.items():
                if isinstance(c, str):
                    obj_list = [el for el in box.correctors if c in el.id]
                    if len(obj_list) == 1:
                        c = obj_list[0]
                    else:
                        raise Exception(f'Found matches ({obj_list}) for search string ({c}) - require exactly 1 match')
                if isinstance(c, Vcor):
                    item = 'VKICK'
                elif isinstance(c, Hcor):
                    item = 'HKICK'
                else:
                    continue
                if isinstance(c.ref_el, SBend):
                    if item != 'HKICK':
                        raise Exception(
                            f'Corrector ({c.id}) can only be horizontal, since its in a dipole ({c.ref_el.id})!')
                    item = 'FSE_DIPOLE'
                self.add_term(f"{c.ref_el.id}.{item} {vmax} {tol} segt")
                self.add_term(f"{c.ref_el.id}.{item} {vmin} {tol} selt")
                if verbose: print(f'Added constraint {c.ref_el.id}-{item}:({vmax}|{vmin})@{tol}')
        else:
            if not max_kicks: raise Exception('Maximum kicks must be specified')
            max_kicks = np.array(max_kicks)
            if len(max_kicks) == 1:
                max_kicks = np.repeat(max_kicks, len(box.correctors))
            if not min_kicks:
                min_kicks = -1 * np.array(max_kicks)
            elif len(min_kicks) == 1:
                min_kicks = np.array(min_kicks)
                min_kicks = np.repeat(min_kicks, len(box.correctors))
            else:
                min_kicks = np.array(min_kicks)
            assert len(min_kicks) == len(max_kicks)

            for i, c in enumerate(box.correctors):
                if isinstance(c, Vcor):
                    item = 'VKICK'
                elif isinstance(c, Hcor):
                    item = 'HKICK'
                else:
                    continue
                if isinstance(c.ref_el, SBend):
                    if item != 'HKICK':
                        raise Exception(
                            f'Corrector ({c.id}) can only be horizontal, since its in a dipole ({c.ref_el.id})!')
                    item = 'FSE_DIPOLE'
                self.add_term(f"{c.ref_el.id}.{item} {max_kicks[i]} {tol} segt")
                self.add_term(f"{c.ref_el.id}.{item} {min_kicks[i]} {tol} selt")
                if verbose: print(f'Added constraint {c.ref_el.id}-{item}:({max_kicks[i]}|{min_kicks[i]})@{tol}')

    def add_orbit_constraints(self, box: LatticeContainer = None, goals: Union[list, dict] = None, tol: float = 1e-4,
                              zero_other_monitors: bool = True, verbose: bool = False):
        """
        Simple method that adds orbit position goals to all monitors. Consider inserting additional markers and using
        range-based selection for more complicated cases.
        :param verbose:
        :param zero_other_monitors:
        :param box:
        :param goals: list of tuples (x,y) for each monitor
        :param tol:
        :return:
        """
        box = box or self.box
        if goals is None:
            goals = [(0, 0)]
        if isinstance(goals, list):
            monitors = [m for m in box.monitors if m in box.lattice.sequence]
            if len(goals) != len(monitors):
                if len(goals) == 1:
                    goals = goals * len(monitors)
                else:
                    raise Exception(
                        f'Number of goals ({len(goals)}) does not match sequence monitor count ({len(box.monitors)})')
            for m, (x, y) in zip(box.monitors, goals):
                self.add_term(self.sene(f'{m.id}#1.xco', x, tol))
                self.add_term(self.sene(f'{m.id}#1.yco', y, tol))
        elif isinstance(goals, dict):
            monitors_other = [m for m in box.monitors if m in box.lattice.sequence and m not in goals.keys()]
            for m, (x, y) in goals.items():
                self.add_term(self.sene(f'{m.id}#1.xco', x, tol))
                self.add_term(self.sene(f'{m.id}#1.yco', y, tol))
            if zero_other_monitors:
                for m in monitors_other:
                    self.add_term(self.sene(f'{m.id}#1.xco', str(0), tol))
                    self.add_term(self.sene(f'{m.id}#1.yco', str(0), tol))
                if verbose: print(
                    f'Added {len(monitors_other) + len(goals)} orbit constraints, ({len(monitors_other)} set by '
                    f'default to reference orbit)')
            else:
                if verbose: print(f'Added {len(goals)} orbit constraints ({len(monitors_other)} monitors untouched)')
            return np.array([[m.s, x, y] for m, (x, y) in goals.items()]), np.array(
                [[m.s, 0, 0] for m in monitors_other])
        else:
            raise Exception(f'Unknown type of goals provided')

    def add_orbit_constraints_for_region(self, box: LatticeContainer = None, region: tuple = (-1, -1),
                                         orbit: tuple = None, tol: float = 1e-4, touch_markers: bool = False,
                                         eps: float = 1e-10,
                                         bump_terms_weight: float = 10.0, verbose: bool = False):
        box = box or self.box
        assert len(region) == 2
        assert not touch_markers # can't do closed orbit with elegant markers
        if orbit is None:
            orbit = (0, 0)
        box.update_element_positions()
        bound_lower = -np.inf if region[0] <= -40 else region[0] - eps
        bound_upper = np.inf if region[1] >= 40 else region[1] + eps
        ring_len = box.lattice.totalLen

        for el in box.lattice.sequence:
            if isinstance(el, Monitor) or (touch_markers and isinstance(el, Marker)):
                if ((bound_lower <= el.s_mid <= bound_upper) or
                        (bound_lower < 0 and el.s_mid > ring_len + bound_lower) or
                        (bound_upper > ring_len and el.s_mid < bound_upper - ring_len)):
                    el.orbit_goal_x = orbit[0]
                    el.orbit_goal_y = orbit[1]
                    self.add_term(self.sene(f'{el.id}#1.xco', el.orbit_goal_x, tol), weight=bump_terms_weight)
                    self.add_term(self.sene(f'{el.id}#1.yco', el.orbit_goal_y, tol), weight=bump_terms_weight)
                else:
                    el.orbit_goal_x = 0
                    el.orbit_goal_y = 0
                    self.add_term(self.sene(f'{el.id}#1.xco', el.orbit_goal_x, tol))
                    self.add_term(self.sene(f'{el.id}#1.yco', el.orbit_goal_y, tol))
        return np.array(
            [[m.s_mid, m.orbit_goal_x, m.orbit_goal_y] for m in box.get_elements(Monitor)]), None

    def set_NL_drift_optics(self, shiftx=False, shifty=False, verbose: bool = False):
        self.add_comment('!Betastar')
        if shiftx:
            self.add_term("MN01_2#1.alphax 0 0.00001 sene")
        else:
            self.add_term("IOR#1.alphax 0 0.00001 sene")

        if shifty:
            self.add_term("MN01_2#1.alphay 0 0.00001 sene")
        else:
            self.add_term("IOR#1.alphay 0 0.00001 sene")

    # def set_orbit_fitpoints(self, Lattice):
