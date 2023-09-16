__all__ = ['Knob', 'KnobVariable', 'parse_knobs', 'parse_lattice']

import functools
import itertools
import logging
import operator
import random
import re
import time
import uuid
from pathlib import Path
from re import Pattern
from typing import Callable, Dict

import numpy as np
from ocelot import Cavity, Drift, Edge, Hcor, Monitor, Multipole, Quadrupole, SBend, Sextupole, \
    Solenoid, Vcor

from pyiota.acnet import DoubleDevice, DoubleDeviceSet

logger = logging.getLogger(__name__)


# Static methods to parse files from 6Dsim format

@functools.lru_cache(maxsize=None)
def __get_pattern(k) -> Pattern:
    ke = re.escape(k)
    pattern = r'([()\s=\-+*/])' + f'({ke})' + r'([\s()=\-+*/])'
    return re.compile(pattern)


def __replace_vars(assign: str, variables: dict) -> str:
    """
    Replaces all variables with their stored values, adding brackets to maintain eval order.
    :return: Processed string
    """
    assign = ' ' + assign + ' '
    for k, v in variables.items():
        if k in assign:
            # ke = re.escape(k)
            p = __get_pattern(k)
            # pattern = r'[()\s=\-+*/]' + f'({ke})' + r'[\s()=\-+*/]'
            assign = p.sub(r'\1' + f'({v})' + r'\3', assign)
            # print(assign)
            # match = re.search(pattern, assign)
            # matches = re.findall(pattern, assign)
            # for match in matches:
            #     if match is not None:
            #         #print(k, pattern, match)
            #         s = match.start(1)
            #         assign = assign[:s] + f'({v})' + assign[match.end(1):]
            #         #print(assign)
            #         #assign = assign.replace(k, '(' + v + ')', 1)
    assign = assign.strip()
    return assign


def __resolve_vars(assign: str,
                   variables_dicts: list[dict],
                   recursion_limit: int = 50
                   ) -> str:
    """
    Recursively resolves variables and replaces with values, until none are left
    """
    i = 0
    # print(f'Resolve start: {assign}')
    # while '$' in assign or
    # Only true is have letters...EEEEEEE
    # while assign != assign.swapcase():
    result = None
    bad = False
    while '$' in assign:  # while True:
        try:
            for vd in variables_dicts:
                assign = __replace_vars(assign, vd)
            # logger.debug(f'Resolve iteration {i}: {assign}')
            i += 1

            sqrt = np.sqrt
            tan = Tan = np.tan
            sin = Sin = np.sin
            cos = Cos = np.cos
            result = eval(assign)
            break
        except Exception:
            if i > recursion_limit:
                bad = True
                break
    if bad:
        raise Exception(f'Unable to resolve line {assign} against {variables_dicts}')
    return assign  # , result


def __parse_id_line(line: str, output_dict: Dict,
                    resolve_against: Dict, verbose: bool = False
                    ) -> None:
    """
    Parse single element definition line. WARNING - this uses eval(), and is VERY VERY UNSAFE.
    :param line: Line to parse
    :param output_dict: Dictionary into which to place the result
    :param resolve_against: Dictionary of variable values
    :param verbose:
    :return: None
    """
    if not line.startswith('ID:'):
        return
    line = line[4:].strip()
    if verbose: print(f'Analyzing line: {line}')
    splits = line.split()
    name, element = splits[0], splits[1]
    pars_dict = {}
    if len(splits) > 2:
        pars = splits[2:]
        if not len(pars) % 2 == 0:
            raise Exception(f'Number of parameters {pars} is not even (i.e. paired)')
        if verbose: logger.debug(f'Parsing parameters ({pars})')
        for (k, v) in zip(pars[::2], pars[1::2]):
            if verbose:
                logger.debug(f'Property ({k}) is ({v})')
            try:
                resolved = __resolve_vars(v, [resolve_against])
                vr = eval(resolved)
            except Exception as e:
                # if verbose: print(f'Failed to evaluate, using as string ({k})-({v}) : {e})')
                vr = v
            if verbose: logger.debug(f'Property ({k}) resolves to ({vr})')
            pars_dict[k] = vr
    if verbose: print(f'Element ({name}) resolved to ({element})({pars_dict})')
    output_dict[name] = (element, pars_dict)


def parse_lattice(fpath: Path,
                  verbose: bool = False,
                  verbose_vars: bool = False,
                  merge_dipole_edges: bool = True,
                  dipole_merge_method: int = 2,
                  allow_edgeless_dipoles=False,
                  unsafe: bool = False
                  ):
    """
    Parses 6Dsim lattice into native OCELOT object, preserving all compatible elements. All variables
    are resolved to their final numeric values, and evaluated - this is an UNSAFE operation for unstrusted
    input, so make sure your lattice files are not corrupted.
    :return: lattice_ocelot, correctors_ocelot, monitors_ocelot, info_dict, var_dict - latter 2 are dicts
    of parsing information and of all present KnobVariables
    """
    # returned objects
    lattice_list = []
    correctors_ocelot = []
    monitors_ocelot = []
    # internal vars
    keywords = {'INFO:': 1, 'ELEMENTS:': 2, 'CORRECTORS:': 3, 'MONITORS:': 4, 'LATTICE:': 5,
                'END': 6
                }
    variables = {'$PI': str(np.pi)}
    variables_2 = {}
    specials = ['$PI']
    specials_2 = ['PI']
    lattice_vars = {}
    elements = {}
    correctors = {}
    monitors = {}
    lattice_str = ''
    # Other parameters
    pc = None
    N = None

    with open(fpath, 'r') as f:
        lines = f.readlines()
        lcnt = 0
        for line in lines:
            lcnt += 1
            line = line.strip()
            if line.startswith('//') or line == '':
                continue

            if line in keywords:
                if verbose_vars: print('MODE:', line)
                mode = keywords[line]
                if mode == keywords['END']:
                    if verbose_vars: print('DONE')
                    break
                else:
                    continue

            if mode == keywords['INFO:']:
                if '$' not in line or '=' not in line:
                    if line[:2] == 'pc':
                        pc = float(line.split(':')[1])
                        continue
                    elif line[:2] == 'Np':
                        N = float(line.split(':')[1])
                        continue
                    else:
                        continue
                else:
                    var, assign = line.split('=')
                    var = var.strip()
                    assign = assign.strip()
                    # print('Analyzing line: {}'.format(line))
                    assign = __resolve_vars(assign, [variables])
                    # variables[var] = assign
                    # print('Variable ({}) resolved to ({})'.format(var, assign))
                    # print(variables)
                    # extra funcs
                    sqrt = np.sqrt
                    tan = Tan = np.tan
                    sin = Sin = np.sin
                    cos = Cos = np.cos
                    try:
                        value = eval(assign)
                        if var not in specials:
                            assert var not in variables, f'{var} already in {variables}'
                        variables[var] = value
                        vs = var.strip('$')
                        if vs not in specials_2:
                            assert vs not in variables_2, f'{vs} already in {variables_2}'
                        variables_2[var.strip('$')] = value
                    except Exception as e:
                        print(f'Failed on line #{lcnt}: {line}')
                        print(variables)
                        print(variables_2)
                        print(assign)
                        raise e
                    if verbose_vars: logger.debug(f'Variable ({var}) evaluated to ({value})')
                    lattice_vars[var.strip('$')] = value
                    continue
            elif mode == keywords['ELEMENTS:']:
                __parse_id_line(line, elements, variables, verbose_vars)
                continue
            elif mode == keywords['CORRECTORS:']:
                __parse_id_line(line, correctors, variables, verbose_vars)
                continue
            elif mode == keywords['MONITORS:']:
                __parse_id_line(line, monitors, variables, verbose_vars)
                continue
            elif mode == keywords['LATTICE:']:
                lattice_str += line + ' '
                continue
    lattice = lattice_str.split()

    # Sanity checks
    if pc is None:  # or N is None:
        raise Exception(f'Key parameters are not defined! pc:{pc} N:{N}')
    for el in lattice:
        if el not in elements:
            raise Exception(f'Element {el} not in elements list!!!')
    for el, (eltype, props) in correctors.items():
        relative_to = props.get('El', None)
        if relative_to is None or relative_to not in elements:
            raise Exception(
                    f'Relative position of {el} does not have referenced element {relative_to}')
    for el, (eltype, props) in monitors.items():
        assert eltype == 'BPM'
        relative_to = props.get('El', None)
        if relative_to is None or relative_to not in elements:
            raise Exception(
                    f'Relative position of {el} does not have referenced element {relative_to}')
    assert len(set(elements)) == len(elements)
    assert len(set(correctors)) == len(correctors)
    assert len(set(monitors)) == len(monitors)
    # Lattice is ok, now create OCELOT lattice
    # types = [eltype for el, (eltype, props) in elements.items()]
    # print(set(types))
    mapping = {'Dipole': SBend, 'Acc': Cavity, 'Sol': Solenoid, 'DipEdge': Edge,
               'Mult': Multipole, 'Quad': Quadrupole, 'SQuad': Quadrupole,
               'Gap': Drift, 'BPM': Monitor, 'YCorrector': Vcor, 'XCorrector': Hcor,
               'ElementShiftCorrector': None,
               'OrbitShiftCorrector': None,
               'RFCorrector': 'RF'
               }
    field_to_gradient_factor = (pc / 1000 / 0.299792458) / 10.0  # kG*m
    magnetic_rigidity_Tm = (pc / 1000.0 / 0.299792458)
    magnetic_rigidity_kGcm = magnetic_rigidity_Tm * 100 * 10

    def gradient_scale(g):
        return g / field_to_gradient_factor

    def length_scale(l: float):
        return l / 100.0

    shared_parameter_map = {'L': ('l', length_scale)}  # apply function to these variables

    for item in lattice:
        (el_type, props) = elements[item]
        try:
            if el_type not in mapping:
                raise Exception(f'Type {el_type} not in translation map')
            type_mapped = mapping[el_type]
            if type_mapped is None:
                continue
            shared_kwargs = {}
            for k, v in shared_parameter_map.items():
                result = props.get(k, None)
                if result is not None:
                    shared_kwargs[v[0]] = v[1](result)
            shared_kwargs['eid'] = item.upper()
            if verbose:
                print(f'{item} - {shared_kwargs} | {props}')
            if type_mapped == Drift:
                oel = Drift(**shared_kwargs)
            elif type_mapped == SBend:
                if 'G' in props:
                    k1 = props['G'] / field_to_gradient_factor
                else:
                    k1 = 0.0
                fint = props.get('fringeK', 0.0)
                rho = (magnetic_rigidity_kGcm / props['Hy']) / 100.0
                # oel = SBend(angle=props['L'] / ((pc / 0.299792458) / props['Hy']),
                oel = SBend(angle=props['L'] / 100.0 / rho,
                            fint=fint,
                            **shared_kwargs)

                oel.k1 = k1
                oel.h = 1 / rho
                oel.Hy = props['Hy']
                if unsafe:
                    oel.e1 = props.get('inA', 0.0) * np.pi / 180 * np.sign(oel.angle)
                    oel.e2 = props.get('outA', 0.0) * np.pi / 180 * np.sign(oel.angle)
                if verbose:
                    print(f'{item} - {oel.__dict__}')
            elif type_mapped == Edge:
                assert 'Hy' in props
                # H = angle/length = 1/rho -> angle = length * H  = length / rho
                rho = (magnetic_rigidity_kGcm / props['Hy']) / 100
                e = props.get('inA', 0.0) * np.pi / 180
                gap = props.get('poleGap', 0.0) / 100  # full gap, not half
                fint = props.get('fringeK', 0.0)
                if 'poleGap' not in props or e != 0.0:
                    assert unsafe
                    # logger.warning(f'Weird edge {item} - {props}, ignoring inv. curvature')
                oel = Edge(**shared_kwargs, gap=gap, fint=fint, edge=e)
                oel.h = 1 / rho
                oel.Hy = props['Hy']
                if verbose:
                    print(f'{item} - {oel.__dict__}')
            elif type_mapped == Quadrupole:
                if el_type == 'SQuad':
                    oel = Quadrupole(k1=props['G'] / field_to_gradient_factor,
                                     tilt=np.pi / 4,
                                     **shared_kwargs)
                else:
                    oel = Quadrupole(k1=props['G'] / field_to_gradient_factor,
                                     **shared_kwargs)
            elif type_mapped == Multipole:
                if 'M2N' in props and len(props) <= 3:
                    # Sextupole (length and M2N props)
                    oel = Sextupole(k2=float(props['M2N']) / field_to_gradient_factor,
                                    **shared_kwargs)
                else:
                    if unsafe:
                        logger.warning(f'Missing multipole strengths, using drift ({item}|{props})')
                        oel = Drift(**shared_kwargs)
                    else:
                        raise Exception(
                                f"Multipole that is not a sextupole detected ({item}|{el_type}|{type_mapped}|{props})")
            elif type_mapped == Solenoid:
                oel = Drift(**shared_kwargs)
            elif type_mapped == Cavity:
                oel = Cavity(v=props['U'],
                             freq=props['F'],
                             **shared_kwargs)
            else:
                raise Exception(
                        f'Empty ocelot object produced converting ({item}|{el_type}|{type_mapped}|{props})')
            lattice_list.append(oel)
        except Exception as e:
            print('Element conversion failed!')
            print((item, el_type, props))
            raise e

    # Dipole edge combination
    lattice_ocelot = lattice_list.copy()
    insert_offset = 0
    if merge_dipole_edges and dipole_merge_method == 1:
        assert len(lattice) == len(lattice_list)
        edge_count = 0
        edge_warn_exclusions = []
        for i, (k, v) in enumerate(zip(lattice, lattice_list)):
            if isinstance(v, Edge):
                edge_count += 1
            if isinstance(v, SBend):
                e1 = lattice_list[i - 1]
                e2 = lattice_list[i + 1]
                if not isinstance(e1, Edge) or not isinstance(e2, Edge):
                    if allow_edgeless_dipoles:
                        if k not in edge_warn_exclusions:
                            logger.warning(
                                    f'Found sector bend {k} without edge elements (e1={e1.id}, e2={e2.id})')
                            edge_warn_exclusions.append(k)
                        continue
                    else:
                        raise Exception(
                                f'Found sector bend {k} without edge elements - this is not allowed')
                assert e1.gap == e2.gap
                assert e1.fint == e2.fint
                v.fint = e1.fint
                v.fintx = e2.fint
                v.gap = e1.gap
                v.e1 = e1.edge * np.sign(v.angle)
                v.e2 = e2.edge * np.sign(v.angle)
                lattice_ocelot.remove(e1)
                lattice_ocelot.remove(e2)
                if verbose: print(f'Integrated edges for dipole {k}')
                edge_count -= 2
        assert edge_count == 0
    elif merge_dipole_edges and dipole_merge_method == 2:
        assert len(lattice) == len(lattice_list)
        edge_count = 0
        edge_warn_exclusions = []
        for i, (k, v) in enumerate(zip(lattice, lattice_list)):
            if isinstance(v, Edge):
                edge_count += 1
            if isinstance(v, SBend):
                e1 = lattice_list[i - 1]
                e2 = lattice_list[i + 1]

                if (v.e1 is not None and v.e1 != 0.0) or (v.e2 is not None and v.e2 != 0.0):
                    assert v.e1 is not None and v.e2 is not None
                    logger.warning(
                            f'Found bend {k} with nonzero angles {v.e1=} {v.e2=}, adding extra edges')
                    e1l = Edge(eid=v.id + '_1')
                    e2l = Edge(eid=v.id + '_2')
                    assert v.l != 0.0
                    e1l.h = e2l.h = v.angle / v.l
                    e1l.edge = v.e1
                    e2l.edge = v.e2
                    lattice_ocelot.insert(i + insert_offset, e1l)
                    lattice_ocelot.insert(i + 2 + insert_offset, e2l)
                    insert_offset += 2

                if not isinstance(e1, Edge) or not isinstance(e2, Edge):
                    assert not isinstance(e1, Edge) and not isinstance(e2, Edge)
                    if allow_edgeless_dipoles:
                        if k not in edge_warn_exclusions:
                            logger.warning(
                                    f'Found bend {k} without edge elements (e1={e1.id}, e2={e2.id})')
                            edge_warn_exclusions.append(k)
                        continue
                    else:
                        raise Exception(
                                f'Found bend {k} without edge elements - this is not allowed')

                v.fint = e1.fint
                v.fintx = e2.fint
                v.gap = e1.gap

                if v.l != 0.0:
                    h = v.angle / v.l
                else:
                    h = 0.0

                for e in [e1, e2]:
                    eh = getattr(e, 'h', None)
                    if eh is None:
                        e.h = h
                    else:
                        if not np.isclose(eh, h, atol=1e-10, rtol=0.0):
                            raise Exception(
                                    f'Curvature mismatch for edge ({e.id}|{e.h=}) and bend ({v.id}|{h=})')

                e1.angle = e2.angle = v.angle
                if v.angle < 0:
                    e1.edge *= -1
                    e2.edge *= -1
                assert e1.gap == e2.gap == v.gap
                assert e1.dx == e2.dx == v.dx
                assert e1.dy == e2.dy == v.dy
                assert e1.tilt == e2.tilt == v.tilt
                assert e1.dtilt == e2.dtilt == v.dtilt
                assert e1.k1 == e2.k1
                if e1.k1 != v.k1:
                    logger.warning(f'k1 mismatch in {k}: {e1.k1=} vs {v.k1=}')
                assert e1.l == e2.l == 0.0

                assert e1.h_pole == v.h_pole1
                assert e2.h_pole == v.h_pole2

                if getattr(v, 'e1') is None:
                    v.e1 = e1.edge
                if getattr(v, 'e2') is None:
                    v.e2 = e2.edge

                e1.pos = 1
                e2.pos = 2

                if verbose: print(f'Updated edges for dipole {k} (method 2)')
                edge_count -= 2
        assert edge_count == 0

    for item, (el_type, props) in correctors.items():
        shared_kwargs = {}
        if el_type not in mapping:
            raise Exception(f'Type {el_type} not in translation map')
        type_mapped = mapping[el_type]
        shared_kwargs['eid'] = item.upper()
        if type_mapped is None:
            continue
        if 'L' in props and props['L'] != 0.0:
            raise Exception(f"Corrector {item} with non-zero length detected")
        refs = [el for el in lattice_ocelot if el.id == props['El'].upper()]
        if len(refs) > 1:
            raise Exception(f"Corrector {item} is has too many reference elements {props['El']}")
        if len(refs) == 0:
            raise Exception(f"Corrector {item} is missing reference element {props['El']}")
        if type_mapped == 'RF':
            continue
        elif type_mapped == Vcor:
            oel = Vcor(**shared_kwargs)
        elif type_mapped == Hcor:
            oel = Hcor(**shared_kwargs)
        else:
            raise Exception(
                    f'Empty ocelot object produced converting ({item}|{el_type}|{type_mapped}|{props})')
        oel.end_turn = props['endTurn'] if 'endTurn' in props else None
        oel.ref_el = refs[0]
        correctors_ocelot.append(oel)

    for item, (el_type, props) in monitors.items():
        shared_kwargs = {}
        if el_type not in mapping:
            raise Exception(f'Type {el_type} not in translation map')
        type_mapped = mapping[el_type]
        shared_kwargs['eid'] = item.upper()
        if type_mapped is None:
            continue
        if 'L' in props and props['L'] != 0.0:
            raise Exception("Monitor with non-zero length detected!")
        refs = [el for el in lattice_ocelot if el.id == props['El'].upper()]
        if len(refs) > 1:
            raise Exception(f"Monitor {item} is has too many reference elements {props['El']}")
        if len(refs) == 0:
            raise Exception(f"Monitor {item} is missing reference element {props['El']}")
        if 'Shift' not in props:
            raise Exception(f"Monitor {item} has bad shift")
        oel = Monitor(**shared_kwargs)
        oel.ref_el = refs[0]
        oel.shift = props['Shift'] / 100.0
        monitors_ocelot.append(oel)

    logger.info(
            f'Parsed {len(lattice_ocelot)} objects, {len(correctors_ocelot)} correctors, {len(monitors_ocelot)} monitors')
    # print(f'Parsed OK - {len(lattice_ocelot)} objects, '
    #      f'{len(correctors_ocelot)} correctors, {len(monitors_ocelot)} monitors')

    info_dict = {'source_file': str(fpath), 'source': '6dsim', 'pc': pc, 'N': N}
    var_dict = {k: KnobVariable(kind='$', var='$' + k, value=v) for k, v in lattice_vars.items()}
    return lattice_ocelot, correctors_ocelot, monitors_ocelot, info_dict, var_dict


def parse_transfer_maps(file: Path):
    """
    Parses outpout of sixdsim transfer map export. Note special, old format.
    :param file:
    :return: Dictionary with key as target element name
    """
    with open(file, 'r') as f:
        tmaps = {}
        lines = f.readlines()
        lines = lines[14:]
        for i, l in enumerate(lines):
            if i % 7 == 0:
                src, to = l.split('->')[0].strip(), l.split('->')[1].strip()
            else:
                j = i % 7 - 1
                if j == 0:
                    matr = []
                # print(list(filter(None, l.strip().split(' '))))
                matr.append(np.array(list(filter(None, l.strip().split(' ')))).astype(np.float64))
                if j == 5:
                    tmaps[to] = (src, np.stack(matr))  # print(dataarr)
        tmaps['start'] = ('end', np.identity(6))
    return tmaps


class AbstractKnob:
    """
    Superclass of all knobs, which are collections of KnobVariables representing setpoints of devices
    """

    def __init__(self, name: str):
        self.name = name or uuid.uuid4().hex[:10]
        self.verbose = False


class Knob(AbstractKnob):
    """
    ACNET-based implementation of a knob
    """

    def __init__(self, name: str = None, variables: dict = None):
        self.vars = variables or {}
        self.absolute = True
        super().__init__(name)

    def make_absolute(self):
        raise Exception("Not implemented yet")
        # assert not self.absolute
        # ds = DoubleDeviceSet(name=self.name,
        #                      members=[DoubleDevice(d.acnet_var) for d in self.vars.values()])
        # ds.read()
        # for k, v in ds.devices:
        #     pass
        # return self

    def shuffle(self, seed: int = 42):
        """ Shuffle variables """
        random.seed(seed)
        keys = list(self.vars.keys())
        random.shuffle(keys)
        self.vars = {k: self.vars[k] for k in keys}
        return self

    def get_dict(self, as_devices=True) -> dict[str, 'KnobVariable']:
        """ Get dictionary of knob variables """
        if as_devices:
            return {v.acnet_var: v.value for v in self.vars.values()}
        else:
            return {v.var: v.value for v in self.vars.values()}

    @staticmethod
    def from_dict(x: dict[str, float]):
        """
        Create new know from dict. Note that no syntax enforcement will be done, both variable
        and acnet names will be set to dict keys.
        :param x: dict of {var: value}
        :return:
        """
        variables = {k: KnobVariable(kind='$', var=k, acnet_var=k, value=v) for k, v in x.items()}
        return Knob(variables=variables)

    def only_keep_shared(self, other: 'Knob') -> 'Knob':
        """
        Remove all variables not shared with another Knob.
        This operation is done inplace.
        :param other: Knob 2
        """
        self.vars = {k: v for (k, v) in self.vars.items() if k in other.vars}
        return self

    def union(self, other: 'Knob') -> 'Knob':
        """
        Returns knob with only variables that are contained in both and their values match.
        This operation is done inplace.
        :param other: Knob 2
        """
        self.vars = {k: v for (k, v) in self.vars.items() if k in other.vars and other.vars[k] == v}
        return self

    def copy(self, new_vars: dict = None) -> 'Knob':
        """
        Make a deep copy of knob, optionally with new variables
        :param new_vars: New variables dict
        """
        knob = Knob(name=self.name)
        if new_vars:
            knob.vars = new_vars
        else:
            knob.vars = {k: v.copy() for (k, v) in self.vars.items()}
        knob.absolute = self.absolute
        knob.verbose = self.verbose
        return knob

    def read_current_state(self, settings: bool = True, verbose: bool = False,
                           split: bool = False
                           ):
        """
        Read current ACNET values of the knob
        :param settings: If true, read setpoint
        :param verbose: Verbose printing
        :param split: Whether to split ACNET reads into several smaller ones
        :return:
        """
        if verbose or self.verbose:
            verbose = True
        if verbose:
            print(f'Reading in knob {self.name} current values')
        devices = [DoubleDevice(d.acnet_var) for d in self.vars.values()]
        # if settings:
        #     for d in devices:
        #         d.drf2.property = DRF_PROPERTY.SETTING
        ds = DoubleDeviceSet(name=self.name,
                             members=devices,
                             settings=settings)
        ds.read(verbose=verbose, split=split)
        tempdict = {k.acnet_var: k for k in self.vars.values()}
        for k, v in ds.devices.items():
            tempdict[k].value = v.value
        return self

    def prune(self, tol: float = 1e-4, verbose: bool = False):
        if verbose or self.verbose:
            verbose = True
        if verbose:
            pruned = {k.acnet_var: k.value for k in self.vars.values() if np.abs(k.value) <= tol}
            print(f'{len(pruned)} pruned:', pruned)
        self.vars = {k.acnet_var: k for k in self.vars.values() if np.abs(k.value) > tol}
        return self

    def is_empty(self):
        return len(self.vars) == 0

    def set(self, verbose: bool = False, split_types: bool = False,
            split: bool = True,
            calculate_physical_currents: bool = False
            ):
        """
        Sets the current knob value in actual machine
        :param verbose:
        :param split_types: Whether to split settings by device type
        :param split: Whether to ask device sets to split settings
        :param calculate_physical_currents: Request to convert skew quads/correctors to coils
        current
        :return:
        """
        from ..iota import run4 as iota, magnets_run4 as iotamags
        if verbose or self.verbose:
            verbose = True
        if not self.absolute:
            raise Exception('Attempt to set relative knob')
        if verbose:
            print(f'Setting knob {self.name}')

        if split_types:
            skews = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values() if
                     d.acnet_var in
                     iota.SKEWQUADS.ALL_I]
            corrV = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values() if
                     d.acnet_var in
                     iota.CORRECTORS.VIRTUAL_V]
            corrH = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values() if
                     d.acnet_var in
                     iota.CORRECTORS.VIRTUAL_H]
            other = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values()
                     if d.acnet_var not in iota.SKEWQUADS.ALL_I
                     and d.acnet_var not in iota.CORRECTORS.COMBINED_VIRTUAL]
            # random.shuffle(dlist3)

            if len(other) >= 2:
                other1, other2 = other[:len(other) // 2], other[len(other) // 2:]
                assert len(other1) + len(other2) == len(other)
            else:
                other1 = other
                other2 = []
            # print()
            # print(skews, corrV, corrH, other1, other2, sep='\n')

            if len(skews) != 0:
                ds = DoubleDeviceSet(name=self.name, members=[d[0] for d in skews])
                ds.set([d[1] for d in skews], verbose=verbose, split=split)
            if len(other1) != 0:
                ds = DoubleDeviceSet(name=self.name, members=[d[0] for d in other1])
                ds.set([d[1] for d in other1], verbose=verbose, split=split)
            if len(corrV) != 0:
                ds = DoubleDeviceSet(name=self.name, members=[d[0] for d in corrV])
                ds.set([d[1] for d in corrV], verbose=verbose, split=split)
            if len(other2) != 0:
                ds = DoubleDeviceSet(name=self.name, members=[d[0] for d in other2])
                ds.set([d[1] for d in other2], verbose=verbose, split=split)
            time.sleep(0.3)
            if len(corrH) != 0:
                ds = DoubleDeviceSet(name=self.name, members=[d[0] for d in corrH])
                ds.set([d[1] for d in corrH], verbose=verbose, split=split)
        else:
            if not calculate_physical_currents:
                # ds = DoubleDeviceSet(name=self.name, members=[d.acnet_var for d in self.vars.values()])
                # ds.set([d.value for d in self.vars.values()], verbose=verbose, split=split)

                # devs1 = [d.acnet_var for d in self.vars.values() if
                #          d.acnet_var in pyIOTA.iota.run2.CORRECTORS.VIRTUAL_H]
                # devs2 = [d.acnet_var for d in self.vars.values() if
                #          d.acnet_var in pyIOTA.iota.run2.CORRECTORS.VIRTUAL_V]
                devs1 = [d.acnet_var for d in self.vars.values() if
                         d.acnet_var in iota.CORRECTORS.COMBINED_COILS_I]
                devs2 = []
                devs3 = [d.acnet_var for d in self.vars.values() if
                         d.acnet_var in iota.SKEWQUADS.ALL_I]
                dev_temp = devs1 + devs2 + devs3
                devs4 = [d.acnet_var for d in self.vars.values() if d.acnet_var not in dev_temp]
                devs = devs1 + devs2 + devs3 + devs4
                ds = DoubleDeviceSet(name=self.name, members=devs)
                acnet_dict = {d.acnet_var: d for d in self.vars.values()}
                ds.set([acnet_dict[d].value for d in devs], verbose=verbose, split=split)
            else:
                devs_h = iota.CORRECTORS.VIRTUAL_H
                devs_v = iota.CORRECTORS.VIRTUAL_V
                devs_s = iota.SKEWQUADS.ALL_I
                coils = iota.CORRECTORS.COMBINED_COILS_I

                def grouper(n, iterable):
                    it = iter(iterable)
                    while True:
                        chunk = tuple(itertools.islice(it, n))
                        if not chunk:
                            return
                        yield chunk

                coil_groups = list(grouper(4, coils))
                assert len(devs_h) == len(devs_v) == len(devs_s) == len(coils) // 4
                devs_to_set = {d.acnet_var: d for d in self.vars.values()}
                for h, v, s, cg in zip(devs_h, devs_v, devs_s, coil_groups):
                    hor = devs_to_set.get(h, None)
                    ver = devs_to_set.get(v, None)
                    skew = devs_to_set.get(s, None)
                    if any([hor, ver, skew]):
                        currents = iotamags.get_combfun_coil_currents(ch=hor, cv=ver, skew=skew)
                        if hor: del devs_to_set[h]
                        if ver: del devs_to_set[v]
                        if skew: del devs_to_set[s]
                        for i, c in enumerate(cg):
                            devs_to_set[c] = currents[i]
                        print(
                                f'Recomputed virtual knobs ({hor}-{ver}-{skew}) into physical ({cg}-{currents})')
                ds = DoubleDeviceSet(name=self.name, members=list(devs_to_set.keys()))
                ds.set([d.value for d in devs_to_set.values()], verbose=verbose, split=split)

    def convert_to_physical_devices(self, copy: bool = True, silent: bool = True):
        """
        Converts any virtual ACNET devices into physical devices. This is so far only used for combined
        function magnets to go from (H,V,Skew) -> (4 currents), which makes settings faster and reduces
        beam losses. #justACNETthings
        :param copy: Whether to return a copy of the knob
        :param silent: Suppress printing logs
        :return:
        """
        from ..iota import run4 as iota, magnets_run4 as iotamags
        devs_h = iota.CORRECTORS.VIRTUAL_H
        devs_v = iota.CORRECTORS.VIRTUAL_V
        devs_s = iota.SKEWQUADS.ALL_I
        coils = iota.CORRECTORS.COMBINED_COILS_I

        # initial_len = len(self.vars)

        def grouper(n, iterable):
            it = iter(iterable)
            while True:
                chunk = tuple(itertools.islice(it, n))
                if not chunk:
                    return
                yield chunk

        coil_groups = list(grouper(4, coils))
        assert len(devs_h) == len(devs_v) == len(devs_s) == len(coils) // 4
        devs_to_set = {d.acnet_var: d for d in self.vars.values()}
        for h, v, s, cg in zip(devs_h, devs_v, devs_s, coil_groups):
            hor = devs_to_set.get(h, None)
            ver = devs_to_set.get(v, None)
            skew = devs_to_set.get(s, None)
            if any([hor, ver, skew]):
                currents = iotamags.get_combfun_coil_currents(ch=hor, cv=ver, skew=skew)
                if hor:
                    del devs_to_set[h]
                if ver:
                    del devs_to_set[v]
                if skew:
                    del devs_to_set[s]
                for i, c in enumerate(cg):
                    devs_to_set[c] = KnobVariable(kind='$', var=c, value=currents[i])
                if not silent:
                    print(
                            f'Virtual knobs ({hor}|{ver}|{skew}) -> physical ({[devs_to_set[c] for c in cg]})')
        if copy:
            knob = self.copy()
            knob.vars = devs_to_set
            return knob
        else:
            self.vars = devs_to_set
            return self

    def __len__(self):
        return len(self.vars)

    def __math(self, other, operation: Callable, opcode: str = ' ', keep_unique_values: bool = True
               ):
        """
        General math operation method, that takes care of relative/absolute knob complexities
        :param other:
        :param operation:
        :param opcode:
        :param keep_unique_values:
        :return:
        """
        knob = self.copy()
        if isinstance(other, Knob):
            set1 = set(self.vars.keys())
            set2 = set(other.vars.keys())

            if keep_unique_values:
                if self.absolute == other.absolute:
                    # Both absolute or relative
                    new_vars = {kv.acnet_var: KnobVariable(kind=kv.kind, var=kv.var, value=kv.value,
                                                           acnet_var=kv.acnet_var
                                                           ) for knob, kv in self.vars.items()}
                    for k, v in other.vars.items():
                        if v.acnet_var in new_vars:
                            new_vars[v.acnet_var].value = operation(new_vars[v.acnet_var].value,
                                                                    v.value)
                        else:
                            # print(f'Added extra var {k}')
                            new_vars[v.acnet_var] = v.copy()
                if (not other.absolute and not set2.issubset(set1)) or (
                        not self.absolute and not set1.issubset(set2)):
                    raise Exception(
                            "Cannot keep unique relative variables when other knob is absolute")
            else:
                keyset = set1.intersection(set2)
                setvars = {k: self.vars[k] for k in keyset}
                new_vars = {kv.acnet_var: KnobVariable(kind=kv.kind, var=kv.var,
                                                       value=operation(kv.value,
                                                                       other.vars[knob].value),
                                                       acnet_var=kv.acnet_var
                                                       ) for knob, kv in setvars}
            knob.name = '(' + self.name + opcode + other.name + ')'
        else:
            new_vars = {kv.acnet_var: KnobVariable(kind=kv.kind, var=kv.var,
                                                   value=operation(kv.value, other),
                                                   acnet_var=kv.acnet_var
                                                   ) for knob, kv in self.vars.items()}
            knob.name = '(' + self.name + opcode + str(other) + ')'
        knob.vars = new_vars  # noqa
        knob.absolute = True if (self.absolute or other.absolute) else False
        return knob

    def __getitem__(self, item):
        return self.vars[item].value

    def __sub__(self, other):
        assert isinstance(other, Knob)
        if self.verbose:
            print(f'Subtracting ({other.name}) from ({self.name}) | ({len(self.vars)} values)')
        return self.__math(other, operator.sub, '-')

    def __add__(self, other):
        # assert isinstance(other, Knob)
        if self.verbose:
            print(f'Adding ({other.name}) to ({self.name}) | ({len(self.vars)} values)')
        return self.__math(other, operator.add, '+')

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        if self.verbose:
            print(f'Diving knob {self.name} by {other} (returning copy)')
        return self.__math(other, operator.truediv, '/')

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        if self.verbose:
            print(f'Multiplying knob {self.name} by {other} (returning copy)')
        return self.__math(other, operator.mul, '*')

    def __str__(self):
        return f'Knob(Abs:{self.absolute}) ({self.name}) at {hex(id(self))}: ({len(self.vars)}) ' \
               f'devices'

    def __repr__(self):
        return self.__str__()


class KnobVariable:
    """
    A single variable knob, with ACNET name storage as well as native name
    """

    def __init__(self, kind: str, var: str, value: float, acnet_var: str = None):
        self.kind = kind
        self.var = var
        self.acnet_var = acnet_var or var.strip('$').replace('_', ':')
        self.value = value

    def copy(self):
        return KnobVariable(self.kind, self.var, self.value, self.acnet_var, )

    def __str__(self):
        # return f'KnobVar ({self.kind})|({self.var})|({self.acnet_var}) = ({self.value}) at {hex(id(self))}'
        return f'KV ({self.acnet_var})=({self.value:+.5f})'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (self.kind == other.kind and self.var == other.var and self.acnet_var ==
                other.acnet_var and self.value == other.value)


def parse_knobs(fpath: Path, verbose: bool = False, randomize: bool = True) -> Dict[str, Knob]:
    """
    Parses knob files in 6DSim format. All values are assumed absolute unless explicitly specified.
    :param fpath: Full knob file path
    :param verbose: If True, print more infor
    :param randomize: If True, final order is randomized
    :return:
    """
    knobs = []
    with open(str(fpath), 'r') as f:
        lines = f.readlines()
        # if verbose: print(f'Parsing {len(lines)} lines')
        assert lines[0] == "KNOBS:\n"
        line_num = 0
        while line_num <= len(lines) - 1:
            l = lines[line_num]
            if l.startswith('Knob'):
                spl = l.split(' ', 2)
                assert len(spl) == 3, spl[0] == 'Knob:'
                assert spl[2].strip().startswith('{')
                name = spl[1]
                # if verbose: print(f'Parsing knob {name}')

                knobvars = []
                knob = Knob(name=name)

                line_num += 1
                s2str = spl[2].strip()
                vals = s2str[1:] if len(s2str) > 1 else ''
                # print(spl, vals)
                if vals.endswith('}'):
                    vals = vals[:-1]
                else:
                    while not lines[line_num].strip().endswith('}'):
                        vals += lines[line_num]
                        line_num += 1
                        if line_num == len(lines):
                            raise Exception('Unclosed bracket found in knob file')
                    if lines[line_num].strip() != '}':
                        vals += lines[line_num].strip()[:-1]
                knob_str_list = vals.strip().replace('\n', '').split(',')
                # print(knob_str_list)
                for k in knob_str_list:
                    ks = k.strip().strip('(').strip(')')
                    kspl = ks.split('|')
                    assert len(kspl) == 3
                    if kspl[0] != '$':
                        raise Exception(f'Unsupported knob type: {kspl}')
                    knobvar = KnobVariable(kind=kspl[0], var=kspl[1], value=float(kspl[2]))
                    knobvars.append(knobvar)
                knobs.append(knob)
                if randomize:
                    random.seed(42)
                    random.shuffle(knobvars)
                knob.vars = {k.acnet_var: k for k in knobvars}
                if verbose:
                    print(f'Parsed knob {name} - {len(knob.vars)} devices')
            else:
                line_num += 1
    return {k.name: k for k in knobs}
