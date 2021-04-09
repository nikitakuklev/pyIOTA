__all__ = ['Knob', 'KnobVariable', 'parse_knobs', 'parse_lattice']

import itertools
import logging
import operator
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from ocelot import Monitor, Vcor, Hcor, Solenoid, SBend, Cavity, Edge,\
    Quadrupole, Drift, Element, Sextupole, Multipole
from pyIOTA.acnet.frontends import DoubleDevice, DoubleDeviceSet
import pyIOTA.iota.run2

logger = logging.getLogger(__name__)

# Static methods to parse files from 6Dsim format


def __replace_vars(assign: str, variables: Dict) -> str:
    """
    Replaces all variables with their stored values, adding brackets to maintain eval order.
    :param assign:
    :param variables:
    :return: Processed string
    """
    for k, v in variables.items():
        assign = assign.replace(k, '(' + v + ')', 1)
    return assign


def __resolve_vars(assign: str, variables: Dict, recursion_limit: int = 100) -> str:
    """
    Recursively resolves variables and replaces with values, until none are left
    :param assign:
    :param variables:
    :return:
    """
    i = 0
    # print(f'Resolve start: {assign}')
    while '$' in assign:
        assign = __replace_vars(assign, variables)
        # print(f'Resolve iteration {i}: {assign}')
        i += 1
        if i > recursion_limit:
            raise Exception(f'Unable to resolve line {assign} fully against definitions {variables}')
    return assign


def __parse_id_line(line: str, output_dict: Dict, resolve_against: Dict, verbose: bool = False) -> None:
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
        if verbose: print(f'Parsing parameters ({pars})')
        for (k, v) in zip(pars[::2], pars[1::2]):
            if verbose: print(f'Property ({k}) is ({v})')
            try:
                vr = eval(__resolve_vars(v, resolve_against))
            except Exception as e:
                # if verbose: print(f'Failed to evaluate, using as string ({k})-({v}) : {e})')
                vr = v
            if verbose: print(f'Property ({k}) resolves to ({vr})')
            pars_dict[k] = vr
    if verbose: print(f'Element ({name}) resolved to ({element})({pars_dict})')
    output_dict[name] = (element, pars_dict)


def parse_lattice(fpath: Path, verbose: bool = False):
    """
    Parses 6Dsim lattice into native OCELOT object, preserving all compatible elements. All variables
    are resolved to their final numeric values, and evaluated - this is an UNSAFE operation for unstrusted
    input, so make sure your lattice files are not corrupted.
    :param fpath: Full file path to parse
    :param verbose:
    :return: lattice_ocelot, correctors_ocelot, monitors_ocelot, info_dict, var_dict - latter 2 are dicts
    of parsing information and of all present KnobVariables
    """
    # returned objects
    lattice_list = []
    correctors_ocelot = []
    monitors_ocelot = []
    # internal vars
    keywords = {'INFO:': 1, 'ELEMENTS:': 2, 'CORRECTORS:': 3, 'MONITORS:': 4, 'LATTICE:': 5, 'END': 6}
    variables = {'$PI': str(np.pi)}
    lattice_vars = {}
    elements = {}
    correctors = {}
    monitors = {}
    lattice_str = ''
    # Other parameters
    pc = None
    N = None

    with open(str(fpath), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('//') or line == '':
                continue

            if line in keywords:
                if verbose: print('MODE:', line)
                mode = keywords[line]
                if mode == keywords['END']:
                    if verbose: print('DONE')
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
                    assign = __resolve_vars(assign, variables)
                    variables[var] = assign
                    # print('Variable ({}) resolved to ({})'.format(var, assign))
                    # print(variables)
                    value = eval(assign)
                    if verbose: print(f'Variable ({var}) evaluated to ({value})')
                    lattice_vars[var.strip('$')] = value
                    continue
            elif mode == keywords['ELEMENTS:']:
                __parse_id_line(line, elements, variables, verbose)
                continue
            elif mode == keywords['CORRECTORS:']:
                __parse_id_line(line, correctors, variables, verbose)
                continue
            elif mode == keywords['MONITORS:']:
                __parse_id_line(line, monitors, variables, verbose)
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
            raise Exception(f'Relative position of {el} does not have referenced element {relative_to}')
    for el, (eltype, props) in monitors.items():
        assert eltype == 'BPM'
        relative_to = props.get('El', None)
        if relative_to is None or relative_to not in elements:
            raise Exception(f'Relative position of {el} does not have referenced element {relative_to}')
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
               'RFCorrector': 'RF'}
    field_to_gradient_factor = (pc / 1000 / 0.299792458) / 10.0  # kG*m

    def gradient_scale(g):
        return g / field_to_gradient_factor

    def length_scale(l):
        return l / 100.0

    shared_parameter_map = {'L': ('l', length_scale)}  # apply function to these variables

    for item in lattice:
        (el_type, props) = elements[item]
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
        if type_mapped == Drift:
            oel = Drift(**shared_kwargs)
        elif type_mapped == SBend:
            oel = SBend(angle=props['L'] / ((pc / 0.299792458) / props['Hy']),
                        k1=props['G'] / field_to_gradient_factor,
                        **shared_kwargs)
        elif type_mapped == Edge:
            oel = Edge(gap=props['poleGap'] / 100,  # full gap, not half
                       fint=props['fringeK'],
                       **shared_kwargs)
        elif type_mapped == Quadrupole:
            if el_type == 'SQuad':
                oel = Quadrupole(k1=props['G'] / field_to_gradient_factor,
                                 tilt=np.pi / 4,
                                 **shared_kwargs)
            else:
                oel = Quadrupole(k1=props['G'] / field_to_gradient_factor,
                                 **shared_kwargs)
        elif type_mapped == Multipole:
            if 'M2N' in props and len(props) <= 2:
                # Sextupole (length and M2N props)
                oel = Sextupole(k2=float(props['M2N']) / field_to_gradient_factor, **shared_kwargs)
            else:
                raise Exception(f"Multipole that is not a sextupole detected ({item}|{el_type}|{type_mapped}|{props})")
        elif type_mapped == Solenoid:
            oel = Drift(**shared_kwargs)
        elif type_mapped == Cavity:
            oel = Cavity(v=props['U'],
                         freq=props['F'],
                         **shared_kwargs)
        else:
            raise Exception(f'Empty ocelot object produced converting ({item}|{el_type}|{type_mapped}|{props})')
        lattice_list.append(oel)

    # Dipole edge combination
    lattice_ocelot = lattice_list.copy()
    assert len(lattice) == len(lattice_list)
    edge_count = 0
    for i, (k, v) in enumerate(zip(lattice, lattice_list)):
        if isinstance(v, Edge):
            edge_count += 1
        if isinstance(v, SBend):
            e1 = lattice_list[i - 1]
            e2 = lattice_list[i + 1]
            if not isinstance(e1, Edge) or not isinstance(e2, Edge):
                raise Exception(f'Found sector bend {k} without edge elements - this is not allowed')
            assert e1.gap == e2.gap
            assert e1.fint == e2.fint
            v.fint = e1.fint
            v.fintx = e2.fint
            v.gap = e1.gap
            lattice_ocelot.remove(e1)
            lattice_ocelot.remove(e2)
            if verbose: print(f'Integrated edges for dipole {k}')
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
            raise Exception(f'Empty ocelot object produced converting ({item}|{el_type}|{type_mapped}|{props})')
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

    logger.info(f'Parsed {len(lattice_ocelot)} objects, {len(correctors_ocelot)} correctors, {len(monitors_ocelot)} monitors')
    #print(f'Parsed OK - {len(lattice_ocelot)} objects, '
    #      f'{len(correctors_ocelot)} correctors, {len(monitors_ocelot)} monitors')

    info_dict = {'source_file': str(fpath), 'source': '6dsim', 'pc': pc, 'N': N}
    var_dict = {k: KnobVariable(kind='$', var='$' + k, value=v) for k, v in lattice_vars.items()}
    return lattice_ocelot, correctors_ocelot, monitors_ocelot, info_dict, var_dict


def parse_knobs(fpath: Path, verbose: bool = False) -> Dict:
    """
    Parses knob files in 6DSim format. All values are assumed absolute unless explicitly specified.
    :param fpath: Full knob file path
    :param verbose:
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
                knob.vars = {k.var: k for k in knobvars}
                if verbose: print(f'Parsed knob {name} - {len(knob.vars)} devices')
            else:
                line_num += 1
    return {k.name: k for k in knobs}


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
                matr.append(np.array(list(filter(None, l.strip().split(' ')))).astype(np.float))
                if j == 5:
                    tmaps[to] = (src, np.stack(matr));  # print(dataarr)
        tmaps['start'] = ('end', np.identity(6))
    return tmaps


class AbstractKnob:
    """
    Superclass of all knobs, which are collections of KnobVariables representing setpoints of devices
    """
    def __init__(self, name: str):
        self.name = name
        self.verbose = False


class Knob(AbstractKnob):
    """
    Experimental, ACNET-based implementation of a knob
    """
    def __init__(self, name: str, variables: dict = None):
        self.vars = variables or {}
        self.absolute = True
        super().__init__(name)

    def make_absolute(self):
        raise Exception("Not implemented yet")
        assert not self.absolute
        ds = DoubleDeviceSet(name=self.name, members=[DoubleDevice(d.acnet_var) for d in self.vars.values()])
        ds.readonce()
        for k, v in ds.devices:
            pass
        return self

    def get_dict(self, as_devices=False):
        if as_devices:
            return {v.acnet_var: v.value for v in self.vars.values()}
        else:
            return {v.var: v.value for v in self.vars.values()}

    def only_keep_shared(self, other: 'Knob'):
        self.vars = {k: v for (k, v) in self.vars.items() if k in other.vars}
        return self

    def union(self, other: 'Knob'):
        """
        Returns knob with only variables contained in both and their values match
        :param other:
        :return:
        """
        self.vars = {k: v for (k, v) in self.vars.items() if k in other.vars and other.vars[k] == v}
        return self

    def copy(self, new_vars: dict = None):
        """
        Make a deep copy of knob, optionally with new knobvaribles
        :param new_vars:
        :return:
        """
        knob = Knob(name=self.name)
        if new_vars:
            knob.vars = new_vars
        else:
            knob.vars = {k.var: k.copy() for k in self.vars.values()}
        knob.absolute = self.absolute
        knob.verbose = self.verbose
        return knob

    def read_current_state(self, settings: bool = True, verbose: bool = False):
        if verbose or self.verbose:
            verbose = True
        if verbose: print(f'Reading in knob {self.name} current values')
        ds = DoubleDeviceSet(name=self.name,
                             members=[DoubleDevice(d.acnet_var) for d in self.vars.values()])
        ds.readonce(settings=settings, verbose=verbose)
        tempdict = {k.acnet_var: k for k in self.vars.values()}
        for k, v in ds.devices.items():
            tempdict[k].value = v.value
        return self

    def prune(self, tol: float = 1e-4, verbose: bool = False):
        if verbose or self.verbose:
            verbose = True
        if verbose:
            pruned = {k.var: k.value for k in self.vars.values() if np.abs(k.value) <= tol}
            print(f'{len(pruned)} pruned:', pruned)
        self.vars = {k.var: k for k in self.vars.values() if np.abs(k.value) > tol}
        return self

    def is_empty(self):
        return len(self.vars) == 0

    def set(self, verbose: bool = False, split_types: bool = False, split: bool = True,
            calculate_physical_currents: bool = False):
        """
        Sets the current knob value in actual machine
        :param verbose:
        :param split_types: Whether to split settings by device type
        :param split:
        :param calculate_physical_currents:
        :return:
        """
        if verbose or self.verbose:
            verbose = True
        if not self.absolute:
            raise Exception('Attempt to set relative knob')
        if verbose: print(f'Setting knob {self.name}')

        if split_types:
            skews = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values() if d.acnet_var in
                     pyIOTA.iota.run2.SKEWQUADS.ALL_CURRENTS]
            corrV = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values() if d.acnet_var in
                     pyIOTA.iota.run2.CORRECTORS.VIRTUAL_V]
            corrH = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values() if d.acnet_var in
                     pyIOTA.iota.run2.CORRECTORS.VIRTUAL_H]
            other = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values()
                     if d.acnet_var not in pyIOTA.iota.run2.SKEWQUADS.ALL_CURRENTS
                     and d.acnet_var not in pyIOTA.iota.run2.CORRECTORS.COMBINED_VIRTUAL]
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
                         d.acnet_var in pyIOTA.iota.run2.CORRECTORS.COMBINED_COILS_I]
                devs2 = []
                devs3 = [d.acnet_var for d in self.vars.values() if
                         d.acnet_var in pyIOTA.iota.run2.SKEWQUADS.ALL_CURRENTS]
                dev_temp = devs1 + devs2 + devs3
                devs4 = [d.acnet_var for d in self.vars.values() if d.acnet_var not in dev_temp]
                devs = devs1 + devs2 + devs3 + devs4
                ds = DoubleDeviceSet(name=self.name, members=devs)
                acnet_dict = {d.acnet_var: d for d in self.vars.values()}
                ds.set([acnet_dict[d].value for d in devs], verbose=verbose, split=split)
            else:
                devs_h = pyIOTA.iota.CORRECTORS.VIRTUAL_H
                devs_v = pyIOTA.iota.CORRECTORS.VIRTUAL_V
                devs_s = pyIOTA.iota.run2.SKEWQUADS.ALL_CURRENTS
                coils = pyIOTA.iota.run2.CORRECTORS.COMBINED_COILS_I

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
                        currents = pyIOTA.iota.magnets.get_combfun_coil_currents(ch=hor, cv=ver, skew=skew)
                        if hor: del devs_to_set[h]
                        if ver: del devs_to_set[v]
                        if skew: del devs_to_set[s]
                        for i, c in enumerate(cg):
                            devs_to_set[c] = currents[i]
                        print(f'Recomputed virtual knobs ({hor}-{ver}-{skew}) into physical ({cg}-{currents})')
                ds = DoubleDeviceSet(name=self.name, members=list(devs_to_set.keys()))
                ds.set([d.value for d in devs_to_set.values()], verbose=verbose, split=split)

    def convert_to_physical_devices(self, copy: bool = True):
        """
        Converts any virtual ACNET devices into physical devices. This is so far only used for combined
        function magnets to go from (H,V,Skew) -> (4 currents), which makes settings faster and reduces
        beam losses. #justACNETthings
        :param copy: Whether to return a copy of the knob
        :return:
        """
        devs_h = pyIOTA.iota.CORRECTORS.VIRTUAL_H
        devs_v = pyIOTA.iota.CORRECTORS.VIRTUAL_V
        devs_s = pyIOTA.iota.run2.SKEWQUADS.ALL_CURRENTS
        coils = pyIOTA.iota.run2.CORRECTORS.COMBINED_COILS_I
        #initial_len = len(self.vars)

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
                currents = pyIOTA.iota.magnets.get_combfun_coil_currents(ch=hor, cv=ver, skew=skew)
                if hor: del devs_to_set[h]
                if ver: del devs_to_set[v]
                if skew: del devs_to_set[s]
                for i, c in enumerate(cg):
                    devs_to_set[c] = KnobVariable(kind='$', var=c, value=currents[i])
                print(f'Virtual knobs ({hor}|{ver}|{skew}) -> physical ({[devs_to_set[c] for c in cg]})')
        if copy:
            knob = self.copy()
            knob.vars = devs_to_set
            return knob
        else:
            self.vars = devs_to_set
            return self

    def __len__(self):
        return len(self.vars)

    def __math(self, other, operation: Callable, opcode: str = ' ', keep_unique_values: bool = True):
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
                    new_vars = {kv.var: KnobVariable(kind=kv.kind, var=kv.var,
                                                     value=kv.value
                                                     ) for knob, kv in self.vars.items()}
                    for k, v in other.vars.items():
                        if k in new_vars:
                            new_vars[k].value = operation(new_vars[k].value, v.value)
                        else:
                            new_vars[k] = v.copy()
                if (not other.absolute and not set2.issubset(set1)) or (not self.absolute and not set1.issubset(set2)):
                    raise Exception("Cannot keep unique relative variables when other knob is absolute")
            else:
                keyset = set1.intersection(set2)
                setvars = {k: self.vars[k] for k in keyset}
                new_vars = {kv.var: KnobVariable(kind=kv.kind, var=kv.var,
                                                 value=operation(kv.value, other.vars[knob].value)
                                                 ) for knob, kv in setvars}
            knob.name = '(' + self.name + opcode + other.name + ')'
        else:
            new_vars = {kv.var: KnobVariable(kind=kv.kind, var=kv.var,
                                             value=operation(kv.value, other)
                                             ) for knob, kv in self.vars.items()}
            knob.name = '(' + self.name + opcode + str(other) + ')'
        knob.vars = new_vars
        knob.absolute = True if (self.absolute or other.absolute) else False
        return knob

    def __sub__(self, other):
        assert isinstance(other, Knob)
        if self.verbose: print(f'Subtracting ({other.name}) from ({self.name}) | ({len(self.vars)} values)')
        return self.__math(other, operator.sub, '-')

    def __add__(self, other):
        assert isinstance(other, Knob)
        if self.verbose: print(f'Adding ({other.name}) to ({self.name}) | ({len(self.vars)} values)')
        return self.__math(other, operator.add, '+')

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        if self.verbose: print(f'Diving knob {self.name} by {other} (returning copy)')
        return self.__math(other, operator.truediv, '/')

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        if self.verbose: print(f'Multiplying knob {self.name} by {other} (returning copy)')
        return self.__math(other, operator.mul, '*')

    def __str__(self):

        return f'Knob(A:{self.absolute}) ({self.name}) at {hex(id(self))}: ({len(self.vars)}) devices'

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
