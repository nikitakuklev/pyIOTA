import time
from pathlib import Path
import random
import numpy as np

import pyIOTA
from ocelot import Monitor, Vcor, Hcor, Solenoid, SBend, Cavity, Edge, Quadrupole, Drift, Element, Sextupole, Multipole
from pyIOTA.acnet.frontends import DoubleDevice, DoubleDeviceSet
import pyIOTA.iota.run2


def __replace_vars(assign: str, variables: dict):
    """
    Replaces all possible variables with their stored values
    :param assign:
    :param variables:
    :return:
    """
    for k, v in variables.items():
        assign = assign.replace(k, '(' + v + ')', 1)
    return assign


def __resolve_vars(assign: str, variables: dict, recursion_limit: int = 100):
    """
    Recursively resolved variables until none are left
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


def __parse_id_line(line: str, output_dict: dict, resolve_against: dict, verbose: bool = False):
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

    print(f'Parsed OK - {len(lattice_ocelot)} objects, '
          f'{len(correctors_ocelot)} correctors, {len(monitors_ocelot)} monitors')

    info_dict = {'source_file': str(fpath), 'source': '6dsim', 'pc': pc, 'N': N}
    var_dict = {k: KnobVariable(kind='$', var='$'+k, value=v) for k, v in lattice_vars.items()}
    return lattice_ocelot, correctors_ocelot, monitors_ocelot, info_dict, var_dict


def parse_knobs(fpath: Path, verbose: bool = True):
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


class AbstractKnob:
    def __init__(self, name: str):
        self.name = name
        self.verbose = False


class Knob(AbstractKnob):
    def __init__(self, name: str):
        self.vars = {}
        self.absolute = True
        super().__init__(name)

    def make_absolute(self):
        assert not self.absolute
        ds = DoubleDeviceSet(name=self.name, members=[DoubleDevice(d.acnet_var) for d in self.vars.values()])
        ds.readonce()
        for k, v in ds.devices:
            pass

    def get_dict(self, as_devices=False):
        if as_devices:
            return {v.acnet_var: v.value for v in self.vars.values()}
        else:
            return {v.var: v.value for v in self.vars.values()}

    def only_keep_shared(self, other: 'Knob'):
        self.vars = {k: v for (k, v) in self.vars.items() if k in other.vars}
        return self

    def union(self, other: 'Knob'):
        self.vars = {k: v for (k, v) in self.vars.items() if k in other.vars and other.vars[k] == v}
        return self

    def copy(self, devices_only: bool = True):
        k = Knob(name=self.name)
        knobvars = [kv.copy() for kv in self.vars.values()]
        k.vars = {k.var: k for k in knobvars}
        k.absolute = self.absolute
        k.verbose = self.verbose
        return k

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

    def set(self, verbose: bool = False, split: bool = False):
        if verbose or self.verbose:
            verbose = True
        if not self.absolute:
            raise Exception('Attempt to set relative knob')
        if verbose: print(f'Setting knob {self.name}')
        if split:
            skews = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values() if d.acnet_var in
                     pyIOTA.iota.run2.SKEWQUADS.ALL_CURRENTS]
            corrV = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values() if d.acnet_var in
                     pyIOTA.iota.run2.CORRECTORS.VIRTUAL_V]
            corrH = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values() if d.acnet_var in
                     pyIOTA.iota.run2.CORRECTORS.VIRTUAL_H]
            other = [(DoubleDevice(d.acnet_var), d.value) for d in self.vars.values() if d.acnet_var not in
                     pyIOTA.iota.run2.SKEWQUADS.ALL_CURRENTS and d.acnet_var not in pyIOTA.iota.run2.CORRECTORS.COMBINED_VIRTUAL]
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
                ds.set([d[1] for d in skews], verbose=verbose)
            if len(other1) != 0:
                ds = DoubleDeviceSet(name=self.name, members=[d[0] for d in other1])
                ds.set([d[1] for d in other1], verbose=verbose)
            if len(corrV) != 0:
                ds = DoubleDeviceSet(name=self.name, members=[d[0] for d in corrV])
                ds.set([d[1] for d in corrV], verbose=verbose)
            if len(other2) != 0:
                ds = DoubleDeviceSet(name=self.name, members=[d[0] for d in other2])
                ds.set([d[1] for d in other2], verbose=verbose)
            time.sleep(0.3)
            if len(corrH) != 0:
                ds = DoubleDeviceSet(name=self.name, members=[d[0] for d in corrH])
                ds.set([d[1] for d in corrH], verbose=verbose)
        else:
            ds = DoubleDeviceSet(name=self.name, members=[DoubleDevice(d.acnet_var) for d in self.vars.values()])
            ds.set([d.value for d in self.vars.values()], verbose=verbose)

    def __len__(self):
        return len(self.vars)

    def __sub__(self, other):
        assert isinstance(other, Knob)
        if self.verbose: print(f'Subtracting ({other.name}) from ({self.name}) | ({len(self.vars)} values)')
        if not set(self.vars.keys()) == set(other.vars.keys()):
            raise Exception
        knobvars = []
        for k, kv in self.vars.items():
            knobvars.append(KnobVariable(kind=kv.kind, var=kv.var,
                                         value=kv.value - other.vars[k].value))
        k = self.copy()
        k.name = self.name + '-' + other.name
        k.vars = {k.var: k for k in knobvars}
        k.absolute = True if (self.absolute or other.absolute) else False
        return k

    def __add__(self, other):
        assert isinstance(other, Knob)
        if self.verbose: print(f'Adding ({other.name}) to ({self.name}) | ({len(self.vars)} values)')
        if not set(self.vars.keys()) == set(other.vars.keys()):
            raise Exception
        knobvars = []
        for k, kv in self.vars.items():
            knobvars.append(KnobVariable(kind=kv.kind, var=kv.var,
                                         value=kv.value + other.vars[k].value))
        k = self.copy()
        k.name = self.name + '+' + other.name
        k.vars = {k.var: k for k in knobvars}
        k.absolute = True if (self.absolute or other.absolute) else False
        return k

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        if self.verbose: print(f'Diving knob {self.name} by {other} (returning copy)')
        knobvars = []
        for k, kv in self.vars.items():
            knobvars.append(KnobVariable(kind=kv.kind, var=kv.var,
                                         value=kv.value / other))
        k = self.copy()
        k.name = self.name + '/' + str(other)
        k.vars = {k.var: k for k in knobvars}
        return k

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        if self.verbose: print(f'Multiplying knob {self.name} by {other} (returning copy)')
        knobvars = []
        for k, kv in self.vars.items():
            knobvars.append(KnobVariable(kind=kv.kind, var=kv.var,
                                         value=kv.value * other))
        k = self.copy()
        k.name = self.name + '*' + str(other)
        k.vars = {k.var: k for k in knobvars}
        return k

    def __str__(self):
        return f'Knob (abs {self.absolute}) ({self.name}) at {hex(id(self))}:({len(self.vars)}) devices'

    def __repr__(self):
        return self.__str__()


class KnobVariable:
    """
    A single variable in a knob. Should be extended to provide tool-specific methods.
    """

    def __init__(self, kind: str, var: str, value: float):
        self.kind = kind
        self.var = var
        self.acnet_var = var.strip('$').replace('_', ':')
        self.value = value

    def copy(self):
        return KnobVariable(self.kind, self.var, self.value)
