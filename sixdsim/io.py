import time
from pathlib import Path
import random
import numpy as np

import pyIOTA
from pyIOTA.acnet.frontends import DoubleDevice, DoubleDeviceSet
import pyIOTA.iota.run2


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
        for k,v in ds.devices:
            pass

    def get_dict(self, as_devices=False):
        if as_devices:
            return {v.acnet_var: v.value for v in self.vars.values()}
        else:
            return {v.var: v.value for v in self.vars.values()}

    def copy(self, devices_only=True):
        k = Knob(name=self.name)
        knobvars = [kv.copy() for kv in self.vars.values()]
        k.vars = {k.var: k for k in knobvars}
        k.absolute = self.absolute
        k.verbose = self.verbose
        return k

    def read_current_state(self, verbose: bool = False):
        if verbose or self.verbose:
            verbose = True
        if verbose: print(f'Reading in knob {self.name} current values')
        ds = DoubleDeviceSet(name=self.name,
                             members=[DoubleDevice(d.acnet_var) for d in self.vars.values()])
        ds.readonce(settings=True, verbose=verbose)
        tempdict = {k.acnet_var: k for k in self.vars.values()}
        for k, v in ds.devices.items():
            tempdict[k].value = v.value

    def prune(self, tol: float = 1e-4, verbose: bool = False):
        if verbose or self.verbose:
            verbose = True
        if verbose:
            pruned = {k.var: k.value for k in self.vars.values() if np.abs(k.value) <= tol}
            print(f'{len(pruned)} pruned:', pruned)
        self.vars = {k.var: k for k in self.vars.values() if np.abs(k.value) > tol}
        return self.vars

    def set(self, verbose: bool = False):
        if verbose or self.verbose:
            verbose = True
        if not self.absolute:
            raise Exception('Attempt to set relative knob')
        if verbose: print(f'Setting knob {self.name}')
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
        print()
        print(skews, corrV, corrH, other1, other2, sep='\n')

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
        return f'Knob {self.name} at {hex(id(self))}: {len(self.vars)} devices, absolute:{self.absolute}'

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
