from pathlib import Path


def parse_knobs(fpath: Path):
    knobs = []
    with open(str(fpath), 'r') as f:
        lines = f.readlines()
        print(f'Parsing {len(lines)} lines')
        assert lines[0] == "KNOBS:\n"
        line_num = 0
        while line_num <= len(lines) - 1:
            l = lines[line_num]
            if l.startswith('Knob'):
                spl = l.split(' ', 2)
                assert len(spl) == 3, spl[0] == 'Knob:'
                assert spl[2].strip().startswith('{')
                name = spl[1]
                print(f'Parsing knob {name}')

                knobvars = []
                knob = Knob(name=name)

                line_num += 1
                s2str = spl[2].strip()
                vals = s2str[1:] if len(s2str) > 1 else ''
                print(spl, vals)
                while not lines[line_num].strip().endswith('}'):
                    vals += lines[line_num]
                    line_num += 1
                    if line_num == len(lines):
                        raise Exception('Unclosed bracket found in knob file')
                if lines[line_num].strip() != '}':
                    vals += lines[line_num].strip()[:-1]
                knob_str_list = vals.strip().replace('\n', '').split(',')
                for k in knob_str_list:
                    ks = k.strip().strip('(').strip(')')
                    kspl = ks.split('|')
                    assert len(kspl) == 3
                    if kspl[0] != '$':
                        raise Exception(f'Unsupported knob type: {kspl}')
                    knobvar = KnobVariable(kind=kspl[0], var=kspl[1], value=float(kspl[2]))
                    knobvars.append(knobvar)
                knobs.append(knob)
                knob.vars = knobvars
            else:
                line_num += 1
    return {k.name:k for k in knobs}


class Knob:
    def __init__(self, name: str):
        self.name = name
        self.vars = []

    def get_dict(self, as_devices=False):
        if as_devices:
            return {v.var.strip('$').replace('_',':'): v.value for v in self.vars}
        else:
            return {v.var:v.value for v in self.vars}




class KnobVariable:
    def __init__(self, kind: str, var: str, value: float):
        self.kind = kind
        self.var = var
        self.value = value
