import re
from typing import Optional


def parse_event(parse_str: Optional[str]):
    if parse_str is None:
        return None
    assert len(parse_str) > 0
    char = parse_str[0].upper()
    if char == 'U':
        return DefaultEvent(parse_str, char)
    elif char == 'I':
        return ImmediateEvent(parse_str, char)
    elif char in ['P', 'Q']:
        return PeriodicEvent(parse_str, char)
    elif char == 'E':
        return ClockEvent(parse_str, char)
    elif char == 'S':
        return StateEvent(parse_str, char)
    else:
        raise ValueError(f'Invalid event: {parse_str}')


class DRF_EVENT:
    def __init__(self, raw_string: str, mode):
        self.raw_string = raw_string
        self.mode = mode

    def __eq__(self, other):
        return self.raw_string == other.raw_string

    def __repr__(self):
        return f'<DRF_EVENT mode {self.mode}: ({self.raw_string})>'


class DefaultEvent(DRF_EVENT):
    def __init__(self, raw_string='U', mode='U'):
        assert raw_string == mode == 'U'
        super().__init__(raw_string, mode)


class ImmediateEvent(DRF_EVENT):
    def __init__(self, raw_string='I', mode='I'):
        assert raw_string == mode == 'I'
        super().__init__(raw_string, mode)


class PeriodicEvent(DRF_EVENT):
    def __init__(self, raw_string, mode):
        super().__init__(raw_string, mode)
        # This doesn't set things for now
        match = re.match("(?i)(P|Q)(?:,(\\w+)(?:,(F|FALSE|T|TRUE))?)?" + "$", raw_string)
        if match is None:
            raise ValueError(f'Bad periodic event {raw_string}')
        imm = True
        freq = 1000
        if match.group(2) is not None:
            freq = match.group(2)
            if match.group(3) is not None:
                imm = match.group(3)[0].upper() == 'T'
        self.cont = match.group(1)[0] == 'P'
        self.imm = imm
        self.freq = freq


class ClockEvent(DRF_EVENT):
    def __init__(self, raw_string, mode):
        super().__init__(raw_string, mode)
        match = re.match("(?i)E,([0-9A-F]+)(?:,([HSE])(?:,(\\w+))?)?" + "$", raw_string)
        if match is None:
            raise ValueError(f'Bad clock event {raw_string}')
        # Not used
        evt = int(match.group(1), 16)
        delay = 0
        clock_type = 'either'
        if match.group(2) is not None:
            clock_type = match.group(2)
            if match.group(3) is not None:
                delay = match.group(3)
        self.evt = evt
        self.delay = delay
        self.clock_type = clock_type


class StateEvent(DRF_EVENT):
    def __init__(self, raw_string, mode):
        super().__init__(raw_string, mode)
        match = re.match("(?i)S,(\\S+),(\\d+),(\\w+),(=|!=|\\*|>|<|<=|>=)" + "$", raw_string)
        if match is None:
            raise ValueError(f'Bad state event {raw_string}')
