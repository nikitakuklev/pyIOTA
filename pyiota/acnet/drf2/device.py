import re

from .property import DRF_PROPERTY

PATTERN_NAME = re.compile("(?i)[A-Z0][:?_|&@$~][A-Z0-9_:]{1,62}")


class Device:
    def __init__(self, raw_string, canonical_string):
        self.raw_string = raw_string
        self.canonical_string = canonical_string

    @property
    def canonical(self):
        return self.canonical_string

    def qualified_name(self, prop: DRF_PROPERTY):
        return get_qualified_device(self.raw_string, prop)


def get_qualified_device(device_str: str, prop: DRF_PROPERTY):
    if len(device_str) < 3:
        raise ValueError(f'{device_str} is too short for device')
    assert prop in DRF_PROPERTY
    ext = prop.value
    ld = list(device_str)
    ld[1] = ext
    return ''.join(ld)


def parse_device(raw_string):
    assert raw_string is not None
    match = PATTERN_NAME.match(raw_string)
    if match is None:
        raise ValueError(f'{raw_string} is not a valid device')
    ld = list(raw_string)
    ld[1] = ':'
    dev = Device(raw_string=raw_string, canonical_string=''.join(ld))
    return dev
