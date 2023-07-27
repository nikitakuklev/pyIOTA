from enum import Enum, auto


class DRF_EXTRA(Enum):
    FTP = auto()


def parse_extra(raw_string):
    prop_map = {el.name: el for el in DRF_EXTRA}
    if raw_string not in prop_map:
        raise ValueError(f'Invalid extra {raw_string}')
    return prop_map[raw_string]
