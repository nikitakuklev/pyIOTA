from enum import Enum


class DRF_PROPERTY(Enum):
    READING = ':'
    SETTING = '_'
    STATUS = '|'
    CONTROL = '&'
    ANALOG = '@'
    DIGITAL = '$'
    DESCRIPTION = '~'
    INDEX = '^',
    LONG_NAME = '#',
    ALARM_LIST_NAME = '!'


DRF_PROPERTY_NAMES = [el.name for el in DRF_PROPERTY]


def parse_property(raw_string):
    prop_map = {el.name: el for el in DRF_PROPERTY}
    if raw_string not in prop_map:
        raise ValueError(f'Invalid property {raw_string}')
    return prop_map[raw_string]


def get_default_property(raw_string):
    char = raw_string[1]
    if len(raw_string) > 2:
        values = [el.value for el in DRF_PROPERTY]
        if char in values:
            return DRF_PROPERTY(char)
    return DRF_PROPERTY.READING
