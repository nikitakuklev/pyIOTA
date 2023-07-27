from enum import Enum, auto

from .property import DRF_PROPERTY


class DRF_FIELD(Enum):
    # reading/setting
    RAW = auto()
    PRIMARY = auto()
    VOLTS = auto()
    SCALED = auto()
    COMMON = auto()
    # status
    ALL = auto()
    TEXT = auto()
    EXTENDED_TEXT = auto()
    ON = auto()
    READY = auto()
    REMOTE = auto()
    POSITIVE = auto()
    RAMP = auto()


DEFAULT_FIELD_FOR_PROPERTY = {
    DRF_PROPERTY.READING: DRF_FIELD.SCALED,
    DRF_PROPERTY.SETTING: DRF_FIELD.SCALED,
    DRF_PROPERTY.STATUS: None, #DRF_FIELD.ALL
    DRF_PROPERTY.CONTROL: None,
    DRF_PROPERTY.ANALOG: DRF_FIELD.ALL,
    DRF_PROPERTY.DIGITAL: DRF_FIELD.ALL,
    DRF_PROPERTY.DESCRIPTION: None,
    DRF_PROPERTY.INDEX: None,
    DRF_PROPERTY.LONG_NAME: None,
    DRF_PROPERTY.ALARM_LIST_NAME: None
}

ALLOWED_FIELD_FOR_PROPERTY = {
    DRF_PROPERTY.STATUS: [DRF_FIELD.ALL, DRF_FIELD.TEXT, DRF_FIELD.EXTENDED_TEXT,
                          DRF_FIELD.ON, DRF_FIELD.READY, DRF_FIELD.REMOTE,
                          DRF_FIELD.POSITIVE, DRF_FIELD.RAMP],
    DRF_PROPERTY.CONTROL: []
}


def get_default_field(prop: DRF_PROPERTY):
    return DEFAULT_FIELD_FOR_PROPERTY[prop]
