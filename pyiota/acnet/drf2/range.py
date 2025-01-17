import re
from typing import Literal, Optional

RANGE_RE = re.compile("(\\[(\\d*)(?::(\\d*))?\\])|(\\{(\\d*)(?::(\\d*))?\\})" + "$")

# Integer.MIN_VALUE
MAXIMUM = -2147483648
MAX_UPPER_BOUND = 2147483648


def parse_range(raw_string: Optional[str]):
    if raw_string is None:
        return None
    match = RANGE_RE.match(raw_string)
    if match is None:
        raise ValueError(f'Bad range {raw_string}')
    if match.group(1) is not None:
        s1, s2 = match.group(2), match.group(3)
        if s1 is None and s2 is None:
            # []
            return ARRAY_RANGE(mode='full')
        idx1 = int(s1) if s1 != '' and s1 is not None else None
        idx2 = int(s2) if s2 != '' and s2 is not None else None
        if idx1 is None and idx2 is None:
            # [:]
            return ARRAY_RANGE(mode='full')
        if idx2 is None and ':' not in raw_string:
            #print(f'detected singlet from {raw_string}')
            return ARRAY_RANGE(mode='single', low=idx1, high=idx2)
        return ARRAY_RANGE(mode='std', low=idx1, high=idx2)
    elif match.group(4) is not None:
        s1 = match.group(5)
        s2 = match.group(6)
        s1empty = s1 is None
        s2empty = s2 is None
        if s1empty and s2empty:
            return BYTE_RANGE(mode='full')
        idx1 = int(s1) if s1 != '' and s1 is not None else None
        idx2 = int(s2) if s2 != '' and s2 is not None else None
        if idx1 is None and idx2 is None:
            return BYTE_RANGE(mode='full')
        if idx2 is None and ':' not in raw_string:
            return BYTE_RANGE(mode='single', offset=idx1, length=idx2)
        return BYTE_RANGE(mode='std', offset=idx1, length=idx2)
    else:
        raise Exception('Unrecognized range specifier')


class ARRAY_RANGE:
    def __init__(self,
                 mode: Literal['full', 'std', 'single'] = None,
                 low: Optional[int] = None,
                 high: Optional[int] = None
                 ):
        self.low = low
        self.high = high
        self.mode = mode or ('full' if (low is None and high is None) else 'std')

    def __eq__(self, other):
        return self.low == other.low and self.high == other.high and self.mode == other.mode

    def __str__(self):
        if self.mode == 'full':
            return '[:]'
        elif self.mode == 'single':
            s = f'[{self.low}]'
            return s
        else:
            s = '['
            if self.low is not None:
                s += f'{self.low}'
            s += ':'
            if self.high is not None:
                s += f'{self.high}'
            s += ']'
            return s

    def __repr__(self):
        return f'<ARRAY_RANGE: {str(self)} ({self.mode} mode)>'


class BYTE_RANGE():
    def __init__(self,
                 mode: Literal['full', 'std', 'single'] = None,
                 offset: Optional[int] = None,
                 length: Optional[int] = None
                 ):
        if offset is not None and offset < 0:
            raise ValueError('offset must be non-negative')
        if length is not None and (length != MAXIMUM and length < 0):
            raise ValueError('length must be non-negative')
        if offset is not None and length is not None:
            if length != MAXIMUM and offset + length > MAX_UPPER_BOUND:
                raise ValueError('offset + length must be less than Integer.MAX_VALUE')
            if offset == 0 and length == MAXIMUM:
                assert mode == 'full', 'mode must be full'
        self.offset = offset
        self.length = length
        self.mode = mode  #or ('full' if (low is None and high is None) else 'std')

    def __eq__(self, other):
        return self.offset == other.offset and self.length == other.length and self.mode == other.mode

    def __str__(self):
        if self.mode == 'full':
            return '{:}'
        elif self.mode == 'single':
            s = f'{{{self.offset}}}]'
            return s
        else:
            s = '{'
            if self.offset is not None:
                s += f'{self.offset}'
            s += ':'
            if self.length is not None:
                s += f'{self.length}'
            s += '}'
            return s

    def __repr__(self):
        return f'<BYTE_RANGE: {str(self)} ({self.mode} mode)>'
