import re
from typing import Literal, Optional

RANGE_RE = re.compile("(\\[(\\d*)(?::(\\d*))?\\])|(\\{(\\d*)(?::(\\d*))?\\})" + "$")


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
            return DRF_RANGE(mode='full')
        idx1 = int(s1) if s1 != '' and s1 is not None else None
        idx2 = int(s2) if s2 != '' and s2 is not None else None
        if idx1 is None and idx2 is None:
            # [:]
            return DRF_RANGE(mode='full')
        if idx2 is None and ':' not in raw_string:
            #print(f'detected singlet from {raw_string}')
            return DRF_RANGE(mode='single', low=idx1, high=idx2)
        return DRF_RANGE(mode='std', low=idx1, high=idx2)
    else:
        raise Exception('Byte ranges not supported')


class DRF_RANGE:
    def __init__(self,
                 mode: Literal['full', 'std', 'single'] = None,
                 low: Optional[int] = None,
                 high: Optional[int] = None):
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
        return f'<DRF_RANGE: {str(self)} ({self.mode=})>'
