import logging

import math
import numpy as np

logger = logging.getLogger(__name__)

def addsubtract_wrap(data, delta, minval=0, maxval=2 * np.pi):
    """
    Addition and subtraction of bounded values.
    :param data:
    :param delta:
    :param minval:
    :param maxval:
    :return:
    """
    assert delta < (maxval - minval)
    ret = data + delta
    for i, v in enumerate(ret):
        if v < minval:
            ret[i] += (maxval - minval)
        if v >= maxval:
            ret[i] -= (maxval - minval)
    return ret


def delta_wrap(data1, data2, minval, maxval):
    """
    Smallest distance (forward or backward) between two bounded values, elementwise.
    :param data1: 
    :param data2:
    :param minval:
    :param maxval:
    :return:
    """
    assert len(data1) == len(data2)
    assert not np.any((data1 < minval) | (data1 > maxval) | (data2 < minval) | (data2 > maxval))
    ret = np.zeros(len(data1))
    for i, (v, v2) in enumerate(zip(data1, data2)):
        if v <= v2:
            distance1 = v2 - v
            distance2 = (v - minval) + (maxval - v2)
            ret[i] = distance1 if distance1 <= distance2 else -distance2
        else:
            distance1 = v - v2
            distance2 = (v2 - minval) + (maxval - v)
            ret[i] = -distance1 if distance1 <= distance2 else distance2
    return ret


def forward_distance(data1, data2, minval, maxval):
    """
    Smallest distance forward between two bounded values
    :param data1:
    :param data2:
    :param minval:
    :param maxval:
    :return:
    """
    assert (maxval > data1 > minval) and (maxval > data2 > minval)
    if data1 <= data2:
        return data2 - data1
    else:
        return (data2 - minval) + (maxval - data1)


def remove_integer_part(data, minval, maxval):
    span = maxval - minval
    return np.mod(data, span)


class Wrapper:
    def __init__(self, minval, maxval):
        assert maxval > minval
        self.minval = minval
        self.maxval = maxval

    def add(self, data, delta, minval: float = None, maxval: float = None):
        minval = minval or self.minval
        maxval = maxval or self.maxval
        assert maxval > minval
        span = maxval - minval
        #assert delta < (maxval - minval)
        ret = data + delta
        for i, v in enumerate(ret):
            if v < minval:
                ret[i] += span * math.ceil(abs((v-minval) / span))
                #ret[i] += span
            if v >= maxval:
                #ret[i] -= span
                ret[i] -= span * math.ceil(abs((v-maxval) / span))
        if np.any((ret < minval) | (ret > maxval)):
            raise Exception(f'Out of bounds encountered for add: ({delta}) ({minval}->{maxval}): {data} | {ret}')
        return ret

    def wrap(self, data, minval: float = None, maxval: float = None):
        ret = data.copy()
        minval = minval or self.minval
        maxval = maxval or self.maxval
        span = maxval - minval
        for i, v in enumerate(ret):
            if v < minval:
                ret[i] += span * abs(int(v / span))
            if v >= maxval:
                ret[i] -= span * abs(int(v / span))
        return ret

    def delta(self, data1, data2, minval: float = None, maxval: float = None):
        """
        Smallest distance (forward or backward) between two bounded values, elementwise.
        :param data1:
        :param data2:
        :param minval:
        :param maxval:
        :return:
        """
        assert len(data1) == len(data2)
        minval = minval or self.minval
        maxval = maxval or self.maxval
        if np.any((data1 < minval) | (data1 > maxval) | (data2 < minval) | (data2 > maxval)):
            raise Exception(f'Out of bounds encountered for delta: {data1} | {data2}')
        ret = np.zeros(len(data1))
        for i, (v, v2) in enumerate(zip(data1, data2)):
            if v <= v2:
                distance1 = v2 - v
                distance2 = (v - minval) + (maxval - v2)
                ret[i] = distance1 if distance1 <= distance2 else -distance2
            else:
                distance1 = v - v2
                distance2 = (v2 - minval) + (maxval - v)
                ret[i] = -distance1 if distance1 <= distance2 else distance2
        return ret
