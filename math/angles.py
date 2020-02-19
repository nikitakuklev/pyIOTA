import numpy as np


def addsubtract_wrap(data, delta, minval=0, maxval=2*np.pi):
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
            ret[i] += (maxval-minval)
        if v >= maxval:
            ret[i] -= (maxval-minval)
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
    assert not np.any((data1<minval) | (data1>maxval) | (data2<minval) | (data2>maxval))
    ret = np.zeros(len(data1))
    for i, (v, v2) in enumerate(zip(data1, data2)):
        if v <= v2:
            distance1 = v2-v
            distance2 = (v-minval)+(maxval-v2)
            ret[i] = distance1 if distance1 <= distance2 else -distance2
        else:
            distance1 = v-v2
            distance2 = (v2-minval)+(maxval-v)
            ret[i] = -distance1 if distance1 <= distance2 else distance2
    return ret
