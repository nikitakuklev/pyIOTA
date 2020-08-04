from collections.abc import Iterable


def flatten(*args):
    """
    Return generator of deep flat (i.e. recursively flattened) representation of arguments
    :param args: Any number of iterables
    :return: Single 1D list
    """
    return (e for arg in args for e in (flatten(*arg) if isinstance(arg, (tuple, list, Iterable)) else (arg,)))

def flatten_l(*args):
    """
    Return a deep flat list
    :param args: Any number of iterables
    :return: Single 1D list
    """
    return [e for arg in args for e in (list(flatten(*arg)) if isinstance(arg, (tuple, list, Iterable)) else (arg,))]

def flatten_unique(*args):
    """
    Return a deep flat unique list
    :param args: Any number of iterables
    :return: Single 1D list
    """
    entries = [e for arg in args for e in (list(flatten(*arg)) if isinstance(arg, (tuple, list, Iterable)) else (arg,))]
    return list(set(entries))
