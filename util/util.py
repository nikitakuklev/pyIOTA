from collections.abc import Iterable


def flatten(*args):
    """
    Return generator of deep flat (i.e. recursively flattened) representation of arguments
    :param args: Any number of iterables
    :return: Single 1D list
    """
    return (e for arg in args for e in (flatten(*arg) if isinstance(arg, (tuple, list, Iterable)) else (arg,)))
