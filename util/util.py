from collections.abc import Iterable
import typing
from typing import List, Generator, Tuple, Union


def flatten(*args: typing.Iterable[Union[Tuple, List, typing.Iterable]]) -> Generator:
    """
    Return generator of deep flat (i.e. recursively flattened) representation of arguments
    :param args: Any number of iterables
    :return: Single 1D list
    """
    return (e for arg in args for e in (flatten(*arg) if isinstance(arg, (tuple, list, Iterable)) else (arg,)))


def flatten_l(*args: typing.Iterable[Union[Tuple, List, typing.Iterable]]) -> List:
    """
    Return a deep flat list
    :param args: Any number of iterables
    :return: Single 1D list
    """
    return [e for arg in args for e in (list(flatten(*arg)) if isinstance(arg, (tuple, list, Iterable)) else (arg,))]


def flatten_unique(*args: typing.Iterable[Union[Tuple, List, typing.Iterable]]) -> List:
    """
    Return a deep flat unique list
    :param args: Any number of iterables
    :return: Single 1D list
    """
    entries = [e for arg in args for e in (list(flatten(*arg)) if isinstance(arg, (tuple, list, Iterable)) else (arg,))]
    return list(set(entries))


def rotate(L: typing.List, idx: int):
    """ Rotate list to the specified index """
    assert 0 <= idx <= len(L) - 1
    return L[idx:] + L[:idx]


def strictly_increasing(L: typing.Iterable) -> bool:
    return all(x < y for x, y in zip(L, L[1:]))


def strictly_decreasing(L: typing.Iterable) -> bool:
    return all(x > y for x, y in zip(L, L[1:]))


def non_increasing(L: typing.Iterable) -> bool:
    return all(x >= y for x, y in zip(L, L[1:]))


def non_decreasing(L: typing.Iterable) -> bool:
    return all(x <= y for x, y in zip(L, L[1:]))


def monotonic(L: typing.Iterable) -> bool:
    return non_increasing(L) or non_decreasing(L)
