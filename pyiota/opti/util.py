from typing import Union

import numpy as np


#def sene(a: Union[float, np.ndarray], b: Union[float, np.ndarray], tol: float = 1.0e-3):
def sene(a: float, b: float, tol: float = 1.0e-3):
    """ Implements elegant sene - square of error (a-b) if above threshold """
    assert tol > 0
    assert (isinstance(a, float) and isinstance(b, float))
    #assert (isinstance(a, float) and isinstance(b, float)) or (isinstance(a, np.ndarray) and isinstance(b, np.ndarray))
    delta = a - b
    if delta > tol:
        delta -= tol
        return (delta * delta) / (tol * tol)
    elif delta < -tol:
        delta += tol
        return (delta * delta) / (tol * tol)
    else:
        return 0.0


def sene2(a: Union[float, np.ndarray], b: Union[float, np.ndarray], tol: float = 1.0e-3):
    """ Modified sene with no cutoff - smoothly goes to 0 when a == b """
    assert tol > 0
    return ((a - b) / tol) ** 2
