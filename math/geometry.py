import numpy as np


def point_line_distance(a: np.ndarray, n: np.ndarray, p: np.ndarray) -> float:
    """
    Minimum distance of point to line
    See https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    :param a: Vector to line origin (from where n starts)
    :param n: Unit direction vector of line
    :param p: Vector to point
    :return: Norm of distance vector
    """
    d = (a - p) - ((a - p) @ n) * n
    return np.linalg.norm(d)
