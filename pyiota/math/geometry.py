import numpy as np
from scipy.spatial.transform import Rotation


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


def cart2pol(x: float, y: float):
    """ (x,y) -> (r,θ) """
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return r, theta


def pol2cart(r: float, theta: float):
    """ (r,θ) -> (x,y) """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


class Vector3D:

    @staticmethod
    def rotate_z(v: np.ndarray, angle: float):
        """ Rotate 3-vector around external z """
        r = Rotation.from_euler('z', angle)
        return r.apply(v)

    @staticmethod
    def norm(v: np.ndarray):
        """ Vector norm """
        return np.linalg.norm(v)

    @staticmethod
    def normalize(v: np.ndarray):
        """ Normalize vector """
        return v / Vector3D.norm(v)

