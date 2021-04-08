from enum import Enum

import numpy as np

"""
Omega matrices are used to check for symplecticity
"""
omega62 = np.array(
    [[0, 1, 0, 0, 0, 0],
     [-1, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, -1, 0, 0, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, -1, 0]])

omega6 = np.array(
    [[0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [-1, 0, 0, 0, 0, 0],
     [0, -1, 0, 0, 0, 0],
     [0, 0, -1, 0, 0, 0]])

omega4 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]])

omega42 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])

omega2 = np.array([[0, 1], [-1, 0]])

