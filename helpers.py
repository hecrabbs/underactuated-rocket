"""Helper functions."""

import numpy as np

from constants import COL_VEC


def new_col_vec(*args):
    """Create a new column vector."""
    return np.array([*args])[:, np.newaxis]

def cross_2d(v1: COL_VEC, v2: COL_VEC) -> float:
    """Cross product of 2D column vectors which returns a scalar."""
    return v1[0][0]*v2[1][0] - v1[1][0]*v2[0][0]
