"""Helper functions."""

from typing import Literal

import numpy as np

# Type hint for column vector
COL_VEC = np.ndarray[tuple[int, Literal[1]], np.float64]

def new_col_vec(*args) -> COL_VEC:
    """Create a new column vector."""
    return np.array([*args])[:, np.newaxis]

def cross_2d(v1: COL_VEC, v2: COL_VEC) -> float:
    """Cross product of 2D column vectors which returns a scalar."""
    return v1[0][0]*v2[1][0] - v1[1][0]*v2[0][0]
