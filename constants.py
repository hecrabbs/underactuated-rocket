"""Common constants."""

from typing import Literal

import numpy as np

ACCEL_GRAVITY = 9.81 # m/s^2

# Type hint for column vector
COL_VEC = np.ndarray[tuple[int, Literal[1]], np.float64]
