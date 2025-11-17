from dataclasses import dataclass, field
from typing import Literal

import numpy as np

COL_VEC = np.ndarray[tuple[Literal[2], Literal[1]], np.float64]

@dataclass
class State:
    # --- Constant user params ---
    _m_total: float

    # Position
    p: COL_VEC = field(default_factory=lambda:
                       np.array([0.0,0.0],dtype=np.float64)[:, np.newaxis])
    # Velocity
    v: COL_VEC = field(default_factory=lambda:
                       np.array([0.0,0.0], dtype=np.float64)[:, np.newaxis])
    # Yaw
    psi: float = 0.0
    # Angular velocity
    omega: float = 0.0
    # Control mass position
    r: float = 0.0
    # Control mass velocity
    r_dot: float = 0.0

    def transition(self, F_total_I, alpha, r_ddot):
        self.p += self.v
        self.v += F_total_I/self._m_total
        self.psi += self.omega
        self.omega += alpha
        self.r += self.r_dot
        self.r_dot += r_ddot
