import logging
import math
import random
from dataclasses import dataclass, field

import numpy as np

from constants import ACCEL_GRAVITY, I_BASIS, J_BASIS, K_BASIS
from helpers import COL_VEC, cross_2d, new_col_vec


@dataclass(frozen=True)
class RocketParams:
    m_B: float # Mass of body (kg)
    m_C: float # Mass of control mass (kg)
    m_fuel: float # Initial mass of fuel (kg)
    m_fuel_rate: float # Rate of fuel mass
    height: float # Rocket height (m)
    radius: float # Rocket radius (m)
    mag_d_0: float # Control mass vertical offset (m)
    mag_d_1: float # Control mass horizontal offset (m)
    sigma_thrust: float # Thrust noise standard deviation (variance = sigma^2)
    F_thrust_nominal: float # Nominal force of thrust (i.e. without noise) (N)
    mag_a_g: float=9.81 # Magnitude of acceleration due to gravity (m/s^2)
    seed: int=0 # Seed for random variables so that randomness is reproducible

    def __post_init__(self):
        # C vertical offset. Use object.__setattr__ since frozen==True
        object.__setattr__(self,"d_0_B",
                           new_col_vec(0.0, self.mag_d_0, 0.0))
        object.__setattr__(self, "r_B_to_thrust_B",
                           new_col_vec(0.0, -self.height/2))

@dataclass
class RocketState:
    params: RocketParams

    p: COL_VEC = field(default_factory=lambda: new_col_vec(0.0, 10.0)) # B pos
    v: COL_VEC = field(default_factory=lambda: new_col_vec(0.0, 0.0)) # B vel
    psi: float = 0.0 # Yaw
    omega: float = 0.0 # Angular vel
    theta_C: float = math.pi/2 # Control mass vector direction (body frame)
    omega_C: float = 0.0 # Control mass angular velocity (body frame)
    m_fuel: float = 0.0 # Fuel mass set in post init

    # Use logger name to differentiate multiple instances
    logger_name: str = __name__

    def __post_init__(self):
        object.__setattr__(self, "m_fuel", self.params.m_fuel) # Fuel mass

        # Set random seed for reproducibility
        random.seed(self.params.seed)

        # Set up logger
        self.logger = logging.getLogger(self.logger_name)

        # Log initial state
        self.logger.debug(f"{self=}\n")

    def set_state(self, state_arr):
        self.p[0][0] = state_arr[0]
        self.p[1][0] = state_arr[1]
        self.v[0][0] = state_arr[2]
        self.v[1][0] = state_arr[3]
        self.psi     = state_arr[4]
        self.omega   = state_arr[5]
        self.theta_C = state_arr[6]
        self.omega_C = state_arr[7]
        self.m_fuel  = state_arr[8]

    def get_state(self):
        return np.array([*self.p.T[0],
                         *self.v.T[0],
                         self.psi,
                         self.omega,
                         self.theta_C,
                         self.omega_C,
                         self.m_fuel])

    def _m_total(self):
        """Total mass of rocket."""
        params = self.params
        return params.m_B + params.m_C + self.m_fuel

    def _R_B_to_I(self):
        """Rotation matrix from the body frame to the inertial frame."""
        psi = self.psi
        return np.array([
            [math.cos(psi), -math.sin(psi)],
            [math.sin(psi),  math.cos(psi)]
        ])

    def _F_thrust_I(self, R_B_to_I):
        """Force due to thrust (inertial frame)."""
        params = self.params
        # Check if out of fuel
        if self.m_fuel == 0:
            # No thrust.
            return new_col_vec(0.0, 0.0)
        else:
            noise = random.gauss(0.0, params.sigma_thrust)
            # Force vector in body frame.
            F_thrust_B = new_col_vec(0.0, params.F_thrust_nominal + noise)
            # Rotate from B to I.
            return R_B_to_I @ F_thrust_B

    def _F_g_I(self, m_total):
        """Force due to gravity (interial frame)."""
        return  new_col_vec(0.0, m_total * -self.params.mag_a_g)

    def _F_C_I(self, alpha_C, R_B_to_I):
        """Force due to acceleration of C (inertial frame)."""
        #print("CALCULATING F_C_I:")
        theta_C = self.theta_C
        #print(f"{theta_C=}")
        mag_d_1 = self.params.mag_d_1
        #print(f"{mag_d_1=}")

        # Define radial and tangential unit vectors
        u_R = math.cos(theta_C)*I_BASIS + math.sin(theta_C)*K_BASIS
        #print(f"{u_R=}")
        u_T = -math.sin(theta_C)*I_BASIS + math.cos(theta_C)*K_BASIS
        #print(f"{u_T=}")

        # Accel. vector of C is sum of radial and tangential accel.
        a_R = -(self.omega_C**2)*mag_d_1*u_R
        #print(f"{a_R=}")
        a_T = alpha_C*mag_d_1*u_T
        #print(f"{a_T=}")
        a_C_B = a_R + a_T
        #print(f"{a_C_B=}")
        a_C_Bx = a_C_B[0][0]
        #print(f"{a_C_Bx=}")

        # Calculate in body frame then rotate from B to I
        F_C_B = new_col_vec(self.params.m_C * a_C_Bx, 0.0)
        #print(f"{F_C_B=}")
        return R_B_to_I @ F_C_B

    def _F_total_I(self, F_thrust_I, F_g_I, F_C_I):
        """Sum of all translational forces (inertial frame)."""
        return F_thrust_I + F_g_I - F_C_I

    def _a_I(self, F_total_I, m_total):
        """Rocket acceleration (inertial frame)."""
        return F_total_I/m_total

    def _d_1_B(self):
        """
        Distance vector with tail at d_0_B and head at the controll mass C
        (3D body frame).
        """
        mag_d_1 = self.params.mag_d_1
        x = mag_d_1*math.cos(self.theta_C)
        y = 0
        z = mag_d_1*math.sin(self.theta_C)
        return new_col_vec(x, y, z)

    def _r_C_B(self, d_1_B):
        """2D Distance vector to the control mass "C" (body frame)."""
        return self.params.d_0_B + d_1_B[0][0]*I_BASIS

    def _r_CM_B(self, m_total, r_C_B):
        """Distance vector to the Center of Mass (CM) (body frame)."""
        return self.params.m_C * r_C_B[0:2][:] / m_total

    def _r_CM_to_thrust_I(self, R_B_to_I, r_CM_B):
        """
        Distance vector from CM to point of thrust (inertial
        frame).
        """
        # Distance vector in body frame.
        r_CM_to_thrust_B = -r_CM_B + self.params.r_B_to_thrust_B
        # Rotate from B to I
        return R_B_to_I @ r_CM_to_thrust_B

    def _tau_thrust_I(self, r_CM_to_thrust_I, F_thrust_I):
        """Torque due to thrust (inertial frame)."""
        return cross_2d(r_CM_to_thrust_I, F_thrust_I)

    # def F_r_ddot_on_C_B(self):
    #     """Force on C due to r_ddot in body frame."""
    #     return new_col_vec(self.m_C*self.state.r_ddot)

    # def r_CM_to_C_B(self):
    #     """Vector from center of mass to C in body frame."""
    #     return -self.r_CM_B() + self.d_0 + self.state.r * self.d_1

    # def F_r_ddot_on_B_B(self):
    #     """Force on B due to r_ddot in body frame."""
    #     return new_col_vec(self.m_B*self.state.r_ddot)

    # def r_CM_to_d_0_B(self):
    #     """Vector from center of mass to head of d_0 in body frame."""
    #     return -self.r_CM_B() + self.d_0

    # def tau_r_ddot(self):
    #     """Torque due to accel. of r (same in inertial and body frames)."""
    #     tau_C = cross_2d(self.r_CM_to_C_B(), self.F_r_ddot_on_C_B())
    #     tau_B = cross_2d(self.r_CM_to_d_0_B(), self.F_r_ddot_on_B_B())
    #     return tau_C + tau_B

    def _tau_total_I(self, tau_thrust_I):
        """Sum of all torques (inertial fram).
        TODO: Add torque due to accel on C"""
        return tau_thrust_I

    def _I(self, m_total):
        """Moment of inertia of the rocket.
        TODO"""
        return m_total*self.params.radius**2 / 2
        # length = 1
        # return 0.25*m_total*(0.5)**2 + (1/3)*m_total*(length**2)

    def _alpha_I(self, tau_total_I, I):
        """Angular acceleration (inertial frame)."""
        return tau_total_I / I

    def transition(self, alpha_C: float):
        """
        In place state transition.

        Parameters
        ----------
        alpha_C : float
            Control mass angular acceleration (input/action)
        """
        self.logger.debug(f"{alpha_C=}")

        # Calculate some intermediate values
        m_total = self._m_total()
        self.logger.debug(f"{m_total=}")

        R_B_to_I = self._R_B_to_I()
        self.logger.debug(f"{R_B_to_I=}")

        F_thrust_I = self._F_thrust_I(R_B_to_I)
        self.logger.debug(f"{F_thrust_I=}")

        F_g_I = self._F_g_I(m_total)
        self.logger.debug(f"{F_g_I=}")

        F_C_I = self._F_C_I(alpha_C, R_B_to_I)
        self.logger.debug(f"{F_C_I=}")

        F_total_I = self._F_total_I(F_thrust_I, F_g_I, F_C_I)
        self.logger.debug(f"{F_total_I=}")

        d_1_B = self._d_1_B()
        self.logger.debug(f"{d_1_B=}")
        r_C_B = self._r_C_B(d_1_B)
        self.logger.debug(f"{r_C_B=}")
        r_CM_B = self._r_CM_B(m_total, r_C_B)
        self.logger.debug(f"{r_CM_B=}")
        r_CM_to_thrust_I = self._r_CM_to_thrust_I(R_B_to_I, r_CM_B)
        self.logger.debug(f"{r_CM_to_thrust_I=}")

        tau_thrust_I = self._tau_thrust_I(r_CM_to_thrust_I, F_thrust_I)
        self.logger.debug(f"{tau_thrust_I=}")

        tau_total_I = self._tau_total_I(tau_thrust_I)
        self.logger.debug(f"{tau_total_I=}")

        I = self._I(m_total)
        self.logger.debug(f"{I=}")

        # Update state
        self.p += self.v
        self.v += self._a_I(F_total_I, m_total)
        self.psi += self.omega
        self.omega += self._alpha_I(tau_total_I, I)
        self.theta_C += self.omega_C
        self.omega_C += alpha_C
        self.m_fuel -= self.params.m_fuel_rate

        # ---- prevent y < 0 ----
        if self.p[1, 0] < 0.0:
            self.p[1, 0] = 0.0
            if self.v[1, 0] < 0.0:
                self.v[1, 0] = 0.0

        self.logger.debug(f"{self=}\n")

    def observe(self, sigma_meas: float = 0.05) -> np.ndarray:
        """
        Noisy observation of the full state vector.

        z = x + v,  v ~ N(0, sigma_meas^2 I)

        Returns
        -------
        z : np.ndarray, shape (len(state),)
            Noisy measurement of the true state.
        """
        x_true = self.get_state()
        noise = np.random.normal(0.0, sigma_meas, size=x_true.shape)
        return x_true + noise
