import logging
import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from underactuated_rocket.constants import ACCEL_GRAVITY, I_BASIS, J_BASIS
from underactuated_rocket.helpers import COL_VEC, cross_2d, new_col_vec


@dataclass(frozen=True)
class RocketParams:
    """
    Parameters for a rocket state. Default values loosely based on NASA's Space
    Launch System (SLS).

    https://www3.nasa.gov/sites/default/files/atoms/files/sls_reference_guide_2022_v2_508_0.pdf
    """
    # Mass of control mass C (kg)
    m_C: float

    # Control mass vertical offset (m) (body frame)
    mag_d_0: float=49 # Near top of rocket
    # Control mass horizontal offset (m) (body frame)
    mag_d_1: float=4 # Near radius of rocket

    # Mass of body B (kg)
    m_B: float=1587573.295 # ~3.5e6 lbs
    # Rocket height (m)
    height: float=98.3
    # Rocket radius (m)
    radius: float=4.2
    # Nominal force of thrust (N)
    F_thrust_nominal: float=4*1852e3 + 2*14679e3 # Thrust (engines & boosters)
    # Thrust standard deviation
    sigma_thrust: float=321e3 # Selected 4*1852e3*.13/3
    # Initial mass of fuel (kg)
    m_fuel_init: float=1016046.909 # 5.74e6 - 3.5e6 = 2.24e6 lbs
    # Rate of fuel mass
    m_fuel_rate: float=2116.764 # Burn time = ~480 sec
    # Magnitude of acceleration due to gravity (m/s^2)
    mag_a_g: float=ACCEL_GRAVITY

    # Optional seed if you want randomness to be reproducible
    seed: int=None

    def set_seed(self, n):
        object.__setattr__(self, "seed", n)

class RocketState:
    def __init__(self,
                 params: RocketParams,
                 p: COL_VEC=new_col_vec(0.0, 0.0),
                 v: COL_VEC=new_col_vec(0.0, 0.0),
                 psi: float=0.0,
                 omega: float=0.0,
                 theta_C: float=math.pi/2,
                 omega_C: float=0.0,
                 logger_name: str=__name__):
        """
        Create a new rocket state.

        Parameters
        ----------
        params : RocketParams
            Constant user parameters
        p : COL_VEC
            Body position (m) (inertial frame).
        v : COL_VEC
            Body velocity (m) (inertial frame).
        psi : float
            Rocket yaw (rad) (inertial frame).
        omega : float
            Rocket angular velocity (rad/s) (inertial frame).
        theta_C: float
            Control mass direction (input frame).
        omega_C: float
            Control mass angular velocity (input frame).
        logger_name: str
            Passed to `logging.getLogger()`. Use unique logger name to
            differentiate logs from multiple instances.
        """
        # Rocket parameters
        self.params=params

        # States
        self._p = p
        self._v = v
        self.psi = psi
        self.omega = omega
        self.theta_C = theta_C
        self.omega_C = omega_C
        self.m_fuel = self.params.m_fuel_init

        # Create RNG w/ fixed seed for reproducibility
        self.random = random.Random(self.params.seed)

        # Some constant intermediate variables
        self.d_0_B = new_col_vec(0.0, self.params.mag_d_0)
        self.r_thrust_B = new_col_vec(0.0, -self.params.height/2)

        # Dict for intermediate variables
        self._variables = {}

        # Set up logger
        self.logger = logging.getLogger(logger_name)
        self.logger.debug(f"{self.get_state()=}\n")

    @property
    def p(self): return self._p.copy()

    @p.setter
    def p(self, arr): self._p = arr.copy()

    @property
    def v(self): return self._v.copy()

    @v.setter
    def v(self, arr): self._v = arr.copy()

    @property
    def variables(self): return self._variables.copy()

    @variables.setter
    def variables(self, variables): self._variables = variables.copy()

    def clear_vars(self, alpha_C: float=None):
        """Reset the intermediate variables before transition."""
        if alpha_C is not None:
            # Optionally place input in `self.variables`
            self._variables = {"alpha_C": alpha_C}
        else:
            self._variables = {}

    def get_var(self, fcn: Callable):
        """
        Read result of of `fcn()` from `self.variables`, if already present.
        Otherwise, calculate, store. and return new result.
        """
        result = self._variables.get(fcn.__name__)
        if result is None:
            result = fcn()
            self._variables[fcn.__name__] = result
        return result

    def get_state(self):
        """Get internal state as an array."""
        return np.array([self._p[0][0],
                         self._p[1][0],
                         self._v[0][0],
                         self._v[1][0],
                         self.psi,
                         self.omega,
                         self.theta_C,
                         self.omega_C,
                         self.m_fuel])

    def set_state(self, state_arr):
        """Set internal state from an array."""
        self._p[0][0] = state_arr[0]
        self._p[1][0] = state_arr[1]
        self._v[0][0] = state_arr[2]
        self._v[1][0] = state_arr[3]
        self.psi     = state_arr[4]
        self.omega   = state_arr[5]
        self.theta_C = state_arr[6]
        self.omega_C = state_arr[7]
        self.m_fuel  = state_arr[8]

    def m_total(self):
        """Total mass of rocket."""
        params = self.params
        return params.m_B + params.m_C + self.m_fuel

    def d_1_B(self):
        """
        Distance vector with tail at d_0_B and head at the control mass C
        (3D body frame).
        """
        mag_d_1 = self.params.mag_d_1
        x = mag_d_1*math.cos(self.theta_C)
        y = 0
        z = mag_d_1*math.sin(self.theta_C)
        return new_col_vec(x, y, z)

    def R_IB(self):
        """
        Rotation matrix to get vectors from the body frame in the inertial
        frame.
        """
        psi = self.psi
        return np.array([[math.cos(psi), -math.sin(psi)],
                         [math.sin(psi),  math.cos(psi)]])

    def F_thrust_I(self):
        """Force due to thrust (inertial frame)."""
        # Check if out of fuel
        if self.m_fuel == 0:
            # No thrust.
            return new_col_vec(0.0, 0.0)
        else:
            params = self.params
            noise = self.random.gauss(0.0, params.sigma_thrust)
            # Force vector in body frame.
            F_thrust_B = new_col_vec(0.0, params.F_thrust_nominal + noise)
            # Rotate from B to I.
            return self.get_var(self.R_IB) @ F_thrust_B

    def F_g_I(self):
        """Force due to gravity (interial frame)."""
        m_total = self.get_var(self.m_total)
        return  new_col_vec(0.0, -m_total*self.params.mag_a_g)

    def F_C_I(self):
        """Force due to acceleration of C (inertial frame)."""
        d_1_B = self.get_var(self.d_1_B)
        a_R = -(self.omega_C**2)*d_1_B
        a_T = self._variables["alpha_C"]*np.linalg.cross(d_1_B.T, J_BASIS.T)
        a_C_B = a_R + a_T
        a_C_Bx = a_C_B[0][0]

        # Calculate in body frame then rotate from B to I
        F_C_B = new_col_vec(self.params.m_C*a_C_Bx, 0.0)
        return self.get_var(self.R_IB) @ F_C_B

    def F_total_I(self):
        """Sum of all translational forces (inertial frame)."""
        F_thrust_I = self.get_var(self.F_thrust_I)
        F_g_I = self.get_var(self.F_g_I)
        F_C_I = self.get_var(self.F_C_I)
        return F_thrust_I + F_g_I - F_C_I

    def a_I(self):
        """Rocket acceleration (inertial frame)."""
        F_total_I = self.get_var(self.F_total_I)
        m_total = self.get_var(self.m_total)
        return F_total_I/m_total

    def r_C_B(self):
        """2D Distance vector to the control mass "C" (body frame)."""
        d_1_B = self.get_var(self.d_1_B)
        return self.d_0_B + (d_1_B[0][0]*I_BASIS)[0:2,:]

    def r_CM_B(self):
        """Distance vector to the Center of Mass (CM) (body frame)."""
        r_C_B = self.get_var(self.r_C_B)
        m_total = self.get_var(self.m_total)
        return self.params.m_C * r_C_B[0:2][:] / m_total

    def r_CM_to_thrust_I(self):
        """
        Distance vector from CM to point of thrust (inertial
        frame).
        """
        r_CM_B = self.get_var(self.r_CM_B)
        # Distance vector in body frame.
        r_CM_to_thrust_B = -r_CM_B + self.r_thrust_B
        # Rotate from B to I
        return self.get_var(self.R_IB) @ r_CM_to_thrust_B

    def tau_thrust_I(self):
        """Torque due to thrust (inertial frame)."""
        r_CM_to_thrust_I = self.get_var(self.r_CM_to_thrust_I)
        F_thrust_I = self.get_var(self.F_thrust_I)
        return cross_2d(r_CM_to_thrust_I, F_thrust_I)

    def r_CM_to_d_0_I(self):
        r_CM_B = self.get_var(self.r_CM_B)
        r_CM_to_d_0_B = -r_CM_B + self.d_0_B
        return self.get_var(self.R_IB) @ r_CM_to_d_0_B

    def tau_C_I(self):
        """Torque due to acceleration of C (inertial frame)."""
        r_CM_to_d_0_I = self.get_var(self.r_CM_to_d_0_I)
        F_C_I = self.get_var(self.F_C_I)
        return cross_2d(r_CM_to_d_0_I, -F_C_I)

    def tau_total_I(self):
        """Sum of all torques (inertial frame)."""
        tau_thrust_I = self.get_var(self.tau_thrust_I)
        tau_C_I = self.get_var(self.tau_C_I)
        return tau_thrust_I + tau_C_I

    def I_z(self):
        """Moment of inertia of the rocket, if rotating about z-axis.
        TODO: Add sphere for control mass???"""
        m_total = self.get_var(self.m_total)
        params = self.params
        return (1/12)*m_total*(3*params.radius**2 + params.height**2)

    def alpha_I(self):
        """Angular acceleration (inertial frame)."""
        tau_total_I = self.get_var(self.tau_total_I)
        I_z = self.get_var(self.I_z)
        return tau_total_I / I_z

    def transition(self, alpha_C: float, dt: float=1.0):
        """
        In place state transition.

        Parameters
        ----------
        alpha_C : float
            Control mass angular acceleration (input/action).
        dt : float
            Delta time.
        ulims : tuple[float, float]
            Lower and upper limits for the input alpha_C.
        """
        # Clear old variables
        self.clear_vars(alpha_C)

        # Calculate new variables
        a_I = self.get_var(self.a_I)
        alpha_I = self.get_var(self.alpha_I)

        # Update state
        self.p += self.v*dt
        self.v += a_I*dt**2
        self.psi += self.omega*dt
        self.omega += alpha_I*dt**2
        self.theta_C += self.omega_C*dt
        self.omega_C += alpha_C*dt**2
        # Only update fuel if it isn't empty
        if self.m_fuel > 0:
            self.m_fuel = self.m_fuel - self.params.m_fuel_rate*dt
        # Clip negative fuel to zero
        if self.m_fuel < 0: self.m_fuel = 0

        # Log variables and new state
        self.logger.debug(f"{self.get_state()=}")
        self.logger.debug(f"{self._variables=}")

        return self.get_state()
