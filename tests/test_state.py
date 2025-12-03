
import math
import random
from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import (assert_allclose, assert_array_almost_equal_nulp,
                           assert_array_equal)
from pytest import raises

from underactuated_rocket.constants import ACCEL_GRAVITY
from underactuated_rocket.helpers import new_col_vec
from underactuated_rocket.state import RocketParams, RocketState


class TestParams:
    def test_frozen(self):
        params = RocketParams(0,0,0)
        for var in vars(params):
            with raises(FrozenInstanceError):
                params.__setattr__(var, 1)

class TestForces:
    def test_rocket_falls(self):
        # Turn off thrust and see if rocket falls at correct acceleration
        params = RocketParams(0,0,0,
                              m_fuel_rate=0,
                              F_thrust_nominal=0,
                              sigma_thrust=0)
        state = RocketState(params)
        expected = new_col_vec(0.0, -ACCEL_GRAVITY)
        for _ in range(10):
            v1 = state.v
            state.transition(0)
            v2 = state.v
            assert_allclose(v2-v1, expected)

        # Set thrust equal to gravity - 1
        params = RocketParams(0,0,0,
                              m_B=0,
                              m_fuel_init=1,
                              m_fuel_rate=0,
                              F_thrust_nominal=ACCEL_GRAVITY-1,
                              sigma_thrust=0)
        state = RocketState(params)
        expected = new_col_vec(0.0, -1.0)
        for _ in range(10):
            v1 = state.v
            state.transition(0)
            v2 = state.v
            assert_allclose(v2-v1, expected)

        # Set fuel to zero
        params = RocketParams(0,0,0,
                              m_fuel_init=0,
                              m_fuel_rate=1,
                              F_thrust_nominal=ACCEL_GRAVITY+1,
                              sigma_thrust=0)
        state = RocketState(params)
        expected = new_col_vec(0.0, -ACCEL_GRAVITY)
        for _ in range(10):
            v1 = state.v
            state.transition(0)
            v2 = state.v
            assert state.m_fuel == 0
            assert_allclose(v2-v1, expected)

    def test_rocket_hovers(self):
        # Set thrust equal to gravity and see if it hovers
        params = RocketParams(0,0,0,
                              m_B=0,
                              m_fuel_init=1,
                              m_fuel_rate=0,
                              F_thrust_nominal=ACCEL_GRAVITY,
                              sigma_thrust=0)
        state = RocketState(params)
        expected = new_col_vec(0.0, 0.0)
        for _ in range(10):
            v1 = state.v
            state.transition(0)
            v2 = state.v
            assert_allclose(v2-v1, expected)

    def test_rocket_flies(self):
        # Set thrust equal to gravity + 1
        params = RocketParams(0,0,0,
                              m_B=0,
                              m_fuel_init=1,
                              m_fuel_rate=0,
                              F_thrust_nominal=ACCEL_GRAVITY+1,
                              sigma_thrust=0)
        state = RocketState(params)
        expected = new_col_vec(0.0, 1.0)
        for _ in range(10):
            v1 = state.v
            state.transition(0)
            v2 = state.v
            assert_allclose(v2-v1, expected)

    def test_omega(self):
        # Start w/ C on right
        params = RocketParams(1,1,1)
        state = RocketState(params, theta_C=0)
        for _ in range(10):
            state.transition(0)
            omega = state.omega
            assert omega < 0

        # Start w/ C on left
        params = RocketParams(1,1,1)
        state = RocketState(params, theta_C=math.pi)
        for _ in range(10):
            state.transition(0)
            omega = state.omega
            assert omega > 0

    def test_fuel_rate(self):
        params = RocketParams(0,0,0,
                              m_B=0,
                              m_fuel_init=1,
                              m_fuel_rate=0.01,
                              F_thrust_nominal=ACCEL_GRAVITY+1,
                              sigma_thrust=0)
        state = RocketState(params)
        for _ in range(10):
            v1 = state.v
            m1 = state.m_fuel
            state.transition(0)
            v2 = state.v
            m2 = state.m_fuel
            state.transition(0)
            v3 = state.v
            m3 = state.m_fuel
            assert m3 < m2
            assert m2 < m1
            # Check increasing acceleration af m decreases
            assert (v3-v2)[1][0] > (v2-v1)[1][0]


class TestState:
    def test_set_state_get_state(self):
        params = RocketParams(0,0,0)
        state = RocketState(params)
        assert not np.all(state.get_state() == 0)

        new_state = [0]*9
        state.set_state(new_state)
        assert_array_equal(state.get_state(), new_state)

        new_state = list(range(9))
        state.set_state(new_state)
        assert_array_equal(state.get_state(), new_state)

#     def test_m_total(self):
#         m_arr = m_C, m_B, m_fuel = 1,2,3
#         m_total = sum(m_arr)
#         m_fuel_rate = 0.5
#         params = RocketParams(m_C, 0, 0,
#                               m_B=m_B,
#                               m_fuel=m_fuel,
#                               m_fuel_rate=m_fuel_rate)
#         state = RocketState(params)

#         # Check fuel got passed from param to state
#         assert state.m_fuel == m_fuel

#         # Check m_total
#         assert state._m_total() == m_total

#         # Check decrease by m_fuel_rate
#         state.transition(0)
#         assert state.params.m_fuel == m_fuel
#         assert state.m_fuel == m_fuel - m_fuel_rate
#         assert state._m_total() == m_total - m_fuel_rate

#     def test_R_B_to_I(self):
#         params = RocketParams(0,0,0)
#         state = RocketState(params, psi=math.radians(-45))
#         v_B = new_col_vec(0,1)
#         v_I = new_col_vec(math.sqrt(2)/2, math.sqrt(2)/2)
#         assert_array_almost_equal_nulp(state._R_B_to_I() @ v_B, v_I)

#     def test_F_thrust_I(self):
#         # w/o noise
#         F_thrust_nominal = 1
#         params = RocketParams(0,0,0,
#                               F_thrust_nominal=F_thrust_nominal,
#                               sigma_thrust=0)
#         state = RocketState(params, psi=math.radians(-45))
#         F_thrust_I = new_col_vec(F_thrust_nominal*math.sqrt(2)/2,
#                                  F_thrust_nominal*math.sqrt(2)/2)
#         assert_array_almost_equal_nulp(state._F_thrust_I(state._R_B_to_I()),
#                                        F_thrust_I)

#         # w/ noise
#         sigma_thrust = 0.1
#         seed = 123
#         rand = random.Random(seed)
#         thrust_noise = rand.gauss(0, sigma_thrust)

#         params = RocketParams(0,0,0,
#                               F_thrust_nominal=1,
#                               sigma_thrust=sigma_thrust,
#                               seed=seed)
#         state = RocketState(params, psi=math.radians(60))
#         F_thrust_mag = F_thrust_nominal + thrust_noise
#         F_thrust_I = new_col_vec(F_thrust_mag*-math.sqrt(3)/2,
#                                  F_thrust_mag*1/2)
#         assert_array_almost_equal_nulp(state._F_thrust_I(state._R_B_to_I()),
#                                        F_thrust_I)

#         # w/o fuel
#         params = RocketParams(0,0,0, m_fuel=0)
#         state = RocketState(params)
#         assert_array_almost_equal_nulp(state._F_thrust_I(state._R_B_to_I()),
#                                        new_col_vec(0,0))

#     def test_F_g_I(self):
#         F_g_I = new_col_vec(0, -ACCEL_GRAVITY)
#         m_total = 1
#         params = RocketParams(0,0,0, m_B=m_total, m_fuel=0)
#         state = RocketState(params)
#         assert_array_equal(state._F_g_I(m_total), F_g_I)

#         state.psi = math.radians(-45)
#         assert_array_equal(state._F_g_I(m_total), F_g_I)

#     def test_F_C_I(self):
#         m_C, m_B = 1,1
#         mag_d_1 = 10
#         theta_C = math.radians(90)
#         alpha_C = math.radians(1)
#         params = RocketParams(m_C,0,mag_d_1, m_B=m_B, m_fuel=0)
#         state = RocketState(params, theta_C=theta_C)

#         # Starting from rest, so only expect tangential acceleration.
#         expected_F_C_I = new_col_vec(-alpha_C*mag_d_1, 0)
#         actual_F_C_I = state._F_C_I(state._d_1_B(), alpha_C, state._R_B_to_I())
#         assert_allclose(actual_F_C_I, expected_F_C_I)

#         # Angular velocity but no accel, expect only radial/centripetal accel.
#         alpha_C = 0
#         omega_C = math.radians(1)
#         state.omega_C = omega_C
#         expected_F_C_I = new_col_vec(0.0, 0.0)
#         actual_F_C_I = state._F_C_I(state._d_1_B(), alpha_C, state._R_B_to_I())
#         assert_allclose(actual_F_C_I, expected_F_C_I, atol=1e-18)

#         state.theta_C = math.radians(180)
#         expected_F_C_I = new_col_vec(omega_C**2*mag_d_1, 0.0)
#         actual_F_C_I = state._F_C_I(state._d_1_B(), alpha_C, state._R_B_to_I())
#         assert_allclose(actual_F_C_I, expected_F_C_I, atol=1e-18)

#     def test_F_total_I(self):
#         # Make sure Fc is subtracted.
#         Fthr = new_col_vec(0,1)
#         Fg = new_col_vec(0,-1)
#         Fc = new_col_vec(1,0)
#         params = RocketParams(0,0,0)
#         state = RocketState(params)
#         assert_array_equal(state._F_total_I(Fthr, Fg, Fc), new_col_vec(-1,0))
