"""Common constants."""

from underactuated_rocket.helpers import new_col_vec

ACCEL_GRAVITY = 9.81 # m/s^2

I_BASIS = new_col_vec(1.0, 0.0, 0.0)
J_BASIS = new_col_vec(0.0, 1.0, 0.0)
K_BASIS = new_col_vec(0.0, 0.0, 1.0)
