from time import time

import control as ctl
import control.optimal as opt
import matplotlib.pyplot as plt
import numpy as np

from helpers import new_col_vec
from state import RocketParams, RocketState

t0 = time()

rocket_params = RocketParams(
    m_B=100,
    m_C=10,
    m_fuel=10,
    mag_d_0=0.5,
    mag_d_1=0.25,
    sigma_thrust=0.1,
    F_thrust_nominal=2000,
    r_B_to_thrust_B=new_col_vec(0.0, -0.5)
)

tmp_state = RocketState(rocket_params)
LEN_STATE_SPACE = len(tmp_state.get_state())

def updfcn(t, x, u, params):
    tmp_state.set_state(x)
    tmp_state.transition(u[0])
    return tmp_state.get_state()

sys = ctl.nlsys(updfcn, inputs=["alpha_C"], states=LEN_STATE_SPACE, dt=1)

xf = [10, 1e9, 0, 0, 0, 0, 0, 0]

Q1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0.1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]])
R1 = 0.1
traj_cost = opt.quadratic_cost(sys, Q1, R1, x0=xf)

Q2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]])
R2 = 0
term_cost = opt.quadratic_cost(sys, Q2, R2, x0=xf)

iter_dur = 15 # Compute n second trajectory
use_dur = 5 # Only use first m seconds trajectory

num_iter = 9 # Repeat

total_dur = num_iter*use_dur + 1 # Plus one for initial state

inputs = np.empty((num_iter, iter_dur))
computed_states = np.empty((num_iter, LEN_STATE_SPACE, iter_dur))
actual_states = np.empty((LEN_STATE_SPACE, total_dur))

rocket_state = RocketState(rocket_params)
actual_states[:,0] = rocket_state.get_state()

for i in range(num_iter):
    # Get optimal trajectory
    res = opt.solve_optimal_trajectory(sys,
                                       np.arange(0, iter_dur, 1),
                                       tmp_state.get_state(),
                                       traj_cost,
                                       terminal_cost=term_cost)
    print(f"Iter {i} compute time: {time()-t0}")
    print(f"Success: {res.success}, Cost: {res.cost}")
    print()

    # Update data structs
    inputs[i] = res.inputs[0]
    computed_states[i] = res.states

    # Take optimal trajectory for portion of computed
    for j in range(use_dur):
        idx = i*use_dur + j + 1
        u = res.inputs[0][j]
        rocket_state.transition(u)
        actual_states[:,idx] = rocket_state.get_state()

    # Set solve state to match actual trajectory
    tmp_state.set_state(rocket_state.get_state())

fig,axs = plt.subplots(3,3)
fig.tight_layout()

t_full = np.arange(total_dur)

axs[0,0].set_title("Input (deg/s^2)")
for i, arr in enumerate(inputs):
    t = np.arange(i*use_dur, i*use_dur + iter_dur, 1)
    axs[0,0].plot(t, arr, '*-')

axs[0,1].set_title("OmegaC (deg/s)")
for i, arr in enumerate(computed_states):
    t = np.arange(i*use_dur, i*use_dur + iter_dur, 1)
    axs[0,1].plot(t, arr[-1], '*-', label="exp")
axs[0,1].plot(t_full, actual_states[-1], '.-', c='b', label="act")

axs[0,2].set_title("ThetaC (deg)")
for i, arr in enumerate(computed_states):
    t = np.arange(i*use_dur, i*use_dur + iter_dur, 1)
    axs[0,2].plot(t, np.degrees(arr[-2]), '*-')
axs[0,2].plot(t_full, np.degrees(actual_states[-2]), '.-', c='b')

axs[1,0].set_title("Pos (m)")
for i, arr in enumerate(computed_states):
    t = np.arange(i*use_dur, i*use_dur + iter_dur, 1)
    axs[1,0].plot(arr[0], arr[1], '*-')
axs[1,0].plot(actual_states[0], actual_states[1], '.-', c='b')
axs[1,0].set_xlim(-100,100)
axs[1,0].set_ylim(0,10e3)

axs[1,1].set_title("Vx (m/s)")
for i, arr in enumerate(computed_states):
    t = np.arange(i*use_dur, i*use_dur + iter_dur, 1)
    axs[1,1].plot(t, arr[2], '*-')
axs[1,1].plot(t_full, actual_states[2], '.-', c='b')

axs[1,2].set_title("Vy (m/s)")
for i, arr in enumerate(computed_states):
    t = np.arange(i*use_dur, i*use_dur + iter_dur, 1)
    axs[1,2].plot(t, arr[3], '*-')
axs[1,2].plot(t_full, actual_states[3], '.-')

axs[2,0].set_title("Omega (deg/s)")
for i, arr in enumerate(computed_states):
    t = np.arange(i*use_dur, i*use_dur + iter_dur, 1)
    axs[2,0].plot(t, np.degrees(arr[5]), '*-')
axs[2,0].plot(t_full, np.degrees(actual_states[5]), '.-', c='b')

axs[2,1].set_title("Yaw (deg)")
for i, arr in enumerate(computed_states):
    t = np.arange(i*use_dur, i*use_dur + iter_dur, 1)
    axs[2,1].plot(t, np.degrees(arr[4]), '*-')
axs[2,1].plot(t_full, np.degrees(actual_states[4]), '.-', c='b')

axs[2,2].set_title("C pos 1D body frame")
mag_d_1 = rocket_params.mag_d_1
for i, arr in enumerate(computed_states):
    t = np.arange(i*use_dur, i*use_dur + iter_dur, 1)
    c_1d_exp = mag_d_1*np.cos(arr[-2])
    axs[2,2].plot(c_1d_exp, t, '*-')
c_1d_act = mag_d_1*np.cos(actual_states[-2])
axs[2,2].plot(c_1d_act, t_full, '.-', c='b')
axs[2,2].set_xlabel("C horizontal offset")
axs[2,2].set_ylabel("Time")
axs[2,2].set_xlim(-mag_d_1, mag_d_1)

fig.legend()
plt.show(block=False)
input()
