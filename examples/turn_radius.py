import math

import numpy as np

from underactuated_rocket.state import RocketParams, RocketState
from matplotlib import pyplot as plt

HPI = math.pi/2

def check_turn_radius(rocket_state: RocketState):
    """
    Checks how long it takes a rocket to return to vertical from horizontal.
    """
    trajectory = []
    while True:
        s = rocket_state.transition(0)
        trajectory.append(s)

        if s[4] <= 0:
            return True, trajectory
        elif rocket_state.m_fuel == 0:
            return False, trajectory

if __name__ == "__main__":
    m_C_arr = [1, 10, 100, 1e3, 10e3, 100e3]
    trajectory_arr = []

    for m_C in m_C_arr:
        rocket_params = RocketParams(m_C=m_C, sigma_theta_C=0, sigma_thrust=0)
        rocket_state = RocketState(rocket_params, psi=HPI, theta_C=0)

        success, trajectory = check_turn_radius(rocket_state)
        print(success)
        trajectory_arr.append(trajectory)

    fig, ax = plt.subplots()
    for i, trajectory in enumerate(trajectory_arr):
        trajectory = np.vstack(trajectory)
        ax.plot(np.degrees(trajectory[:,4]), label=f"{int(m_C_arr[i])}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Yaw (deg)")
        ax.legend(title="Mass of C (kg)")
    plt.show()
