import math

import numpy as np
from matplotlib import pyplot as plt

from underactuated_rocket.state import RocketParams, RocketState

HPI = math.pi/2

def check_no_control_statistics():
    """
    Checks how long it takes a rocket to return to vertical from horizontal.
    """
    n_trials = 3
    trials = []
    rocket_params = RocketParams(1e3)
    rocket_state = RocketState(rocket_params)
    INIT_STATE = rocket_state.get_state()
    for i in range(n_trials):
        print(i)
        rocket_state.set_state(INIT_STATE)
        trajectory = []
        while rocket_state.m_fuel > 0:
            trajectory.append(rocket_state.transition(0))
        trials.append(trajectory)

    return trials


if __name__ == "__main__":
    trials = check_no_control_statistics()

    avg = np.mean(trials, axis=0)
    std = np.std(trials, axis=0)

    fig,ax = plt.subplots()

    ax.plot(avg[:,0], avg[:,1])
    ax.plot(avg[:,0] + std[:,0],  avg[:,1] + std[:,1], c='b', alpha=0.5, linewidth=0.5)
    ax.plot(avg[:,0] + std[:,0],  avg[:,1] - std[:,1], c='b', alpha=0.5, linewidth=0.5)
    ax.plot(avg[:,0] - std[:,0],  avg[:,1] + std[:,1], c='b', alpha=0.5, linewidth=0.5)
    ax.plot(avg[:,0] - std[:,0],  avg[:,1] - std[:,1], c='b', alpha=0.5, linewidth=0.5)
    # ax.fill_betweenx(avg[:,1] + std[:,1], avg[:,0] - std[:,0], avg[:,0] + std[:,0], alpha=0.2)
    # ax.fill_betweenx(avg[:,1] - std[:,1], avg[:,0] - std[:,0], avg[:,0] + std[:,0], alpha=0.2)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    x_upper = avg[:,0] + std[:,0]
    y_upper = avg[:,1] + std[:,1]
    x_lower = avg[:,0] - std[:,0]
    y_lower = avg[:,1] - std[:,1]

    x_fill = np.concatenate([x_upper, x_lower[::-1]])
    y_fill = np.concatenate([y_upper, y_lower[::-1]])
    ax.fill(x_fill, y_fill, 'b')

    plt.show()
