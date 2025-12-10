import numpy as np
from matplotlib import pyplot as plt


def plot_results(rocket_params,
                 inputs,
                 states,
                 control_horizon=None,
                 prediction_horizon=None,
                 predicted_inputs=None,
                 predicted_states=None,
                 predicted_cost=None,
                 block=True):

    fig,axs = plt.subplots(3,4)
    fuel_empty = states[-1] == 0
    t_arr = np.arange(states.shape[1])
    alpha = 0.25

    def predicted_t_arr(i):
        start = i*control_horizon
        stop = start + prediction_horizon
        return np.arange(start, stop, 1)

    ax = axs[0,0]
    ax.set_title("Input (deg/s^2)")

    if predicted_inputs is not None:
        for i, arr in enumerate(predicted_inputs):
            ax.plot(predicted_t_arr(i),
                    np.degrees(arr), '.-', alpha=alpha, label="predicted")

    ax.plot(t_arr[:-1], np.degrees(inputs), '*-', c='b', label="actual")


    ax = axs[0,1]
    ax.set_title("C horizontal offset B frame (m)")
    mag_d_1 = rocket_params.mag_d_1

    if predicted_states is not None:
        for i, arr in enumerate(predicted_states):
            ax.plot(mag_d_1*np.cos(arr[-3]),
                    predicted_t_arr(i), '.-', alpha=alpha)

    ax.plot(mag_d_1*np.cos(states[-3]), t_arr, '*-', c='b')
    ax.set_xlim(-mag_d_1, mag_d_1)


    ax = axs[0,2]
    ax.set_title("Pos (m)")

    if predicted_states is not None:
        for arr in predicted_states:
            ax.plot(arr[0], arr[1], '.-', alpha=alpha)

    ax.plot(np.where(~fuel_empty, states[0], np.nan),
            np.where(~fuel_empty, states[1], np.nan),'*-', c='b')

    ax.plot(np.where(fuel_empty, states[0], np.nan),
            np.where(fuel_empty, states[1], np.nan),'*-', c='r')

    ax = axs[0,3]
    ax.set_title("Cost")
    if predicted_cost is not None:
        ax.plot(t_arr[:-1],
                np.repeat(predicted_cost, control_horizon), '*-', c='b')



    ax = axs[1,0]
    ax.set_title("x pos")

    if predicted_states is not None:
        for i, arr in enumerate(predicted_states):
            ax.plot(predicted_t_arr(i), arr[0], '.-', alpha=alpha)

    ax.plot(t_arr, states[0], '*-', c='b')


    ax = axs[1,1]
    ax.set_title("y pos")

    if predicted_states is not None:
        for i, arr in enumerate(predicted_states):
            ax.plot(predicted_t_arr(i), arr[1], '.-', alpha=alpha)

    ax.plot(t_arr, states[1], '*-', c='b')


    ax = axs[1,2]
    ax.set_title("x vel")

    if predicted_states is not None:
        for i, arr in enumerate(predicted_states):
            ax.plot(predicted_t_arr(i), arr[2], '.-', alpha=alpha)

    ax.plot(t_arr, states[2], '*-', c='b')


    ax = axs[1,3]
    ax.set_title("y vel")

    if predicted_states is not None:
        for i, arr in enumerate(predicted_states):
            ax.plot(predicted_t_arr(i), arr[3], '.-', alpha=alpha)

    ax.plot(t_arr, states[3], '*-', c='b')



    ax = axs[2,0]
    ax.set_title("Yaw (deg)")

    if predicted_states is not None:
        for i, arr in enumerate(predicted_states):
            ax.plot(predicted_t_arr(i), np.degrees(arr[4]), '.-', alpha=alpha)

    ax.plot(t_arr, np.degrees(states[4]), '*-', c='b')


    ax = axs[2,1]
    ax.set_title("Omega (deg/s)")

    if predicted_states is not None:
        for i, arr in enumerate(predicted_states):
            ax.plot(predicted_t_arr(i), np.degrees(arr[5]), '.-', alpha=alpha)

    ax.plot(t_arr, np.degrees(states[5]), '*-', c='b')


    ax = axs[2,2]
    ax.set_title("ThetaC (deg)")

    if predicted_states is not None:
        for i, arr in enumerate(predicted_states):
            ax.plot(predicted_t_arr(i), np.degrees(arr[6]), '.-', alpha=alpha)

    ax.plot(t_arr, np.degrees(states[6]), '*-', c='b')


    ax = axs[2,3]
    ax.set_title("OmegaC (deg/s)")

    if predicted_states is not None:
        for i, arr in enumerate(predicted_states):
            ax.plot(predicted_t_arr(i), np.degrees(arr[7]), '.-', alpha=alpha)

    ax.plot(t_arr, np.degrees(states[7]), '*-', c='b')

    fig,ax = plt.subplots()
    ax.set_title("Pos (m)")

    if predicted_states is not None:
        for arr in predicted_states:
            ax.plot(arr[0], arr[1], '.-', alpha=alpha)

    ax.plot(np.where(~fuel_empty, states[0], np.nan),
            np.where(~fuel_empty, states[1], np.nan),'*-', c='b')

    ax.plot(np.where(fuel_empty, states[0], np.nan),
            np.where(fuel_empty, states[1], np.nan),'*-', c='r')

    fig.legend()
    plt.show(block=block)
