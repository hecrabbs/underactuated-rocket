import multiprocessing as mp
from time import time

import matplotlib.pyplot as plt
import numpy as np

import underactuated_rocket.cost_functions as cost_fcns
from underactuated_rocket.mpc import mpc
from underactuated_rocket.plotting import plot_results
from underactuated_rocket.state import RocketParams, RocketState


def mpc_wrapper(method):
    rocket_params = RocketParams(1e3)
    rocket_state = RocketState(params=rocket_params)
    goal_state = [1000, 0, 0, 0, 0, 0, 0, 0, 0]
    cost_fcn = cost_fcns.cost1(goal_state)
    duration = 200
    prediction_horizon = 8
    control_horizon = 2
    num_iter = int(duration/control_horizon)
    t0 = time()
    result =  mpc(rocket_state,
                cost_fcn,
                prediction_horizon,
                control_horizon,
                num_iter,
                minimize_method=method)
    print()
    print("METHOD: ", method, "TIME:", time()-t0)
    print()
    return result



def trial_wrapper(trial_num):
    rocket_params = RocketParams(1e3)
    rocket_state = RocketState(params=rocket_params)
    goal_state = [1e6, 0, 0, 0, 0, 0, 0, 0, 0]
    cost_fcn = cost_fcns.cost1(goal_state)
    duration = 200
    prediction_horizon = 8
    control_horizon = 2
    num_iter = int(duration/control_horizon)

    results = mpc(rocket_state,
            cost_fcn,
            prediction_horizon,
            control_horizon,
            num_iter,
            minimize_method="COBYLA")

    return results[1]

if __name__ == "__main__":

    # with mp.Pool() as pool:
    #     methods = [
    #         "COBYLA",
    #         "Powell",
    #     ]
    #     results = pool.map(mpc_wrapper, methods)

    # rocket_params = RocketParams(1e3)
    # for result in results:
    #     plot_results(rocket_params, *result, block=False)

    # rocket_params = RocketParams(1e3)
    # rocket_state = RocketState(params=rocket_params)
    # INIT_STATE = rocket_state.get_state()

    # num_trials = 2
    # goal_state = [1000, 0, 0, 0, 0, 0, 0, 0, 0]
    # cost_fcn = cost_fcns.cost1(goal_state)
    # duration = 4
    # prediction_horizon = 8
    # control_horizon = 2
    # num_iter = int(duration/control_horizon)

    trials = []

    # for i in range(num_trials):
    # def trial_wrapper(trial_num):
    #     t0 = time()

    #     results = mpc(rocket_state,
    #             cost_fcn,
    #             prediction_horizon,
    #             control_horizon,
    #             num_iter,
    #             minimize_method="COBYLA")
    #     # trials.append(results[1])

    #     print()
    #     print("TRIAL: ", trial_num, "TIME:", time()-t0)
    #     print()

    #     return results[1]

    num_trials = 12
    trial_num_arr = list(range(num_trials))
    with mp.Pool() as pool:
        trials = pool.map(trial_wrapper, trial_num_arr)

    # print(np.mean(results[0][1], axis=0))
    avg = np.mean(trials, axis=0)
    std = np.std(trials, axis=0)


    fig,axs = plt.subplots(3,1)

    t = np.arange(len(avg[:,0]))
    axs[0].plot(t, avg[:,0], label="mean")
    axs[0].fill_between(t, avg[:,0] - std[:,0], avg[:,0] + std[:,0], color="lightblue", label="stdev.")
    axs[0].set_xlabel("Timestep (s)")
    axs[0].set_ylabel("x position (m)")
    axs[0].grid(True)
    axs[0].legend(loc="upper left")

    axs[1].plot(t, avg[:,1])
    axs[1].fill_between(t, avg[:,1] - std[:,1], avg[:,1] + std[:,1], color="lightblue")
    axs[1].set_xlabel("Timestep (s)")
    axs[1].set_ylabel("y position (m)")
    axs[1].grid(True)

    axs[2].plot(avg[:,0], avg[:,1])
    axs[2].fill_betweenx(avg[:,1] - std[:,1], avg[:,0] - std[:,0], avg[:,0] + std[:,0], color="lightblue")
    axs[2].fill_betweenx(avg[:,1] + std[:,1], avg[:,0] - std[:,0], avg[:,0] + std[:,0], color="lightblue")
    axs[2].fill_between(avg[:,0] + std[:,0], avg[:,1] - std[:,1], avg[:,1] + std[:,1], color="lightblue")
    axs[2].fill_between(avg[:,0] - std[:,0], avg[:,1] - std[:,1], avg[:,1] + std[:,1], color="lightblue")
    axs[2].set_xlabel("x position (m)")
    axs[2].set_ylabel("y position (m)")
    axs[2].grid(True)

    fig.tight_layout()

    plt.show(block=True)
