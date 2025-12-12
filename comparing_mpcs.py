from time import time

from matplotlib import pyplot as plt

import underactuated_rocket.cost_functions as cost_fcns
from underactuated_rocket.mpc import mpc
from underactuated_rocket.plotting import plot_results
from underactuated_rocket.state import RocketParams, RocketState

duration = 4
methods = ["CG", "COBYLA", "Powell", "BFGS"]
goal_pxs = [100, 1000, 10e3, 100e3, 1e6]
goal_states = [
    [goal_pxs[i], 0, 0, 0, 0, 0, 0, 0, 0] for i in range(len(goal_pxs))]

results_arr = []

for i, method in enumerate(methods):
    for j, goal_state in enumerate(goal_states):
        rocket_params = RocketParams(m_C=100e3, mag_d_0=0)
        rocket_state = RocketState(params=rocket_params)
        cost_fcn = cost_fcns.cost1(goal_state)

        prediction_horizon = 8
        control_horizon = 2
        num_iter = int(duration/control_horizon)
        print(f"Running MPC for {num_iter} iterations "
                f"({duration} seconds)...\n")

        t0 = time()
        results = mpc(rocket_state,
                      cost_fcn,
                      prediction_horizon,
                      control_horizon,
                      num_iter,
                      minimize_method=method)
        print("Total execution time:", time()-t0)

        if i == len(methods)-1 and j == len(goal_states)-1: block = True
        else: block = False
        plot_results(rocket_params, *results, block=block)
        plt.pause(0.1)
