import argparse
import logging
from dataclasses import replace
from typing import Callable

import control as ctl
import control.optimal as opt
import matplotlib.pyplot as plt
import numpy as np

import underactuated_rocket.cost_functions as cost_fcns
from underactuated_rocket.logger import FileLogListener
from underactuated_rocket.plotting import plot_results
from underactuated_rocket.state import RocketParams, RocketState


def mpc(rocket_state: RocketState,
        cost_fcn: Callable,
        prediction_horizon: int,
        control_horizon: int,
        num_iter: int,
        minimize_method: str="COBYLA"):
    """
    Run MPC.

    Parameters
    ----------
    rocket_state : RocketState
        State object for the rocket being controlled.

    prediction_horizon : int
        Duration over which cost is minimized each iteration.

    control_horizon : int
        Duration of predicted horizon for which predicted inputs are used.

    num_iter : int
        Number of prediction horizions to minimize.

    cost_fcn : Callable
        Cost function. See `underactuated_rocket.cost_functions` for examples.

    minimize_method : str, default="COBYLA"
        Which method should `scipy.optimize.minimize` use?

    """
    # Save initial state
    INIT_STATE = rocket_state.get_state()
    STATE_LEN = len(INIT_STATE)
    DURATION = 1 + num_iter*control_horizon

    # Copy rocket params, but w/ different seed
    model_params = replace(rocket_state.params, seed=654321)
    model_state = RocketState(model_params)

    def sys_updfcn(t,x,u,params):
        model_state.set_state(x)
        return model_state.transition(u[0])

    model_sys = ctl.nlsys(sys_updfcn,
                          inputs=["alpha_C"],
                          states=STATE_LEN,
                          dt=True)

    # Data structures
    predicted_inputs = np.empty((num_iter, prediction_horizon))
    predicted_states = np.empty((num_iter, STATE_LEN,
                                 prediction_horizon))
    predicted_cost = np.empty(num_iter)

    actual_inputs = np.empty(DURATION-1)
    actual_states = np.empty((STATE_LEN, DURATION))
    actual_states[:,0] = INIT_STATE

    pred_t_arr = np.arange(prediction_horizon)
    current_state = INIT_STATE

    for i in range(num_iter):
        print(i)

        res = opt.solve_optimal_trajectory(model_sys,
                                           pred_t_arr,
                                           current_state,
                                           cost_fcn,
                                           minimize_method=minimize_method)
        print(f"{res.message}\n"
              f"{res.success}")

        # Update data structures
        predicted_inputs[i] = res.inputs[0]
        predicted_states[i] = res.states
        predicted_cost[i] = res.cost

        # Update actual system
        for j in range(control_horizon):
            u = res.inputs[0][j]
            s = rocket_state.transition(u)
            actual_idx = i*control_horizon + j
            actual_inputs[actual_idx] = u
            actual_states[:,actual_idx+1] = s

        # Update current model state
        current_state = s

        print()

    plot_results(rocket_params,
                 actual_inputs,
                 actual_states,
                 control_horizon,
                 prediction_horizon,
                 predicted_inputs,
                 predicted_states,
                 predicted_cost)

if __name__ == "__main__":
    # Logging
    logger_name = "mpc_logger"
    level = logging.DEBUG
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # Log to file in background
    listener = FileLogListener("mpc.log", logger)
    listener.start()

    try:
        rocket_params = RocketParams(100e3, 40, 4, seed=123456)
        rocket_state = RocketState(params=rocket_params,
                                   logger_name=logger_name)

        goal_state = [1000, 0, 0, 0, 0, 0, 0, 0, 0]

        cost_fcn = cost_fcns.cost1(goal_state)

        mpc(rocket_state, cost_fcn, 8, 2, 80, minimize_method="COBYLA")
    except:
        raise
    finally:
        listener.stop()
