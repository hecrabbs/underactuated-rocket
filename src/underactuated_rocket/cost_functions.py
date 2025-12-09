"""
Some example cost functions. These functions should all take a goal state and
return a cost function.

Default goal states have been tested and seem to work, but other goal states
may not.
"""

import math

import control as ctl
import control.optimal as opt
import numpy as np

DUMMY_SYS = ctl.nlsys(lambda _: None, inputs=1, states=9, dt=True)

def cost1(goal_state=[1000, 0, 0, 0, 0, 0, 0, 0, 0]):
    """Drives rocket to x position, and then up. Works well with "COBYLA"
    minimize method, but produces fairly large inputs & omega C."""
    Q = np.diag((1,     # Drive x to goal x
                 0,
                 8,     # Keep vx small
                 0,
                 100e3, # Keep yaw small
                 0,
                 0,
                 10,    # Keep omega_C small
                 0
    ))
    R = 10 # Keep inputs small
    return opt.quadratic_cost(DUMMY_SYS, Q, R, x0=goal_state)

# def cost2(goal_state):
#     Q = np.diag((1,0,1,0,100e3,0,0,0,0))
#     R = 200
#     return opt.quadratic_cost(DUMMY_SYS, Q, R, x0=goal_state)

# def cost3(goal_state):
#     Q = np.diag((1,1,0,0,0,100e3,0,0,0))
#     R = 1
#     return opt.quadratic_cost(DUMMY_SYS, Q, R, x0=goal_state)

# def desire_yaw_cost(goal_state):
#     def cost_fcn(x,u):
#         pxdiff = goal_state[0] - x[0]
#         pydiff = goal_state[1] - x[1]
#         desired_yaw = -math.atan2(pxdiff, pydiff)
#         return (1e3*abs(desired_yaw - x[4])**2
#               + 1e2*(x[5]**2))
#     return cost_fcn
