import matplotlib.pyplot as plt
#from Updated_PID_Implementation import run_PID
import numpy as np
from state import RocketParams, RocketState

FINAL_POINTS = [100, 1000, 10000, 100000, 1000000]


def run_PID(start=(0, 0), target=(5000, 10000)):
    
    
    rocket_params = RocketParams(100e3, 40, 4)

    # End goal
    xf = np.array([target[0], target[1], 0, 0, 0, 0, 0, 0])

    tmp_state = RocketState(rocket_params)
    
    # Overwrite the initial state’s position
    initial_state = tmp_state.get_state()
    initial_state[0] = start[0]
    initial_state[1] = start[1]
    tmp_state.set_state(initial_state)

    actual_states = []
    RocketYaw = []
    DesYaw = []
    YawError = []
    DirecYaw = []
    inputs = []

    numItr = 200
    dist = np.linalg.norm(xf[:2] - tmp_state.get_state()[:2])
    kP = (1/dist) * 0.09
    #kP = 0
    #kI = 2e-10 * kP
    kI = 0
    kD = 0.5 * kP

    iError = 0.0
    prevX = None
    prevY = None
    prev_e_angle = None

    # ---- PID Simulation Loop ----
    for i in range(numItr):
        state = tmp_state.get_state()
        x, y = state[0], state[1] #Current xy position of rocket
        currYaw_rad = state[4]    #Current yaw angle of rocket

        dx = dy = 0
        if prevX is not None: dx = x - prevX
        if prevY is not None: dy = y - prevY

        actual_states.append(state)

        # Desired direction based on target
        currError_x = xf[0] - x
        currError_y = xf[1] - y

        yaw_des = np.arctan2(currError_y, currError_x)
        yaw_des = (np.pi / 2.0) - yaw_des

        #Current angle direction of movement (from +y axis, moving CCW)
        curr_yaw = (np.pi / 2.0) - np.arctan2(dy, dx)
        #Error in current angle from the desired angle of movement
        e_angle = (yaw_des - curr_yaw)
        iError += e_angle

        e_dot = 0
        if prev_e_angle is not None:
            e_dot = e_angle - prev_e_angle

        control = kP * e_angle + kI * iError + kD * e_dot
        prev_control = 0
        RocketYaw.append(currYaw_rad)
        DirecYaw.append(curr_yaw)
        DesYaw.append(yaw_des)
        YawError.append(e_angle)
        inputs.append(-control)
        #Update the input every other iteration
        if(i % 2 == 0 and i > 0):
            #print(i)
            tmp_state.transition(-control, 1)
            prev_control = -control
        # elif -1 <= e_angle <= 1:
        #     tmp_state.transition(0)
        #     print("No Error!")
        else: 
            tmp_state.transition(prev_control, 1)
        #tmp_state.transition(0)

        prev_e_angle = e_angle
        prevX = x
        prevY = y

    
    x_vals = np.array([s[0] for s in actual_states])
    y_vals = np.array([s[1] for s in actual_states])

    return actual_states, x_vals, y_vals, RocketYaw, DesYaw , YawError, DirecYaw, inputs



# fig, ax = plt.subplots(figsize=(12,8))
# ax.set_title("Rocket Trajectory Averages + STD")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.grid(True)
# ax.axis('equal')


# for x in FINAL_POINTS: 
#     x_sum = np.zeros(200)
#     y_sum = np.zeros(200)

#     data_X = []
#     data_Y = []
#     for i in range(50): 
    
#         actual_states, x_vals, y_vals, RocketYaw, DesYaw , YawError, DirecYaw, inputs = run_PID(target=(x, 1e5))
#         data_X.append(x_vals)
#         data_Y.append(y_vals)
#         x_sum += np.array(x_vals)
#         y_sum += np.array(y_vals)
        
    
#     x_avg = x_sum / 50
#     y_avg = y_sum / 50        

#     x_std = np.std(data_X, axis=0)
#     y_std = np.std(data_Y, axis=0)
#     for x_v, y_v in zip(data_X, data_Y): 
#         #print("plotting iter: ", i)
#         # if i == 10: 
#         #     plt.plot(x_v, y_v - 10, 'k-', lw=1.5)
#         # else: 
        
#         plt.plot(x_v, y_v, 'k-', lw=1.5)
#     # Plot shaded standard deviation region
#     # plt.fill_between([i for i in range(200)],
#     #                 x_avg - x_std,
#     #                 x_avg + x_std,
#     #                 alpha=0.2,        # transparency
#     #                 label='±1 std')
#     plt.show()
# plt.show()
    


for x in FINAL_POINTS: 
    data_X = []
    data_Y = []

    for i in range(50): 
        actual_states, x_vals, y_vals, RocketYaw, DesYaw , YawError, DirecYaw, inputs = run_PID(target=(x, 1e5))
        data_X.append(x_vals)   # length 200
        data_Y.append(y_vals)   # length 200

    # Convert to arrays: shape -> (n_runs, n_steps) = (50, 200)
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)

    # Mean and std over runs (axis=0)
    x_avg = np.mean(data_X, axis=0)
    y_avg = np.mean(data_Y, axis=0)
    x_std = np.std(data_X, axis=0)
    y_std = np.std(data_Y, axis=0)

    t = np.arange(data_X.shape[1])  # time index 0..199

    # ---- Plot X trajectory with shaded std ----
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, x_avg, label='x mean', linewidth=2)
    plt.fill_between(t, x_avg - x_std, x_avg + x_std, alpha=0.2, label='x ±1σ')
    plt.xlabel('Timestep')
    plt.ylabel('x position')
    plt.title(f'Trajectories for target x = {x}')
    plt.grid(True)
    plt.legend()

    # ---- Plot Y trajectory with shaded std ----
    plt.subplot(3, 1, 2)
    plt.plot(t, y_avg, label='y mean', linewidth=2)
    plt.fill_between(t, y_avg - y_std, y_avg + y_std, alpha=0.2, label='y ±1σ')
    plt.xlabel('Timestep')
    plt.ylabel('y position')
    plt.grid(True)
    plt.legend()

    # ---- Plot Y trajectory with shaded std ----
    plt.subplot(3, 1, 3)
    plt.plot(x_avg, y_avg, label='Average Rocket Trajectory', linewidth=2)
    #plt.fill_between(t, y_avg - y_std, y_avg + y_std, alpha=0.2, label='y ±1σ')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
