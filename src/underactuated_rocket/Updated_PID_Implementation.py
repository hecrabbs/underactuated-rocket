import matplotlib.animation as animation
import os
import sys
import time
import matplotlib.pyplot as plt
#from simple_pid import PID
from state import RocketParams, RocketState
from helpers import COL_VEC, cross_2d, new_col_vec

import logging

import control as ctl
import control.optimal as opt
import matplotlib.pyplot as plt
import numpy as np

######### Constants ############
ROCKET_PARAMS = RocketParams(
        m_B=100,
        m_C=1,
        m_fuel=10,
        m_fuel_rate=0.0000,
        height=1,
        radius=0.1,
        mag_d_0=0.5,
        mag_d_1=0.1,
        sigma_thrust=0.1,
        F_thrust_nominal=1800,
        seed=123456
    )




def ask_xy(prompt, default_xy):
    """
    Ask user for x,y as 'x,y'. Return np.array([x, y]).
    Press Enter to accept default.
    """
    default_str = f"{default_xy[0]}, {default_xy[1]}"
    s = input(f"{prompt} [default: {default_str}]: ").strip()
    if not s:
        return np.array(default_xy, dtype=float)

    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError("Please enter exactly two numbers: x,y")
    x = float(parts[0])
    y = float(parts[1])
    return np.array([x, y], dtype=float)

def run_PID(start=(0, 0), target=(5000, 10000)):
    
    
    rocket_params = ROCKET_PARAMS

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

    numItr = 100
    dist = np.linalg.norm(xf[:2] - tmp_state.get_state()[:2])
    kP = (1/dist) * 0.08
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
            print(i)
            tmp_state.transition(-control)
            prev_control = -control
        # elif -1 <= e_angle <= 1:
        #     tmp_state.transition(0)
        #     print("No Error!")
        else: 
            tmp_state.transition(prev_control)
        #tmp_state.transition(0)

        prev_e_angle = e_angle
        prevX = x
        prevY = y

    
    x_vals = np.array([s[0] for s in actual_states])
    y_vals = np.array([s[1] for s in actual_states])

    return actual_states, x_vals, y_vals, RocketYaw, DesYaw , YawError, DirecYaw, inputs

def run_simulation(x_vals, y_vals, initial_state, target):

    minErrors = min([np.linalg.norm((target[0] - x, target[1] - y)) for x,y in zip(x_vals, y_vals)])
    print(minErrors)

    ux_curr = np.sin(DirecYaw)
    uy_curr = np.cos(DirecYaw)
    ux_des = np.sin(DesYaw)
    uy_des = np.cos(DesYaw)

    # ---- Animation setup ----
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title("Rocket Trajectory Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.axis('equal')

    plt.plot(initial_state[0], initial_state[1], '^', markersize=20)
    plt.plot(target[0], target[1], 'X', markersize=20)
    ann = ax.annotate(("Iteration: ", 0), (x_vals[0], y_vals[0]), textcoords="offset points", xytext=(0,10), ha='center')

    traj_line, = ax.plot([], [], 'k-', lw=1.5)
    rocket_point, = ax.plot([], [], 'bo', markersize=6)

    curr_quiv = ax.quiver([], [], [], [], color='blue', scale=10)
    des_quiv = ax.quiver([], [], [], [], color='red', scale=10)

    pad = 0.5 * max(x_vals.max() - x_vals.min(), y_vals.max() - y_vals.min())
    ax.set_xlim(x_vals.min() - pad, x_vals.max() + pad)
    ax.set_ylim(y_vals.min() - pad, y_vals.max() + pad)

    def update(frame):
        traj_line.set_data(x_vals[:frame], y_vals[:frame])
        rocket_point.set_data(x_vals[frame], y_vals[frame])

        curr_quiv.set_offsets([x_vals[frame], y_vals[frame]])
        curr_quiv.set_UVC(ux_curr[frame], uy_curr[frame])

        des_quiv.set_offsets([x_vals[frame], y_vals[frame]])
        des_quiv.set_UVC(ux_des[frame], uy_des[frame])
        
        # ann = ax.annotate(("Iteration: ", frame), (x_vals[frame], y_vals[frame]), textcoords="offset points", xytext=(0,10), ha='center')
        ann.xy = (x_vals[frame], y_vals[frame])
        ann.set_text(f"Iteration: {frame} \n Error: {YawError[frame]}")       
        

        return traj_line, rocket_point, curr_quiv, des_quiv, ann

    ani = animation.FuncAnimation(
        fig, update, frames=len(x_vals),
        interval=70, blit=False, repeat=False
    )

    is_paused = {"value": False}  # use dict so we can mutate inside closure

    def on_key(event):
        # SPACE toggles pause/resume
        if event.key == ' ':
            if is_paused["value"]:
                ani.event_source.start()
                is_paused["value"] = False
                print("Resume animation")
            else:
                ani.event_source.stop()
                is_paused["value"] = True
                print("Pause animation")
        # 'q' stops completely
        elif event.key == 'q':
            ani.event_source.stop()
            is_paused["value"] = True
            print("Stop animation (q pressed)")

    fig.canvas.mpl_connect('key_press_event', on_key)

    # 7) CM Theta
    plt.figure(figsize=(8,6))
    C_Theta = [np.cos(a[6]) for a in actual_states]
    plt.plot(C_Theta, marker='o', linestyle='--', color='blue')
    plt.title("CM Theta")
    plt.grid(True)
    plt.show()


default_start_xy  = [0.0, 0.0]
default_target_xy = [5000.0, 10000.0]

start_xy  = ask_xy("Start position (x,y)",  default_start_xy)
target_xy = ask_xy("Target position (x,y)", default_target_xy)

xf = np.array([target_xy[0], target_xy[1], 0, 0, 0, 0, 0, 0])


iError = 0
actual_states = []
inputs = []
RocketYaw = []
DesYaw = []
YawError = []
DirecYaw = []

actual_states, x_vals, y_vals, RocketYaw, DesYaw, YawError, DirecYaw, inputs =  run_PID(start_xy, target_xy)
run_simulation(x_vals, y_vals, start_xy, target_xy)

print("Yaw Error")
print([a*(180/np.pi) for a in YawError])
print("Current Yaw")
print([a*(180/np.pi) for a in RocketYaw])
plt.figure(figsize=(12, 8))
print("Desired Yaw")
print([a*(180/np.pi) for a in DesYaw])
plt.figure(figsize=(12, 8))

# 1) Inputs
plt.subplot(3, 3, 2)
plt.plot(inputs, marker='o', linestyle='--', color='blue')
plt.title("Inputs")
plt.grid(True)

# 2) Rocket yaw
plt.subplot(3, 3, 3)
degRYaw = [a * (180/np.pi) for a in RocketYaw]
plt.plot(degRYaw, marker='o', linestyle='--', color='blue')
plt.title("Rocket Yaw")
plt.grid(True)

# 3) Desired yaw
plt.subplot(3, 3, 4, projection="polar")
r = [1]*len(DesYaw)
plt.plot(DesYaw, r, marker='o', linestyle='--', color='blue')
plt.title("Desired Yaw")
plt.grid(True)


# 4) Position trajectory (x vs y)
plt.subplot(3, 3, 1)
x_vals = [a[0] for a in actual_states]
y_vals = [a[1] for a in actual_states]
plt.plot(x_vals, y_vals, marker='o', linestyle='--', color='blue')
plt.title("Rocket Trajectory (x vs y)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)

# 5) yaw error
plt.subplot(3, 3, 5)
degYaw = [a * (180/np.pi) for a in YawError]
plt.plot(degYaw, marker='o', linestyle='--', color='blue')
plt.title("Yaw Error")
plt.grid(True)

# 6) Rocket Omega
R_Om = [a[5]*(180/np.pi) for a in actual_states]
plt.subplot(3, 3, 6)
plt.plot(R_Om, marker='o', linestyle='--', color='blue')
plt.title("Rocket Omega")
plt.grid(True)

# 7) CM Theta
C_Theta = [a[6]*(180/np.pi) for a in actual_states]
plt.subplot(3, 3, 7)
plt.plot(C_Theta, marker='o', linestyle='--', color='blue')
plt.title("CM Theta")
plt.grid(True)

# 8) CM Omega
C_Om = [a[7]*(180/np.pi) for a in actual_states]

plt.subplot(3, 3, 8)
plt.plot(C_Om, marker='o', linestyle='--', color='blue')
plt.title("CM Theta")
plt.grid(True)

plt.show()


