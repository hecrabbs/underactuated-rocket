import os
import sys
import time
import matplotlib.pyplot as plt
#from simple_pid import PID
from state import RocketParams, RocketState

import logging

import control as ctl
import control.optimal as opt
import matplotlib.pyplot as plt
import numpy as np




#t0 = time()

rocket_params = RocketParams(
    m_B=100,
    m_C=1,
    m_fuel=10,
    m_fuel_rate=0.0000,
    height=1,
    radius=0.1,
    mag_d_0=0.1,
    mag_d_1=0.1,
    sigma_thrust=0.1,
    F_thrust_nominal=1700,
    seed = 123456
)


xf = np.array([5000, 1000, 0, 0, 0, 0, 0, 0])

#pid = PID(kP, kI, kD, setpoint=1)

tmp_state = RocketState(rocket_params)
print("Starting state:", tmp_state.get_state())
LEN_STATE_SPACE = len(tmp_state.get_state())

def updfcn(t, x, u, params):
    tmp_state.set_state(x)
    tmp_state.transition(u[0])
    return tmp_state.get_state()

sys = ctl.nlsys(updfcn, inputs=["alpha_C"], states=LEN_STATE_SPACE, dt=1)



# inputs = np.empty((num_iter, iter_dur))
# computed_states = np.empty((num_iter, LEN_STATE_SPACE, iter_dur))
# computed_cost = np.empty(num_iter)

iError = 0
actual_states = []
inputs = []
RocketYaw = []
DesYaw = []
YawError = []
DirecYaw = []


# kP = 2e-2
# kI = 0
# kD = 5e-5
# #input_example = [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1]
# for i in range(numItr):
#     actual_states.append(tmp_state.get_state())
    
#     # Position error vector
#     currError_x = xf[0] - tmp_state.get_state()[0]
#     currError_y = xf[1] - tmp_state.get_state()[1]

#     #Desired yaw (RADIANS)
#     if currError_x >= 0: 
#         if currError_y >= 0: 
#             # print("Q1")
#             # print(currError_x)
#             # print(currError_y)
#             yaw_des_rad =  (np.pi/2) -np.arctan(currError_y/currError_x)
#         else: 
#             # print("Q4")
#             # print(currError_x)
#             # print(currError_y)
#             yaw_des_rad = (np.pi/2) - (np.arctan(currError_y/ currError_x) + (2*np.pi))
#     else: 
#         if currError_y >= 0: 
#             # print("Q2")
#             # print(currError_x)
#             # print(currError_y)
#             yaw_des_rad = ((np.pi/2) -  (np.arctan(currError_y/ currError_x) + np.pi) )
#         else: 
#             # print("Q3")
#             # print(currError_x)
#             # print(currError_y)
#             yaw_des_rad = ((np.pi/2) -  (np.arctan(currError_y/ currError_x) + np.pi) )
#     print(yaw_des_rad)
#     # Current yaw in radians
#     #yaw_des_rad =  np.arctan2(currError_y, currError_x)
#     currYaw_rad = ( tmp_state.get_state()[4] % (2 * np.pi) )

#     # Yaw error (radians)
    
#     desired_Ang_Velo = yaw_des_rad - currYaw_rad

#     currError_velo = desired_Ang_Velo - tmp_state.get_state()[5]

#     #posError = np.linalg.norm([currError_x, currError_y])
#     # Wrap to (-pi, pi]
#     #currError_rad = (currError_rad + np.pi) % (2*np.pi) - np.pi

#     # Convert back to degrees if your PID expects degrees
#     #currError = currError_rad * (180 / np.pi)

#     RocketYaw += [currYaw_rad]
#     DesYaw += [yaw_des_rad]
#     YawError +=[desired_Ang_Velo]

#     iError += desired_Ang_Velo
    
#     if prevError:
#         dError = desired_Ang_Velo - prevError
#     else:
#         dError = 0
        
#     control = kP * desired_Ang_Velo + kI * iError + kD * dError
#     tmp_state.transition(control)
#     #tmp_state.transition(input_example[i])
    
#     inputs += [control]

#     prevError = currError_velo

numItr = 80
kP = (0.5) * 2e-5
kI = 2e-10
kD = 0.5 * kP

iError = 0.0
prevX  = None
prevY  = None
prev_e_angle = None


for i in range(numItr):
    state = tmp_state.get_state()
    x, y = state[0], state[1]
    currYaw_rad  = state[4]         # yaw (rad)
    currYawRate  = state[5]          # yaw rate (rad/s)
    
    dx, dy = 0, 0
    if prevX: dx = x - prevX
    if prevY: dy = y - prevY

    actual_states.append(state)

    # --- Position error ---
    currError_x = xf[0] - x
    currError_y = xf[1] - y

    # --- Desired yaw from +y axis ---
    yaw_des = np.arctan2(currError_y, currError_x)        # from +x axis
    yaw_des = (np.pi / 2.0) - yaw_des                # from +y axis

    # -- Current angle from +y axis --
    curr_yaw = (np.pi / 2.0) - np.arctan2(dy, dx)

    # --- Yaw error (wrap to [-pi, pi]) ---
    e_angle = yaw_des - curr_yaw
    
    # e_angle = -yaw_des_rad - (currYaw_rad % (2*np.pi))
    # e_angle = (e_angle + np.pi) % (2.0 * np.pi) - np.pi

    # --- Integral of yaw error ---
    # If you have dt, do iError += e_angle * dt
    iError += e_angle

    # --- Derivative term from yaw rate ---
    # de/dt ≈ -psi_dot when yaw_des is slowly varying
    e_dot = 0   
    if prev_e_angle: e_dot = e_angle - prev_e_angle

    # --- PID output: angular acceleration of control mass ---
    control = kP * e_angle + kI * iError + kD * e_dot
    
    # Log for plotting
    RocketYaw.append(currYaw_rad)
    DirecYaw.append(curr_yaw)
    DesYaw.append(yaw_des)
    YawError.append(e_angle)
    inputs.append(control)

    if abs(x) - abs(xf[0]) > 0 and abs(y) - abs(xf[1]) >0:
        control = -control

    # Step the plant
    tmp_state.transition(-control)

    prev_e_angle = e_angle
    prevX = x
    prevY = y

# Logging
level = logging.WARNING
logger_name = "rocket_state"
logger = logging.getLogger(logger_name)
logger.setLevel(level)
handler = logging.StreamHandler()
handler.setLevel(level)
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

rocket_state = RocketState(params=rocket_params, logger_name=logger_name)
#actual_states[:,0] = rocket_state.get_state()
#print(inputs)
# for i in range(10,15):
#     print(inputs[i])
#     print(actual_states[i])
#print([a[4] for a in actual_states])
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

# Direction vectors for each yaw (your +y, CCW convention)
# d = (-sin(psi), cos(psi))
ux_curr = np.sin(DirecYaw)
uy_curr =  np.cos(DirecYaw)

ux_des  = np.sin(DesYaw)
uy_des  =  np.cos(DesYaw)

x_vals = np.array(x_vals)
y_vals = np.array(y_vals)
# Optional: subsample so the plot isn't crazy cluttered
N = len(x_vals)
step = max(1, N // 50)   # at most ~50 sets of arrows
idx = np.arange(0, N, step)
span = max(x_vals.max() - x_vals.min(), y_vals.max() - y_vals.min())
arrow_len = 0.05 * span

# ux_curr *= arrow_len
# uy_curr *= arrow_len
# ux_des  *= arrow_len
# uy_des  *= arrow_len

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, 'k-', alpha=0.5, label='trajectory')

# Current yaw (blue)
plt.quiver(x_vals[idx], y_vals[idx],
           ux_curr[idx], uy_curr[idx],
           angles='xy', scale_units='xy', scale=0.5,
           width=0.03, color='b', label='current yaw')

# Desired yaw (red)
plt.quiver(x_vals[idx], y_vals[idx],
           ux_des[idx], uy_des[idx],
           angles='xy', scale_units='xy', scale=0.5,
           width=0.03, color='r', alpha=0.7, label='desired yaw')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Rocket Trajectory with Current vs Desired Yaw')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

# Adjust spacing
plt.tight_layout()
plt.show()

#animate_run(actual_states, RocketYaw, DesYaw)

