import matplotlib.pyplot as plt
import numpy as np

from state import State

ACCEL_GRAVITY = 9.81

def cross_2d(arr1, arr2):
    return arr1[0]*arr2[1] - arr1[1]*arr2[0]

def rocket_step(state: State,     # [x, y, vx, vy, psi, omega, r, r_dot]
    r_ddot,    # control input (scalar)
    params,    # dictionary of system parameters
):
    """
    Computes s[t+1] from s[t] for the underactuated rocket system (Δt = 1).
    """

    # --- Unpack parameters ---
    m_B = params["m_B"] # body mass
    m_C = params["m_C"] # control mass
    y_min = params["y_min"]
    m_total = m_B + m_C
    F_nominal = params["F_nominalthrust"]
    sigma_thrust = params["sigma_thrust"] # for thrust noise
    d0 = params["d0"] # vertical offset
    d1 = params["d1"] # horizontal offset
    I = params["I"] # inertia
    r_thrust = params["r_thrust"]

    # --- Translational Force ---
    # _I for inertial frame, _B for body frame.
    F_g_I = np.array([0.0, -ACCEL_GRAVITY * m_total])[:, np.newaxis]
    w_thrust = np.random.normal(0, sigma_thrust) # Thrust noise
    F_thrust_B = np.array([0.0, F_nominal + w_thrust])[:, np.newaxis]

    # Rotation matrix (B→I)
    psi = state.psi
    R_BI = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi),  np.cos(psi)]
    ])

    F_total_I = (R_BI @ F_thrust_B) + F_g_I

    # --- Rotational Force ---
    r = state.r
    r_CM_B = m_C*(d0 + r*d1) / m_total
    r_CM_thrust_B = -r_CM_B + r_thrust
    tau_total_I = cross_2d(R_BI @ r_CM_thrust_B, R_BI @ F_thrust_B)[0]

    # --- Angular acceleration ---
    alpha_I = tau_total_I / I

    # --- Discrete update equations (Δt = 1) ---
    state.transition(F_total_I, alpha_I, r_ddot)

def main():
    params = {
        "m_B" : 1.5,
        "m_C" : 0.5,
        "F_nominalthrust" : 20,
        "sigma_thrust" : 0.75,
        "d0" : np.array([0, 1])[:, np.newaxis],
        "d1" : np.array([1, 0])[:, np.newaxis],
        "y_min" : 0,
        "r_thrust": np.array([0, -1])[:, np.newaxis]
    }
    mass    = params['m_B'] + params['m_C']
    length  = 1
    params['I'] = (0.25) * float(mass) * (0.5)**2 + float(1/3) * mass * (length ** 2)

    state = State(mass)

    inputs = [0.4, 0.4, 0.2, 0.8, 1.2, 2.0, 2.4, 1.8, 1.2, 0.5, -0.3]
    x_coords = []
    y_coords = []
    for input in inputs:
        print(state.p)
        x_coords.append(state.p[0][0])
        y_coords.append(state.p[1][0])
        rocket_step(state, input, params)
    plt.plot(x_coords, y_coords)
    plt.show()

if __name__ == '__main__':
    main()
