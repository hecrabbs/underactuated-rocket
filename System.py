import numpy as np
import matplotlib.pyplot as plt

# compute next state from current state and inputs
def rocket_step(
    state,     # [x, y, vx, vy, psi, omega, r, r_dot]
    ddot_r,    # control input (scalar)
    params,    # dictionary of system parameters
):
    """
    Computes s[t+1] from s[t] for the underactuated rocket system (Δt = 1).
    """

    # Unpack state
    x, y, vx, vy, psi, omega, r, r_dot = state

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
    g = params.get("g", 9.81)

    # --- Thrust with noise ---
    w = np.random.normal(0, sigma_thrust)
    wx = np.random.normal(0, sigma_thrust)
    F_thrust_B = np.array([0.0, -(F_nominal + w)])      # in body frame
    F_normthrust_B = -F_thrust_B                        # upward direction

    # --- Control mass force in body frame ---
    F_C_B = np.array([m_C * ddot_r, 0.0])

    # --- Rotation matrix (B→I) ---
    R_BI = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi),  np.cos(psi)]
    ])

    # --- Forces in inertial frame ---
    F_g = np.array([0.0, -m_total * g])
    F_total = R_BI @ (F_normthrust_B + F_C_B) + F_g

    # --- Torques ---
    F_normthrust_mag = F_nominal + w
    F_C_mag = m_C * ddot_r

    tau_normthrust = -r * d1 * F_normthrust_mag
    tau_C = -d0 * F_C_mag
    tau_total = tau_normthrust + tau_C

    # --- Angular acceleration ---
    alpha = tau_total / I

    # --- Discrete update equations (Δt = 1) ---
    # Position
    x_next = x + vx
    y_next = max(y + vy, y_min)

    # Velocity
    vx_next = vx + (F_total[0] / m_total)
    vy_next = vy + (F_total[1] / m_total)

    # Yaw
    psi_next = psi + omega

    # Angular velocity
    omega_next = omega + alpha

    # Control mass position & velocity
    r_next = r + r_dot
    r_dot_next = r_dot + ddot_r

    # Construct next state
    next_state = np.array([
        x_next, y_next,
        vx_next, vy_next,
        psi_next, omega_next,
        r_next, r_dot_next
    ])

    return next_state


def main():
    s0 = [0, 0, 0, 0, 0, 0, 0, 0]
    params = {
        "m_B" : 1.5,
        "m_C" : 0.5,
        "F_nominalthrust" : 20,
        "sigma_thrust" : 0.75,
        "d0" : 1,
        "d1" : 0,
        "y_min" : 0
    }
    mass    = params['m_B'] + params['m_C']
    length  = 1
    params['I'] = (0.25) * float(mass) * (0.5)**2 + float(1/3) * mass * (length ** 2)

    inputs = [0.4, 0.4, 0.2, 0.8, 1.2, 2.0, 2.4, 1.8, 1.2, 0.5, -0.3]
    
    state = s0

    x_coords = []
    y_coords = []
    for input in inputs[:5]:
        x_coords.append(state[0])
        y_coords.append(state[1])
        state = rocket_step(state, input, params)
    plt.plot(x_coords, y_coords)
    plt.show()

if __name__ == '__main__':
    main()
