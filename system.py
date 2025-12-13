# system_kf_ilqr.py
import argparse
import logging
import math

import matplotlib.pyplot as plt
import numpy as np

from helpers import new_col_vec
from state import RocketParams
from kf_ilqr import (
    ILQRConfig,
    KFConfig,
    FullStateKF,
    make_model_and_true_rockets,
    ilqr,
)

# --- toggles ---
USE_KF = False          # if False, plan from true state and skip KF predict/update
USE_OBS_NOISE = False   # if False, measurement = true state (no observation noise)
EARLY_STOP = True       # <--- NEW: stop when distance starts increasing



def main(verbosity=logging.WARNING):
    # Logging root level
    logging.basicConfig(level=verbosity)

    # --------------------
    # Rocket parameters
    # --------------------
    params = RocketParams(
        m_B=100,
        m_C=1.2,
        m_fuel=10,
        m_fuel_rate=0.0,
        height=1.0,
        radius=0.1,
        mag_d_0=10.0,
        mag_d_1=0.1,
        sigma_thrust=0.1,       # process noise (true rocket only)
        F_thrust_nominal=1500.0,
        mag_a_g=9.81,
        seed=123456,
    )

    rocket_true, rocket_model = make_model_and_true_rockets(params)

    # --------------------
    # Start / goal states
    # --------------------
    # State = [p_x, p_y, v_x, v_y, psi, omega, theta_C, omega_C, m_fuel]
    x0 = rocket_true.get_state()
    # Example: want to move to x=50, keep everything else near zero
    x_goal = x0.copy()
    x_goal[0] = 100.0     # target x
    x_goal[1] = 200.0      # target y
    # You can tweak the goal further as needed

    # --------------------
    # KF initial state
    # --------------------
    x_hat0 = x0.copy()        # initial estimate = true
    P0 = np.eye(len(x0)) * 1.0

    kf_cfg = KFConfig(
        process_sigma=0.1,
        meas_sigma=0.05,
    )

    kf = FullStateKF(x_hat0, P0, kf_cfg) if USE_KF else None


    # --------------------
    # iLQR configuration
    # --------------------
    ilqr_cfg = ILQRConfig(
        N=150,
        max_iter=20,
        tol=1e-3,
        reg=1e-4,
        u_min=-2.0,
        u_max=2.0,
        eps_fd=1e-4,
    )

    # --------------------
    # Closed-loop simulation
    # --------------------
    T = 20  # max number of control steps overall
    true_traj = []
    est_traj = []
    control_seq = []
    distances = []

    best_dist = None
    best_step = None

    for t in range(T):
        # Current true state
        x_true = rocket_true.get_state()

        # Distance to goal in (p_x, p_y)
        dist = np.linalg.norm(x_true[:2] - x_goal[:2])

        if best_dist is None:
            # First step: initialize best distance
            best_dist = dist
            best_step = t
        else:
            # Only apply the "stop when distance increases" rule if EARLY_STOP is True
            if EARLY_STOP and dist > best_dist:
                print(
                    f"Distance started increasing at step {t}; "
                    f"best distance = {best_dist:.3f} m at step {best_step}"
                )
                break

            # Always track the closest distance reached so far
            if dist < best_dist:
                best_dist = dist
                best_step = t

        # Only append after we've decided *not* to break
        true_traj.append(x_true.copy())
        distances.append(dist)

        # ---- choose state used for planning / "estimate" ----
        if USE_KF:
            x_for_control = kf.x.copy()
            est_traj.append(kf.x.copy())
        else:
            x_for_control = x_true.copy()
            est_traj.append(x_true.copy())

        # Plan from current estimate to goal using iLQR
        u_plan, x_plan = ilqr(params, x_for_control, x_goal, ilqr_cfg, rocket_model)
        u0 = float(u_plan[0])
        control_seq.append(u0)

        # Apply control to true rocket (with noise)
        rocket_true.transition(u0)

        # Measurement (with or without noise)
        if USE_OBS_NOISE:
            z = rocket_true.observe(sigma_meas=kf_cfg.meas_sigma)
        else:
            z = rocket_true.get_state()

        # KF (if enabled)
        if USE_KF:
            kf.predict(u0, rocket_model, ilqr_cfg.eps_fd)
            kf.update(z)

    
    if best_dist is not None:
        if EARLY_STOP:
            print(f"Closest distance = {best_dist:.3f} m at step {best_step}")
        else:
            print(f"(No early stop) Closest distance over {T} steps = "
                  f"{best_dist:.3f} m at step {best_step}")



    true_traj = np.array(true_traj)
    est_traj = np.array(est_traj) if USE_KF else None
    control_seq = np.array(control_seq)
    distances = np.array(distances)

    T_sim = len(true_traj)
    t_axis = np.arange(T_sim)
    t_axis_ctrl = np.arange(len(control_seq))
    t_axis_dist = np.arange(len(distances))



    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    fig.tight_layout(pad=2.0)

    

    # x vs y 
    axs[0, 0].set_title("Position (x vs y)") 
    axs[0, 0].plot(true_traj[:, 0], true_traj[:, 1], label="true")
    if USE_KF:
        axs[0, 0].plot(est_traj[:, 0], est_traj[:, 1], "--", label="KF est")
    axs[0, 0].scatter([x_goal[0]], [x_goal[1]], marker="x", color="red", label="goal") 
    axs[0, 0].legend() 
    axs[0, 0].set_xlabel("x") 
    axs[0, 0].set_ylabel("y")

    # x over time
    axs[0, 1].set_title("x position over time")
    axs[0, 1].plot(t_axis, true_traj[:, 0], label="true")
    if USE_KF:
        axs[0, 1].plot(t_axis, est_traj[:, 0], "--", label="KF est")
    axs[0, 1].axhline(x_goal[0], color="red", linestyle=":", label="goal")
    axs[0, 1].legend()
    axs[0, 1].set_xlabel("time step")
    axs[0, 1].set_ylabel("x")

    # yaw
    axs[1, 0].set_title("Yaw psi (deg)")
    axs[1, 0].plot(t_axis, np.degrees(true_traj[:, 4]), label="true")
    if USE_KF:
        axs[1, 0].plot(t_axis, np.degrees(est_traj[:, 4]), "--", label="KF est")
    axs[1, 0].set_xlabel("time step")
    axs[1, 0].set_ylabel("psi (deg)")
    axs[1, 0].legend()

    # omega_C
    axs[1, 1].set_title("omega_C (deg/s)")
    axs[1, 1].plot(t_axis, np.degrees(true_traj[:, 7]), label="true")
    if USE_KF:
        axs[1, 1].plot(t_axis, np.degrees(est_traj[:, 7]), "--", label="KF est")
    axs[1, 1].set_xlabel("time step")
    axs[1, 1].set_ylabel("omega_C (deg/s)")
    axs[1, 1].legend()

    # control
    axs[2, 0].set_title("Control alpha_C (deg/s^2)")
    axs[2, 0].plot(t_axis_ctrl, np.degrees(control_seq))
    axs[2, 0].set_xlabel("time step")
    axs[2, 0].set_ylabel("alpha_C (deg/s^2)")

    # v_x
    axs[2, 1].set_title("v_x (m/s)")
    axs[2, 1].plot(t_axis, true_traj[:, 2], label="true")
    if USE_KF:
        axs[2, 1].plot(t_axis, est_traj[:, 2], "--", label="KF est")
    axs[2, 1].set_xlabel("time step")
    axs[2, 1].set_ylabel("v_x")
    axs[2, 1].legend()

    # Distance to goal plot
    axs[3, 0].set_title("Distance to Goal")
    axs[3, 0].plot(t_axis_dist, distances, label="distance")
    axs[3, 0].set_xlabel("Time step")
    axs[3, 0].set_ylabel("Distance (m)")
    axs[3, 0].grid(True)
    axs[3, 0].legend()




    fig.savefig("results.png", dpi=300, bbox_inches="tight")  # save as image
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbosity", type=str, default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    )
    args = parser.parse_args()
    verbosity_level = getattr(logging, args.verbosity)
    main(verbosity_level)
