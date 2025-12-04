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


def main(verbosity=logging.WARNING):
    # Logging root level
    logging.basicConfig(level=verbosity)

    # --------------------
    # Rocket parameters
    # --------------------
    params = RocketParams(
        m_B=100,
        m_C=1,
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
    x_goal[0] = 50.0     # target x
    x_goal[1] = 50.0      # target y
    x_goal[2] = 0.0      # v_x
    x_goal[4] = 0.0      # psi
    x_goal[5] = 0.0      # omega
    # You can tweak the goal further as needed

    # --------------------
    # KF initial state
    # --------------------
    x_hat0 = x0.copy()        # initial estimate = true (you can perturb this)
    P0 = np.eye(len(x0)) * 1.0

    kf_cfg = KFConfig(
        process_sigma=0.1,
        meas_sigma=0.05,
    )
    kf = FullStateKF(x_hat0, P0, kf_cfg)

    # --------------------
    # iLQR configuration
    # --------------------
    ilqr_cfg = ILQRConfig(
        N=30,
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
    T = 60  # number of control steps overall
    true_traj = []
    est_traj = []
    control_seq = []

    for t in range(T):
        x_true = rocket_true.get_state()
        true_traj.append(x_true.copy())
        est_traj.append(kf.x.copy())

        # Plan from current estimate x_hat to goal using iLQR
        u_plan, x_plan = ilqr(params, kf.x, x_goal, ilqr_cfg, rocket_model)
        u0 = float(u_plan[0])
        control_seq.append(u0)

        # Apply control to true rocket (with noise)
        rocket_true.transition(u0)

        # Get noisy observation of true state
        z = rocket_true.observe(sigma_meas=kf_cfg.meas_sigma)

        # KF predict + update
        kf.predict(u0, rocket_model, ilqr_cfg.eps_fd)
        kf.update(z)

    true_traj = np.array(true_traj)  # shape (T, 9)
    est_traj = np.array(est_traj)    # shape (T, 9)
    control_seq = np.array(control_seq)

    # --------------------
    # Plot results
    # --------------------
    t_axis = np.arange(T)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    fig.tight_layout(pad=2.0)

    # x vs y
    axs[0, 0].set_title("Position (x vs y)")
    axs[0, 0].plot(true_traj[:, 0], true_traj[:, 1], label="true")
    axs[0, 0].plot(est_traj[:, 0], est_traj[:, 1], "--", label="KF est")
    axs[0, 0].scatter([x_goal[0]], [x_goal[1]], marker="x", color="red", label="goal")
    axs[0, 0].legend()
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")

    # x over time
    axs[0, 1].set_title("x position over time")
    axs[0, 1].plot(t_axis, true_traj[:, 0], label="true")
    axs[0, 1].plot(t_axis, est_traj[:, 0], "--", label="KF est")
    axs[0, 1].axhline(x_goal[0], color="red", linestyle=":", label="goal")
    axs[0, 1].legend()
    axs[0, 1].set_xlabel("time step")
    axs[0, 1].set_ylabel("x")

    # yaw
    axs[1, 0].set_title("Yaw psi (deg)")
    axs[1, 0].plot(t_axis, np.degrees(true_traj[:, 4]), label="true")
    axs[1, 0].plot(t_axis, np.degrees(est_traj[:, 4]), "--", label="KF est")
    axs[1, 0].set_xlabel("time step")
    axs[1, 0].set_ylabel("psi (deg)")
    axs[1, 0].legend()

    # omega_C
    axs[1, 1].set_title("omega_C (deg/s)")
    axs[1, 1].plot(t_axis, np.degrees(true_traj[:, 7]), label="true")
    axs[1, 1].plot(t_axis, np.degrees(est_traj[:, 7]), "--", label="KF est")
    axs[1, 1].set_xlabel("time step")
    axs[1, 1].set_ylabel("omega_C (deg/s)")
    axs[1, 1].legend()

    # control
    axs[2, 0].set_title("Control alpha_C (deg/s^2)")
    axs[2, 0].plot(t_axis, np.degrees(control_seq))
    axs[2, 0].set_xlabel("time step")
    axs[2, 0].set_ylabel("alpha_C (deg/s^2)")

    # v_x
    axs[2, 1].set_title("v_x (m/s)")
    axs[2, 1].plot(t_axis, true_traj[:, 2], label="true")
    axs[2, 1].plot(t_axis, est_traj[:, 2], "--", label="KF est")
    axs[2, 1].set_xlabel("time step")
    axs[2, 1].set_ylabel("v_x")
    axs[2, 1].legend()

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
