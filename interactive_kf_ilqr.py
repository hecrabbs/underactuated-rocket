import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

from state import RocketParams
from kf_ilqr import (
    ILQRConfig,
    KFConfig,
    FullStateKF,
    make_model_and_true_rockets,
    ilqr,
)

logging.basicConfig(level=logging.WARNING)


def create_interactive_fig():
    # ---- Defaults ----
    T_max_default = 60
    m_B_default = 100.0
    m_C_default = 1.2
    F_thrust_default = 1500.0
    x_goal_default = 100.0
    y_goal_default = 100.0
    N_default = 80

    kf_cfg = KFConfig(process_sigma=0.1, meas_sigma=0.05)
    ilqr_cfg = ILQRConfig(
        N=N_default,
        max_iter=20,
        tol=1e-3,
        reg=1e-4,
        u_min=-2.0,
        u_max=2.0,
        eps_fd=1e-4,
    )

    def make_params(m_C, F_thrust):
        return RocketParams(
            m_B=m_B_default,
            m_C=m_C,
            m_fuel=10.0,
            m_fuel_rate=0.0,
            height=1.0,
            radius=0.1,
            mag_d_0=10.0,
            mag_d_1=0.1,
            sigma_thrust=0.1,
            F_thrust_nominal=F_thrust,
            mag_a_g=9.81,
            seed=123456,
        )

    # ---- Mutable simulation state ----
    sim = {
        "params": None,
        "rocket_true": None,
        "rocket_model": None,
        "kf": None,
        "x_goal": None,
        "T_max": T_max_default,
        "t": 0,
        "true_traj": [],
        "est_traj": [],
        "u_seq": [],
    }

    def reset_sim(m_C, F_thrust, xg, yg):
        """Reset rocket, KF, trajectories using current slider params."""
        params = make_params(m_C, F_thrust)
        rocket_true, rocket_model = make_model_and_true_rockets(params)

        x0 = rocket_true.get_state()
        x_goal = x0.copy()
        x_goal[0] = xg
        x_goal[1] = yg
        x_goal[2] = 0.0
        x_goal[4] = 0.0
        x_goal[5] = 0.0

        x_hat0 = x0.copy()
        P0 = np.eye(len(x0)) * 1.0
        kf = FullStateKF(x_hat0, P0, kf_cfg)

        sim["params"] = params
        sim["rocket_true"] = rocket_true
        sim["rocket_model"] = rocket_model
        sim["kf"] = kf
        sim["x_goal"] = x_goal
        sim["t"] = 0
        sim["true_traj"] = []
        sim["est_traj"] = []
        sim["u_seq"] = []

    # initial sim
    reset_sim(m_C_default, F_thrust_default, x_goal_default, y_goal_default)

    # ---- Figure + axes ----
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    fig.subplots_adjust(left=0.1, bottom=0.3, hspace=0.5, wspace=0.4)

    ax_xy = axs[0, 0]
    ax_x_t = axs[0, 1]
    ax_psi = axs[1, 0]
    ax_omegaC = axs[1, 1]
    ax_u = axs[2, 0]
    ax_vx = axs[2, 1]

    # x vs y
    ax_xy.set_title("Position (x vs y)")
    true_xy_line, = ax_xy.plot([], [], label="true")
    est_xy_line, = ax_xy.plot([], [], "--", label="KF est")
    goal_scatter = ax_xy.scatter([sim["x_goal"][0]], [sim["x_goal"][1]],
                                 marker="x", color="red", label="goal")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.legend()

    # x over time
    ax_x_t.set_title("x position over time")
    true_x_line, = ax_x_t.plot([], [], label="true")
    est_x_line, = ax_x_t.plot([], [], "--", label="KF est")
    goal_x_line = ax_x_t.axhline(sim["x_goal"][0], color="red", linestyle=":", label="goal")
    ax_x_t.set_xlabel("time step")
    ax_x_t.set_ylabel("x")
    ax_x_t.legend()

    # yaw
    ax_psi.set_title("Yaw psi (deg)")
    true_psi_line, = ax_psi.plot([], [], label="true")
    est_psi_line, = ax_psi.plot([], [], "--", label="KF est")
    ax_psi.set_xlabel("time step")
    ax_psi.set_ylabel("psi (deg)")
    ax_psi.legend()

    # omega_C
    ax_omegaC.set_title("omega_C (deg/s)")
    true_omegaC_line, = ax_omegaC.plot([], [], label="true")
    est_omegaC_line, = ax_omegaC.plot([], [], "--", label="KF est")
    ax_omegaC.set_xlabel("time step")
    ax_omegaC.set_ylabel("omega_C (deg/s)")
    ax_omegaC.legend()

    # control
    ax_u.set_title("Control alpha_C (deg/s^2)")
    u_line, = ax_u.plot([], [])
    ax_u.set_xlabel("time step")
    ax_u.set_ylabel("alpha_C (deg/s^2)")

    # v_x
    ax_vx.set_title("v_x (m/s)")
    true_vx_line, = ax_vx.plot([], [], label="true")
    est_vx_line, = ax_vx.plot([], [], "--", label="KF est")
    ax_vx.set_xlabel("time step")
    ax_vx.set_ylabel("v_x")
    ax_vx.legend()

    # pause state
    is_paused = {"val": False}

    # ---- Sliders & buttons ----
    ax_mC = plt.axes([0.1, 0.18, 0.3, 0.02])
    ax_Fthrust = plt.axes([0.1, 0.14, 0.3, 0.02])
    ax_xgoal = plt.axes([0.1, 0.10, 0.3, 0.02])
    ax_ygoal = plt.axes([0.1, 0.06, 0.3, 0.02])
    ax_N = plt.axes([0.55, 0.18, 0.3, 0.02])

    slider_mC = Slider(ax_mC, 'm_C', 0.5, 5.0, valinit=m_C_default)
    slider_Fthrust = Slider(ax_Fthrust, 'F_thrust', 500.0, 3000.0, valinit=F_thrust_default)
    slider_xgoal = Slider(ax_xgoal, 'x_goal', -200.0, 200.0, valinit=x_goal_default)
    slider_ygoal = Slider(ax_ygoal, 'y_goal', 0.0, 200.0, valinit=y_goal_default)
    slider_N = Slider(ax_N, 'N (horizon)', 20, 120, valinit=N_default, valstep=5)

    ax_btn_reset = plt.axes([0.55, 0.10, 0.12, 0.05])
    btn_reset = Button(ax_btn_reset, 'Reset')

    ax_btn_pause = plt.axes([0.72, 0.10, 0.12, 0.05])
    btn_pause = Button(ax_btn_pause, 'Pause/Play')

    # ---- Button / slider callbacks ----
    def on_reset(event):
        reset_sim(
            slider_mC.val,
            slider_Fthrust.val,
            slider_xgoal.val,
            slider_ygoal.val,
        )
        # update goal visuals
        goal_scatter.set_offsets([[sim["x_goal"][0], sim["x_goal"][1]]])
        goal_x_line.set_ydata([sim["x_goal"][0], sim["x_goal"][0]])

        # clear lines
        true_xy_line.set_data([], [])
        est_xy_line.set_data([], [])
        true_x_line.set_data([], [])
        est_x_line.set_data([], [])
        true_psi_line.set_data([], [])
        est_psi_line.set_data([], [])
        true_omegaC_line.set_data([], [])
        est_omegaC_line.set_data([], [])
        u_line.set_data([], [])
        true_vx_line.set_data([], [])
        est_vx_line.set_data([], [])

        fig.canvas.draw_idle()

    def on_pause(event):
        is_paused["val"] = not is_paused["val"]

    def on_goal_change(val):
        # update goal in sim; controller will respond on next steps
        sim["x_goal"][0] = slider_xgoal.val
        sim["x_goal"][1] = slider_ygoal.val
        goal_scatter.set_offsets([[sim["x_goal"][0], sim["x_goal"][1]]])
        goal_x_line.set_ydata([sim["x_goal"][0], sim["x_goal"][0]])
        fig.canvas.draw_idle()

    def on_horizon_change(val):
        ilqr_cfg.N = int(slider_N.val)

    slider_xgoal.on_changed(on_goal_change)
    slider_ygoal.on_changed(on_goal_change)
    slider_N.on_changed(on_horizon_change)
    btn_reset.on_clicked(on_reset)
    btn_pause.on_clicked(on_pause)

    # m_C and F_thrust are applied on Reset only (to avoid constant re-instantiation)
    def on_param_change(val):
        pass

    slider_mC.on_changed(on_param_change)
    slider_Fthrust.on_changed(on_param_change)

    # ---- FuncAnimation: one MPC step per frame ----
    def update_frame(frame_idx):
        if is_paused["val"]:
            return ()

        if sim["t"] >= sim["T_max"]:
            return ()

        rocket_true = sim["rocket_true"]
        rocket_model = sim["rocket_model"]
        kf = sim["kf"]
        x_goal = sim["x_goal"]

        # log state
        x_true = rocket_true.get_state()
        sim["true_traj"].append(x_true.copy())
        sim["est_traj"].append(kf.x.copy())

        # plan from current estimate
        u_plan, x_plan = ilqr(sim["params"], kf.x, x_goal, ilqr_cfg, rocket_model)
        u0 = float(u_plan[0])
        sim["u_seq"].append(u0)

        # apply to true rocket + KF
        rocket_true.transition(u0)
        z = rocket_true.observe(sigma_meas=kf_cfg.meas_sigma)
        kf.predict(u0, rocket_model, ilqr_cfg.eps_fd)
        kf.update(z)

        sim["t"] += 1

        # convert trajectory to arrays
        true_arr = np.array(sim["true_traj"])
        est_arr = np.array(sim["est_traj"])
        u_arr = np.array(sim["u_seq"])
        t_axis = np.arange(sim["t"])

        if true_arr.shape[0] > 0:
            # basic autoscaling
            ax_xy.set_xlim(np.min(true_arr[:, 0]) - 50, np.max(true_arr[:, 0]) + 50)
            ax_xy.set_ylim(0, max(np.max(true_arr[:, 1]) + 50, 10))

            ax_x_t.set_xlim(0, sim["T_max"])
            ax_x_t.set_ylim(np.min(true_arr[:, 0]) - 50, np.max(true_arr[:, 0]) + 50)

            ax_psi.set_xlim(0, sim["T_max"])
            ax_omegaC.set_xlim(0, sim["T_max"])
            ax_u.set_xlim(0, sim["T_max"])
            ax_vx.set_xlim(0, sim["T_max"])

            # x vs y
            true_xy_line.set_data(true_arr[:, 0], true_arr[:, 1])
            est_xy_line.set_data(est_arr[:, 0], est_arr[:, 1])

            # x(t)
            true_x_line.set_data(t_axis, true_arr[:, 0])
            est_x_line.set_data(t_axis, est_arr[:, 0])

            # psi(t)
            true_psi_line.set_data(t_axis, np.degrees(true_arr[:, 4]))
            est_psi_line.set_data(t_axis, np.degrees(est_arr[:, 4]))

            # omega_C(t)
            true_omegaC_line.set_data(t_axis, np.degrees(true_arr[:, 7]))
            est_omegaC_line.set_data(t_axis, np.degrees(est_arr[:, 7]))

            # control
            u_line.set_data(t_axis, np.degrees(u_arr))

            # v_x
            true_vx_line.set_data(t_axis, true_arr[:, 2])
            est_vx_line.set_data(t_axis, est_arr[:, 2])

        return (
            true_xy_line, est_xy_line,
            true_x_line, est_x_line,
            true_psi_line, est_psi_line,
            true_omegaC_line, est_omegaC_line,
            u_line, true_vx_line, est_vx_line,
        )

    anim = FuncAnimation(
        fig,
        update_frame,
        frames=10_000,   # plenty; we stop internally at T_max
        interval=60,     # ms between frames
        blit=False,
        repeat=False,
    )

    plt.show()


if __name__ == "__main__":
    create_interactive_fig()
