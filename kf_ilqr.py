# kf_ilqr.py
import math
from dataclasses import dataclass

import numpy as np

from state import RocketParams, RocketState


@dataclass
class ILQRConfig:
    N: int = 40           # horizon length (time steps)
    max_iter: int = 20    # iLQR iterations
    tol: float = 1e-3     # cost improvement tolerance
    reg: float = 1e-4     # regularization on Q_uu
    u_min: float = -1.0   # min alpha_C
    u_max: float = 1.0    # max alpha_C
    eps_fd: float = 1e-4  # finite-difference step


def make_model_and_true_rockets(params: RocketParams):
    """
    True rocket uses process noise (thrust noise) as defined in params.
    Model rocket is deterministic (sigma_thrust=0) for prediction and iLQR.
    """
    rocket_true = RocketState(params=params, logger_name="rocket_true")
    # Copy params but zero out thrust noise for the model
    model_params = RocketParams(
        m_B=params.m_B,
        m_C=params.m_C,
        m_fuel=params.m_fuel,
        m_fuel_rate=params.m_fuel_rate,
        height=params.height,
        radius=params.radius,
        mag_d_0=params.mag_d_0,
        mag_d_1=params.mag_d_1,
        sigma_thrust=0.0,  # <-- deterministic model
        F_thrust_nominal=params.F_thrust_nominal,
        mag_a_g=params.mag_a_g,
        seed=params.seed,
    )
    rocket_model = RocketState(params=model_params, logger_name="rocket_model")
    return rocket_true, rocket_model


def dynamics_f(x: np.ndarray, u: float, rocket_model: RocketState) -> np.ndarray:
    """
    x_{k+1} = f(x_k, u_k) using the noise-free model rocket.
    """
    rocket_model.set_state(x)
    rocket_model.transition(u)
    return rocket_model.get_state()


def linearize_dynamics(x: np.ndarray, u: float,
                       rocket_model: RocketState,
                       eps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference Jacobians A = df/dx, B = df/du around (x,u).
    """
    n_x = x.shape[0]
    A = np.zeros((n_x, n_x))
    B = np.zeros((n_x, 1))

    fxu = dynamics_f(x, u, rocket_model)

    # A: df/dx
    for i in range(n_x):
        x_pert = x.copy()
        x_pert[i] += eps
        f_pert = dynamics_f(x_pert, u, rocket_model)
        A[:, i] = (f_pert - fxu) / eps

    # B: df/du
    u_pert = u + eps
    f_pert = dynamics_f(x, u_pert, rocket_model)
    B[:, 0] = (f_pert - fxu) / eps

    return A, B


# -----------------
# Kalman Filter
# -----------------

@dataclass
class KFConfig:
    process_sigma: float = 0.1   # process noise std (per state dim)
    meas_sigma: float = 0.05     # measurement noise std (per state dim)


class FullStateKF:
    """
    Linear Kalman filter on the full state, with:

        x_{k+1} = f(x_k, u_k) + w_k
        z_k     = x_k + v_k

    where w ~ N(0, Q), v ~ N(0, R).
    The nonlinearity is handled by linearizing f at each step (EKF-style),
    but since H = I, the update is just linear KF math.
    """
    def __init__(self, x0: np.ndarray, P0: np.ndarray, kf_cfg: KFConfig):
        self.x = x0.copy()
        self.P = P0.copy()
        self.kf_cfg = kf_cfg

    def predict(self, u: float, rocket_model: RocketState, eps_fd: float):
        n_x = self.x.shape[0]
        Q = (self.kf_cfg.process_sigma ** 2) * np.eye(n_x)

        # Linearize around current estimate
        A, B = linearize_dynamics(self.x, u, rocket_model, eps_fd)

        # Predict state and covariance
        x_pred = dynamics_f(self.x, u, rocket_model)
        P_pred = A @ self.P @ A.T + Q

        self.x = x_pred
        self.P = P_pred

    def update(self, z: np.ndarray):
        n_x = self.x.shape[0]
        R = (self.kf_cfg.meas_sigma ** 2) * np.eye(n_x)
        H = np.eye(n_x)  # full-state measurement

        y = z - H @ self.x            # innovation
        S = H @ self.P @ H.T + R      # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(n_x)
        self.P = (I - K @ H) @ self.P


# -----------------
# iLQR
# -----------------

def ilqr(params: RocketParams,
         x0: np.ndarray,
         x_goal: np.ndarray,
         config: ILQRConfig,
         rocket_model: RocketState) -> tuple[np.ndarray, np.ndarray]:
    """
    iLQR over the RocketState dynamics.

    Returns:
        u_seq: shape (N-1,)
        x_seq: shape (N, n_x)
    """
    n_x = x0.shape[0]
    n_u = 1

    # Cost weights:
    # (You can tweak these numbers.)
    Q = np.diag([
        10.0,    # p_x
        10.0,    # p_y
        5.0,  # v_x
        5.0,    # v_y
        328281.0,   # psi
        32828064.0, # omega
        0.0,    # theta_C
        1013.0, # omega_C
        0.0,    # m_fuel
    ])
    Qf = 10.0 * Q
    R = 0.01  # small penalty on control effort

    def rollout(u_seq_local: np.ndarray) -> tuple[np.ndarray, float]:
        x_seq = np.zeros((config.N, n_x))
        x_seq[0] = x0
        cost = 0.0

        for k in range(config.N - 1):
            x = x_seq[k]
            u = float(u_seq_local[k])

            dx = x - x_goal
            stage_cost = dx @ Q @ dx + R * u**2

            # Soft ground penalty
            y = x[1]
            if y < 0.0:
                stage_cost += 1e6 * (abs(y) ** 2)  # huge penalty underground

            cost += stage_cost

            x_next = dynamics_f(x, u, rocket_model)
            x_seq[k + 1] = x_next

        dxN = x_seq[-1] - x_goal
        terminal_cost = dxN @ Qf @ dxN
        yN = x_seq[-1][1]
        if yN < 0.0:
            terminal_cost += 1e6 * (abs(yN) ** 2)

        cost += terminal_cost
        return x_seq, cost


    # Initial control sequence (zero)
    u_seq = np.zeros(config.N - 1)
    x_seq, J = rollout(u_seq)

    for _ in range(config.max_iter):
        # Linearize along current trajectory
        A_list = []
        B_list = []
        for k in range(config.N - 1):
            A_k, B_k = linearize_dynamics(x_seq[k], float(u_seq[k]),
                                          rocket_model, config.eps_fd)
            A_list.append(A_k)
            B_list.append(B_k)

        # Backward pass
        V_x = 2.0 * Qf @ (x_seq[-1] - x_goal)
        V_xx = 2.0 * Qf
        K_list = []
        k_list = []

        for k in reversed(range(config.N - 1)):
            x = x_seq[k]
            u = float(u_seq[k])
            dx = x - x_goal

            l_x = 2.0 * Q @ dx
            l_u = 2.0 * R * u
            l_xx = 2.0 * Q
            l_uu = 2.0 * R
            l_xu = np.zeros((n_x, 1))

            A_k = A_list[k]
            B_k = B_list[k]

            Q_x = l_x + A_k.T @ V_x
            Q_u = l_u + float(B_k.T @ V_x)

            Q_xx = l_xx + A_k.T @ V_xx @ A_k
            Q_uu = l_uu + float(B_k.T @ V_xx @ B_k)
            Q_xu = l_xu + A_k.T @ V_xx @ B_k

            Q_uu_reg = Q_uu + config.reg
            inv_Q_uu = 1.0 / Q_uu_reg

            K_k = -inv_Q_uu * Q_xu.T  # (1, n_x)
            k_k = -inv_Q_uu * Q_u     # scalar

            K_list.insert(0, K_k)
            k_list.insert(0, k_k)

            V_x = Q_x \
                  + (K_k.T * Q_uu * k_k).reshape(-1) \
                  + (K_k.T * Q_u).reshape(-1) \
                  + (Q_xu * k_k).reshape(-1)

            V_xx = Q_xx \
                + (K_k.T * Q_uu) @ K_k \
                + K_k.T @ Q_xu.T \
                + Q_xu @ K_k
            V_xx = 0.5 * (V_xx + V_xx.T)

        # Forward line search
        alphas = [1.0, 0.5, 0.25, 0.1]
        best_J = np.inf
        best_u = None
        best_x = None

        for alpha in alphas:
            x_new = np.zeros_like(x_seq)
            u_new = np.zeros_like(u_seq)
            x_new[0] = x0
            J_new = 0.0

            for k in range(config.N - 1):
                dx = x_new[k] - x_seq[k]
                du = alpha * k_list[k] + float(K_list[k] @ dx)
                u_new[k] = np.clip(u_seq[k] + du, config.u_min, config.u_max)

                dx_cost = x_new[k] - x_goal
                J_new += dx_cost @ Q @ dx_cost + R * u_new[k]**2

                x_next = dynamics_f(x_new[k], float(u_new[k]), rocket_model)
                x_new[k + 1] = x_next

            dxN = x_new[-1] - x_goal
            J_new += dxN @ Qf @ dxN

            if J_new < best_J:
                best_J = J_new
                best_u = u_new
                best_x = x_new

        dJ = J - best_J
        u_seq = best_u
        x_seq = best_x
        J = best_J

        if abs(dJ) < config.tol:
            break

    return u_seq, x_seq
