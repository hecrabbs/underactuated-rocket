import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from underactuated_rocket.helpers import new_col_vec
from underactuated_rocket.state import RocketParams, RocketState


def main(verbosity=logging.WARNING):
    # Logging
    logger = logging.getLogger(__name__)
    logger.setLevel(verbosity)
    handler = logging.StreamHandler()
    handler.setLevel(verbosity)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    params = RocketParams(
        m_B=1,
        m_C=1,
        m_fuel=0,
        m_fuel_rate=0,
        height=1,
        radius=0.1,
        mag_d_0=10,
        mag_d_1=10,
        sigma_thrust=0.1,
        F_thrust_nominal=0,
        mag_a_g=0,
        seed=123456
    )

    state = RocketState(params, theta_C=0, logger_name=__name__)
    ang = np.radians(0.09)
    inputs = [0] + [ang]*100 + [0] + [-ang]*100 + [0, 0]
    print(inputs)
    x_coords = []
    y_coords = []
    psi_deg = []
    for input in inputs:
        x_coords.append(state.p[0][0])
        y_coords.append(state.p[1][0])
        psi_deg.append(np.degrees(state.psi))
        state.transition(input)
        print("CM IN I:", state.p + state._r_CM_B(state._m_total(), state._r_C_B(state._d_1_B())))
        print()
        print()
        print()

    fig,axs = plt.subplots(2)
    axs[0].plot(x_coords, y_coords)
    axs[0].set_xlim(-100,100)
    axs[1].plot(psi_deg)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--verbosity", type=str, default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    args = parser.parse_args()

    main(args.verbosity)
