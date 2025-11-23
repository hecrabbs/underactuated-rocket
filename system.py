import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from helpers import new_col_vec
from state import RocketParams, RocketState


def main(verbosity=logging.WARNING):
    params = RocketParams(
        m_B=100,
        m_C=10,
        m_fuel=10,
        mag_d_0=0.5,
        mag_d_1=0.25,
        sigma_thrust=0.1,
        F_thrust_nominal=2000,
        r_B_to_thrust_B=new_col_vec(0.0, -0.5)
    )

    state = RocketState(params, verbosity=verbosity)
    inputs = [0,0.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    x_coords = []
    y_coords = []
    psi_deg = []
    for input in inputs:
        x_coords.append(state.p[0][0])
        y_coords.append(state.p[1][0])
        psi_deg.append(np.degrees(state.psi))
        state.transition(input)

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
