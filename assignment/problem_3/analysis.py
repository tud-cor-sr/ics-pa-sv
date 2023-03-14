from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict


def plot_configuration_space_evolution(
    sim_ts: Dict[str, Array],
    filepath: str = None,
):
    fig, axes = plt.subplots(
        2, 1, num="Evolution in configuration-space", figsize=(6.4, 4.8)
    )

    plt.suptitle("Evolution in configuration-space")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot th1
    axes[0].plot(
        sim_ts["t_ts"],
        sim_ts["th_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$\theta_1$",
    )

    # set axis labels for first column
    # axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel(r"$\theta_1$ [rad]")

    # plot th_d1
    axes[1].plot(
        sim_ts["t_ts"],
        sim_ts["th_d_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$\dot{\theta}_1$",
    )

    # set axis labels for second column
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$\dot{\theta}_1$ [rad / s]")

    for ax in axes.flatten():
        ax.grid()

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()


def plot_operational_space_evolution(
    sim_ts: Dict[str, Array],
    filepath: str = None,
):
    fig, axes = plt.subplots(
        2, 1, num="Operational-space trajectory following", figsize=(6.4, 4.8)
    )

    plt.suptitle("Evolution in operational-space")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot x
    axes[0].plot(
        sim_ts["t_ts"],
        sim_ts["x_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$x_1$",
    )
    axes[0].plot(
        sim_ts["t_ts"],
        sim_ts["x_ts"][:, 1],
        color=colors[1],
        linewidth=3,
        label=r"$x_2$",
    )

    axes[0].legend(loc="upper right")

    # set axis labels for first column
    axes[0].set_ylabel(r"$x$ [m]")

    # plot x_d
    axes[1].plot(
        sim_ts["t_ts"],
        sim_ts["x_d_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$\dot{x}_1$",
    )
    axes[1].plot(
        sim_ts["t_ts"],
        sim_ts["x_d_ts"][:, 1],
        color=colors[1],
        linewidth=3,
        label=r"$\dot{x}_2$",
    )

    axes[1].legend(loc="upper right")

    # set axis labels for second column
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$\dot{x}$ [m / s]")

    for ax in axes.flatten():
        ax.grid()

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()
