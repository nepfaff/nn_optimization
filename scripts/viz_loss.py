"""Script for visualizing 1D or 2D loss functions."""

import argparse

import torch
import matplotlib.pyplot as plt

from nn_optimization.models import qubic_1d, ackley, six_hump_camel, griewank

# Mapping from name to (func, is multi-dimensional)
loss_name_to_func_dict = {
    "qubic_1d": (qubic_1d, False),
    "ackley": (ackley, True),
    "six_hump_camel": (six_hump_camel, True),
    "griewank": (griewank, True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function_name",
        required=True,
        type=str,
        help="The name of the function to plot.",
        choices=loss_name_to_func_dict.keys(),
    )
    parser.add_argument(
        "--range",
        required=True,
        type=float,
        help="The range of each domain dimension [-value, value].",
    )
    parser.add_argument(
        "--num_samples",
        default=1000,
        type=int,
        help="The number of samples per dimension.",
    )
    parser.add_argument(
        "--one_dim",
        action="store_true",
        help="Whether to plot a 1D version of a function that allows for more than one "
        + "dimension (these functions are plotted in 2D by default).",
    )
    args = parser.parse_args()
    function_name = args.function_name

    loss_func, is_2d = loss_name_to_func_dict[function_name]
    if args.one_dim:
        is_2d = False

    lim = args.range
    num_samples = args.num_samples
    if is_2d:
        X, Y = torch.meshgrid(
            torch.linspace(-lim, lim, num_samples),
            torch.linspace(-lim, lim, num_samples),
        )
        Z = loss_func(torch.stack([X, Y], dim=-1).unsqueeze(0))[0]

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")
        plot = ax.plot_surface(X, Y, Z)
        plt.title(function_name)
        plt.show()
    else:
        x = torch.linspace(-lim, lim, num_samples).unsqueeze(-1)
        loss = loss_func(x.unsqueeze(0))[0]
        fig = plt.figure(figsize=(10, 10))
        plot = plt.plot(x, loss)
        plt.title(function_name)
        plt.show()


if __name__ == "__main__":
    main()
