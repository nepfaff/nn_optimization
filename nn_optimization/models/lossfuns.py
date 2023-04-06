import torch


def qubic_1d(x: torch.Tensor) -> torch.Tensor:
    """1D Qubic with a global minimum at -1.036 (loss of -1.305) and local minimum at 0.96 (loss of -0.706)."""
    func = lambda x: x**4 - 2 * x**2 + 0.3 * x
    loss = func(x)
    return loss
