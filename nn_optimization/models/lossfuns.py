import torch


def qubic_1d(x: torch.Tensor) -> torch.Tensor:
    """1D Qubic with a global minimum at -1.036 (loss of -1.305) and local minimum at
    0.96 (loss of -0.706).
    """
    loss = x**4 - 2 * x**2 + 0.3 * x
    return loss


def ackley(x: torch.Tensor) -> torch.Tensor:
    """Ackley function. See http://www.sfu.ca/~ssurjano/ackley.html.
    Gloabl minimum is at origin, with a value of 0. Lots of local minima away from the
    origin.

    Args:
        x (torch.Tensor): The input coordinates of shape (B, D) where D is the input
        dimension.

    Returns:
        torch.Tensor: The function value at the input coordinates.
    """
    # Recommended params
    a = 20
    b = 0.2
    c = 2 * torch.pi

    dim = x.shape[-1]
    loss = (
        -a * torch.exp(-b * torch.sqrt(1 / dim * torch.sum(x**2, dim=-1)))
        - torch.exp(1 / dim * torch.sum(torch.cos(c * x), dim=-1))
        + a
        + torch.exp(torch.tensor([1.0]))
    )
    return loss
