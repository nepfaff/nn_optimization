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
        torch.Tensor: The scalar function value at the input coordinates.
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
        + torch.exp(torch.tensor([1.0], device=x.device))
    )
    return loss


def six_hump_camel(x: torch.Tensor) -> torch.Tensor:
    """Six-hump cammel function. See https://www.sfu.ca/~ssurjano/camel6.html.
    Global minima at (0.0898,-0.7126) and (-0.0898, 0.7126) with a value of -1.0316.

    Args:
        x (torch.Tensor): The input coordinates of shape (B, 2) where 2 is the input
        dimension.

    Returns:
        torch.Tensor: The scalar function value at the input coordinates.
    """
    loss = (
        4 * x[..., 0] ** 2
        - 2.1 * x[..., 0] ** 4
        + (x[..., 0] ** 6) / 3
        + x[..., 0] * x[..., 1]
        - 4 * x[..., 1] ** 2
        + 4 * x[..., 1] ** 4
    )
    return loss


def griewank(x: torch.Tensor) -> torch.Tensor:
    """Griewank function. See http://www.sfu.ca/~ssurjano/griewank.html.
    Gloabl minimum is at origin, with a value of 0. Lots of local minima away from the
    origin.

    Args:
        x (torch.Tensor): The input coordinates of shape (B, D) where D is the input
        dimension.

    Returns:
        torch.Tensor: The scalar function value at the input coordinates.
    """
    prod = torch.ones(x.shape[:-1])
    for i in range(x.shape[-1]):
        prod *= torch.cos(x[..., i] / torch.sqrt(torch.tensor([i + 1])))
    loss = torch.sum(x / 4000.0, dim=-1) - prod + 1
    return loss
