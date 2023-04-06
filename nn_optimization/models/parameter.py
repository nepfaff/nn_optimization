import torch
import torch.nn as nn


class Parameter(nn.Module):
    def __init__(self, out_ch: int):
        """Used fopr direct optimization (not indirectly by optimizing model weights)."""
        super().__init__()

        self._param = nn.Parameter(torch.randn((1, out_ch)) * 0.01)

        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self):
        out = self._param
        return out
