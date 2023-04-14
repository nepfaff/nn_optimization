import torch
import torch.nn as nn


class MLP9(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_ch: int,
    ):
        super().__init__()

        print(f"[MLP-9] in_ch: {in_ch}; hidden_ch: {hidden_ch}")

        self.latent = nn.Parameter(torch.rand((1, in_ch)))

        self.net1 = nn.Sequential(
            nn.Linear(in_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, hidden_ch - in_ch),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, out_ch),
        )

        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self):
        in1 = self.latent
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        return out2
