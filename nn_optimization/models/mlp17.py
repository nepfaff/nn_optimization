import torch
import torch.nn as nn


class MLP17(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_ch: int,
    ):
        super().__init__()

        print(f"[MLP-17] in_ch: {in_ch}; hidden_ch: {hidden_ch}")

        self.latent = nn.Parameter(torch.rand((1, in_ch)))

        self.net1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(hidden_ch + in_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
        )

        self.net3 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(2 * hidden_ch + in_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
        )

        self.net4 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(3 * hidden_ch + in_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
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
        in3 = torch.cat([out2, in2], dim=-1)
        out3 = self.net3(in3)
        in4 = torch.cat([out3, in3], dim=-1)
        out4 = self.net4(in4)
        return out4
