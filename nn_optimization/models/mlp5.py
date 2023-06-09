import torch
import torch.nn as nn


class MLP5(nn.Module):
    def __init__(
        self,
        use_dropout: bool,
        dropout_prob: float,
        in_ch: int,
        out_ch: int,
        hidden_ch: int,
    ):
        super().__init__()

        print(
            f"[MLP-5] Dropout: {use_dropout}; Do_prob: {dropout_prob}; in_ch: {in_ch}; hidden_ch: {hidden_ch}"
        )

        self._latent = nn.Parameter(torch.rand((1, in_ch)))
        if use_dropout is False:
            self._net = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, hidden_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_ch, out_ch),
            )
        else:
            self._net = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, hidden_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(hidden_ch, hidden_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.Linear(hidden_ch, out_ch),
            )

        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self):
        input = self._latent
        out = self._net(input)
        return out
