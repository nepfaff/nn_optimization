import os
from typing import Iterator

import torch
import torch.nn as nn
import wandb
from hydra.utils import instantiate, call
from omegaconf import OmegaConf


class Trainer(nn.Module):
    def __init__(self, cfg: OmegaConf, device: torch.device):
        super().__init__()
        self._cfg = cfg
        self._device = device

        self._model: nn.Module = instantiate(cfg.model)
        self._model.to(self._device)
        print("Model:")
        print(self._model)

        self._lossfun = cfg.lossfun

        self._optim = self._get_optim(self._model.parameters(), self._cfg.trainer.optim)

        wandb.watch(self._model, log="all", log_graph=False, log_freq=100)

    def _get_optim(self, parameters: Iterator[nn.Parameter], cfg: OmegaConf):
        if cfg.type.lower() == "adam":
            optim = torch.optim.Adam(
                parameters,
                lr=cfg.lr,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
                amsgrad=False,
            )
        elif cfg.type.lower() == "sgd":
            optim = torch.optim.SGD(
                parameters,
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise NotImplementedError("Unknow optimizer: {}".format(cfg.type))

        return optim

    def epoch_start(self):
        self.train()

    def step_initialization(self, target):
        """Optimize model weights to make output equal to `target`."""
        self._optim.zero_grad()

        pred = self._model()
        loss = torch.mean((pred - target) ** 2)  # MSE
        loss.backward()

        self._optim.step()

        log_info = {
            "initialization_loss": loss.item(),
            "init_prediction": pred.data,
        }
        return log_info

    def step(self):
        self._optim.zero_grad()

        pred = self._model()
        loss = call(self._lossfun, pred)

        loss.backward()

        self._optim.step()

        log_info = {"loss": loss.item(), "prediction": pred.data}
        return log_info

    def save(self, epoch):
        save_dir = os.path.join(self._cfg.log_dir, "checkpoints")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name = f"epoch_{epoch}.pth"
        path = os.path.join(save_dir, save_name)
        torch.save(
            {
                "trainer_state_dict": self.state_dict(),
                "optim_state_dict": self._optim.state_dict(),
                "epoch": epoch,
            },
            path,
        )

    def resume(self, ckpt_path: str) -> int:
        print(f"Resuming {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=self._device)
        self.load_state_dict(ckpt["trainer_state_dict"], strict=False)
        if "optim_state_dict" in ckpt.keys():
            self._optim.load_state_dict(ckpt["optim_state_dict"])
        else:
            ckpt["epoch"] = 9999
        return ckpt["epoch"]
