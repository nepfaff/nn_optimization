import pathlib
import time

import torch
import hydra
from omegaconf import OmegaConf, open_dict
import wandb
from tqdm import tqdm

from nn_optimization.trainers import Trainer


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("../..", "config")),
)
def main(cfg: OmegaConf):
    # Add log dir to config
    with open_dict(cfg):
        cfg.log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print("Config:\n", cfg)

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="nn_optimization", name=cfg.log_dir, config=wandb_config)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    trainer = Trainer(cfg, device)

    if cfg.trainer.resume:
        start_epoch = trainer.resume(cfg.trainer.resume_dir)
    else:
        start_epoch = 0

        # Initialization loop
        if cfg.trainer.initialization.initialize:
            trainer.train()
            target = torch.tensor([cfg.trainer.initialization.init_condition])
            init_start_time = time.time()
            while True:
                logs_info = trainer.step_initialization(target)
                wandb.log(logs_info)

                if logs_info["initialization_loss"] < 1e-6:
                    print("Initialized successfully.")
                    break
                elif (
                    time.time() - init_start_time > cfg.trainer.initialization.timeout_s
                ):
                    print("Initialization timeout reached!")
                    break

        # Save initial condition
        trainer.save(epoch=0)

    # Main training loop
    print("[Train] Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs))
    for epoch in tqdm(range(start_epoch, cfg.trainer.epochs)):
        trainer.epoch_start()

        logs_info = trainer.step()
        wandb.log(logs_info)

        # Save checkpoints
        if (epoch + 1) % int(cfg.logs.save_interval) == 0:
            trainer.save(epoch=epoch)

    # Always save last epoch
    if (epoch + 1) % int(cfg.logs.save_interval) != 0:
        trainer.save(epoch=epoch)


if __name__ == "__main__":
    main()
