import logging
import math
import warnings
from typing import Union

import hydra
import torch
import transformers.utils.logging as hf_logging
import wandb
from lightning import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from omegaconf import DictConfig, ListConfig

from datamodule.datamodule import DataModule

hf_logging.set_verbosity(hf_logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"It is recommended to use .+ when logging on epoch level in distributed setting to accumulate the metric"
    r" across devices",
    category=PossibleUserWarning,
)
logging.getLogger("torch").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig):
    if isinstance(cfg.devices, str):
        cfg.devices = list(map(int, cfg.devices.split(","))) if "," in cfg.devices else [int(cfg.devices)]
    if isinstance(cfg.max_batches_per_device, str):
        cfg.max_batches_per_device = int(cfg.max_batches_per_device)
    if isinstance(cfg.num_workers, str):
        cfg.num_workers = int(cfg.num_workers)
    cfg.seed = seed_everything(seed=cfg.seed, workers=True)

    logger: Union[Logger, bool] = cfg.get("logger", False) and hydra.utils.instantiate(cfg.get("logger"))
    callbacks: list[Callback] = list(map(hydra.utils.instantiate, cfg.get("callbacks", {}).values()))

    # Calculate gradient_accumulation_steps assuming DDP
    num_devices: int = 1
    if isinstance(cfg.devices, (list, ListConfig)):
        num_devices = len(cfg.devices)
    elif isinstance(cfg.devices, int):
        num_devices = cfg.devices
    cfg.trainer.accumulate_grad_batches = math.ceil(
        cfg.effective_batch_size / (cfg.max_batches_per_device * num_devices)
    )
    batches_per_device = cfg.effective_batch_size // (num_devices * cfg.trainer.accumulate_grad_batches)
    # if effective_batch_size % (accumulate_grad_batches * num_devices) != 0, then
    # cause an error of at most accumulate_grad_batches * num_devices compared in effective_batch_size
    # otherwise, no error
    cfg.effective_batch_size = batches_per_device * num_devices * cfg.trainer.accumulate_grad_batches
    cfg.datamodule.batch_size = batches_per_device

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    datamodule = DataModule(cfg=cfg.datamodule)

    model: LightningModule = hydra.utils.instantiate(cfg.module.cls, hparams=cfg, _recursive_=False)
    if cfg.compile is True:
        model = torch.compile(model)

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best" if not trainer.fast_dev_run else None)

    wandb.finish()


if __name__ == "__main__":
    main()
