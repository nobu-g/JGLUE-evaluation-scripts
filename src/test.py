import logging
import warnings
from typing import Union

import hydra
import torch
import transformers.utils.logging as hf_logging
from dotenv import load_dotenv
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from omegaconf import DictConfig, ListConfig, OmegaConf

from datamodule.datamodule import DataModule

hf_logging.set_verbosity(hf_logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"It is recommended to use .+ when logging on epoch level in distributed setting to accumulate the metric"
    r" across devices",
    category=PossibleUserWarning,
)
logging.getLogger("torch").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(eval_cfg: DictConfig):
    load_dotenv()
    if isinstance(eval_cfg.devices, str):
        try:
            eval_cfg.devices = [int(x) for x in eval_cfg.devices.split(",")]
        except ValueError:
            eval_cfg.devices = None
    if isinstance(eval_cfg.max_batches_per_device, str):
        eval_cfg.max_batches_per_device = int(eval_cfg.max_batches_per_device)
    if isinstance(eval_cfg.num_workers, str):
        eval_cfg.num_workers = int(eval_cfg.num_workers)

    # Load saved model and configs
    model: LightningModule = hydra.utils.call(eval_cfg.module.load_from_checkpoint, _recursive_=False)
    if eval_cfg.compile is True:
        model = torch.compile(model)

    train_cfg: DictConfig = model.hparams
    OmegaConf.set_struct(train_cfg, False)  # enable to add new key-value pairs
    cfg = OmegaConf.merge(train_cfg, eval_cfg)
    assert isinstance(cfg, DictConfig)

    logger: Union[Logger, bool] = cfg.get("logger", False) and hydra.utils.instantiate(cfg.get("logger"))
    callbacks: list[Callback] = list(map(hydra.utils.instantiate, cfg.get("callbacks", {}).values()))

    num_devices: int = 1
    if isinstance(cfg.devices, (list, ListConfig)):
        num_devices = len(cfg.devices)
    elif isinstance(cfg.devices, int):
        num_devices = cfg.devices
    cfg.effective_batch_size = cfg.max_batches_per_device * num_devices
    cfg.datamodule.batch_size = cfg.max_batches_per_device

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        devices=cfg.devices,
    )

    datamodule = DataModule(cfg=cfg.datamodule)
    datamodule.setup(stage=TrainerFn.TESTING)
    if cfg.eval_set == "test":
        dataloader = datamodule.test_dataloader()
    elif cfg.eval_set == "valid":
        dataloader = datamodule.val_dataloader()
    else:
        raise ValueError(f"invalid eval_set: {cfg.eval_set}")
    trainer.test(model=model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
