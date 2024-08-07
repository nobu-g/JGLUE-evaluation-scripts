import copy
from typing import Any

import hydra
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig, OmegaConf


class BaseModule(LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ("bias", "LayerNorm.weight")
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.optimizer.weight_decay,
                "name": "decay",
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "name": "no_decay",
            },
        ]
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=optimizer_grouped_parameters, _convert_="partial"
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps or total_steps * self.hparams.warmup_ratio
        if hasattr(self.hparams.scheduler, "num_warmup_steps"):
            self.hparams.scheduler.num_warmup_steps = warmup_steps
        if hasattr(self.hparams.scheduler, "num_training_steps"):
            self.hparams.scheduler.num_training_steps = total_steps
        lr_scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        hparams: DictConfig = copy.deepcopy(checkpoint["hyper_parameters"])
        OmegaConf.set_struct(hparams, value=False)
        checkpoint["hyper_parameters"] = hparams
