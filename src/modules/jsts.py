from typing import Any

import torch
from omegaconf import DictConfig
from torchmetrics import Metric, PearsonCorrCoef, SpearmanCorrCoef
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from modules.base import BaseModule


class JstsModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)
        config = AutoConfig.from_pretrained(
            hparams.model_name_or_path,
            num_labels=1,
            finetuning_task="JSTS",
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hparams.model_name_or_path,
            config=config,
        )
        self.metrics: dict[str, Metric] = {"spearman": SpearmanCorrCoef(), "pearson": PearsonCorrCoef()}

    def forward(self, batch: dict[str, Any]) -> SequenceClassifierOutput:
        return self.model(**batch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out: SequenceClassifierOutput = self(batch)
        self.log("train/loss", out.loss)
        return out.loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        out: SequenceClassifierOutput = self(batch)
        preds = torch.squeeze(out.logits, dim=-1)  # (b)
        for metric in self.metrics.values():
            metric.update(preds, batch["labels"])

    def on_validation_epoch_end(self) -> None:
        for name, metric in self.metrics.items():
            self.log(f"valid/{name}", metric.compute())
            metric.reset()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        out: SequenceClassifierOutput = self(batch)
        preds = torch.squeeze(out.logits, dim=-1)  # (b)
        for metric in self.metrics.values():
            metric.update(preds, batch["labels"])

    def on_test_epoch_end(self) -> None:
        for name, metric in self.metrics.items():
            self.log(f"test/{name}", metric.compute())
            metric.reset()
