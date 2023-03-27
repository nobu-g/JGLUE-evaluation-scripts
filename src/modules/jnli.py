from typing import Any

import torch
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy
from transformers import AutoConfig, AutoModelForSequenceClassification, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from modules.base import BaseModule


class JnliModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)
        config: PretrainedConfig = AutoConfig.from_pretrained(
            hparams.model_name_or_path,
            num_labels=3,
            finetuning_task="JNLI",
        )
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            hparams.model_name_or_path,
            config=config,
        )
        self.metric = MulticlassAccuracy(num_classes=3)

    def forward(self, batch: dict[str, Any]) -> SequenceClassifierOutput:
        return self.model(**batch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out: SequenceClassifierOutput = self(batch)
        self.log("train/loss", out.loss)
        return out.loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        out: SequenceClassifierOutput = self(batch)
        preds = torch.argmax(out.logits, dim=1)  # (b)
        self.metric.update(preds, batch["labels"])

    def on_validation_epoch_end(self) -> None:
        metrics = self.metric.compute()
        self.metric.reset()
        self.log("valid/accuracy", metrics)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        out: SequenceClassifierOutput = self(batch)
        preds = torch.argmax(out.logits, dim=1)  # (b)
        self.metric.update(preds, batch["labels"])

    def on_test_epoch_end(self) -> None:
        metrics = self.metric.compute()
        self.metric.reset()
        self.log("test/accuracy", metrics)
