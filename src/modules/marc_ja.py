from typing import Any

import torch
import torchmetrics
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from modules.base import BaseModule


class MarcJaModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)
        self.model = AutoModelForSequenceClassification.from_pretrained(hparams.encoder.pretrained_model_name_or_path)
        self.metric = torchmetrics.classification.MulticlassAccuracy(num_classes=2)

    def forward(self, batch: Any) -> SequenceClassifierOutput:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
        )

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
