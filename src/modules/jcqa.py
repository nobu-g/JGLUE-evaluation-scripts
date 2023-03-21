from typing import Any

import torch
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy
from transformers import AutoConfig, AutoModelForMultipleChoice
from transformers.modeling_outputs import MultipleChoiceModelOutput

from modules.base import BaseModule


class JcqaModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)
        config = AutoConfig.from_pretrained(
            hparams.model_name_or_path,
            num_labels=5,
            finetuning_task="JCommonsenseQA",
        )
        self.model = AutoModelForMultipleChoice.from_pretrained(
            hparams.model_name_or_path,
            config=config,
        )
        self.metric = MulticlassAccuracy(num_classes=5)

    def forward(self, batch: dict[str, Any]) -> MultipleChoiceModelOutput:
        return self.model(**batch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out: MultipleChoiceModelOutput = self(batch)
        self.log("train/loss", out.loss)
        return out.loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        out: MultipleChoiceModelOutput = self(batch)
        preds = torch.argmax(out.logits, dim=1)  # (b)
        self.metric.update(preds, batch["labels"])

    def on_validation_epoch_end(self) -> None:
        metrics = self.metric.compute()
        self.metric.reset()
        self.log("valid/accuracy", metrics)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        out: MultipleChoiceModelOutput = self(batch)
        preds = torch.argmax(out.logits, dim=1)  # (b)
        self.metric.update(preds, batch["labels"])

    def on_test_epoch_end(self) -> None:
        metrics = self.metric.compute()
        self.metric.reset()
        self.log("test/accuracy", metrics)
