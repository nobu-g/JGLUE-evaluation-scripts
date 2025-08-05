from typing import Any

import torch
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy
from transformers import AutoConfig, AutoModelForSequenceClassification, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from modules.base import BaseModule


class JNLIModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)
        config: PretrainedConfig = AutoConfig.from_pretrained(
            hparams.model.model_name_or_path,
            num_labels=3,
            finetuning_task="JNLI",
        )
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            hparams.model.model_name_or_path,
            config=config,
        )
        self.metric = MulticlassAccuracy(num_classes=3, average="micro")

    def forward(self, batch: dict[str, Any]) -> SequenceClassifierOutput:
        return self.model(**batch)

    def training_step(self, batch: Any) -> torch.Tensor:
        out: SequenceClassifierOutput = self(batch)
        assert out.loss is not None
        self.log("train/loss", out.loss)
        return out.loss

    def validation_step(self, batch: Any) -> None:
        out: SequenceClassifierOutput = self(batch)
        assert out.logits is not None
        predictions = torch.argmax(out.logits, dim=1)  # (b)
        self.metric.update(predictions, batch["labels"])

    def on_validation_epoch_end(self) -> None:
        self.log("valid/accuracy", self.metric.compute())
        self.metric.reset()

    def test_step(self, batch: Any) -> None:
        out: SequenceClassifierOutput = self(batch)
        assert out.logits is not None
        predictions = torch.argmax(out.logits, dim=1)  # (b)
        self.metric.update(predictions, batch["labels"])

    def on_test_epoch_end(self) -> None:
        self.log("test/accuracy", self.metric.compute())
        self.metric.reset()
