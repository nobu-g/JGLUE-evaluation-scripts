from typing import Any

import torch
from omegaconf import DictConfig
from torchmetrics import SpearmanCorrCoef
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from modules.base import BaseModule


class JstsModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)
        config = AutoConfig.from_pretrained(
            hparams.encoder.pretrained_model_name_or_path,
            num_labels=1,
            finetuning_task="JSTS",
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hparams.encoder.pretrained_model_name_or_path,
            config=config,
        )
        self.metric = SpearmanCorrCoef()

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
        self.metric.update(out.logits, batch["labels"])

    def on_validation_epoch_end(self) -> None:
        metrics = self.metric.compute()
        self.metric.reset()
        self.log("valid/spearman", metrics)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        out: SequenceClassifierOutput = self(batch)
        self.metric.update(out.logits, batch["labels"])

    def on_test_epoch_end(self) -> None:
        metrics = self.metric.compute()
        self.metric.reset()
        self.log("test/spearman", metrics)
