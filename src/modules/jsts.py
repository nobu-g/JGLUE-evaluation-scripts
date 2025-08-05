from typing import Any

import torch
from omegaconf import DictConfig
from torchmetrics import MetricCollection, PearsonCorrCoef, SpearmanCorrCoef
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from modules.base import BaseModule


class JSTSModule(BaseModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)
        config = AutoConfig.from_pretrained(
            hparams.model.model_name_or_path,
            num_labels=1,
            finetuning_task="JSTS",
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hparams.model.model_name_or_path,
            config=config,
        )
        self.metric = MetricCollection({"spearman": SpearmanCorrCoef(), "pearson": PearsonCorrCoef()})

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
        predictions = torch.squeeze(out.logits, dim=-1)  # (b)
        self.metric.update(predictions, batch["labels"])

    def on_validation_epoch_end(self) -> None:
        self.log_dict({f"valid/{key}": value for key, value in self.metric.compute().items()})
        self.metric.reset()

    def test_step(self, batch: Any) -> None:
        out: SequenceClassifierOutput = self(batch)
        assert out.logits is not None
        predictions = torch.squeeze(out.logits, dim=-1)  # (b)
        self.metric.update(predictions, batch["labels"])

    def on_test_epoch_end(self) -> None:
        self.log_dict({f"test/{key}": value for key, value in self.metric.compute().items()})
        self.metric.reset()
