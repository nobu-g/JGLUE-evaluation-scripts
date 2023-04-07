from typing import Any

import torch
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForQuestionAnswering, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from datamodule.datasets.jsquad import JsquadDataset
from metrics import JSQuADMetric
from modules.base import BaseModule


class JsquadModule(BaseModule):
    MODEL_ARGS = ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]

    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)
        config: PretrainedConfig = AutoConfig.from_pretrained(
            hparams.model_name_or_path,
            finetuning_task="JSQuAD",
        )
        self.model: PreTrainedModel = AutoModelForQuestionAnswering.from_pretrained(
            hparams.model_name_or_path,
            config=config,
        )
        self.metric = JSQuADMetric()

    def forward(self, batch: dict[str, torch.Tensor]) -> QuestionAnsweringModelOutput:
        return self.model(**batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out: QuestionAnsweringModelOutput = self({k: v for k, v in batch.items() if k in self.MODEL_ARGS})
        self.log("train/loss", out.loss)
        return out.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        out: QuestionAnsweringModelOutput = self({k: v for k, v in batch.items() if k in self.MODEL_ARGS})
        pred_starts = torch.argmax(out.start_logits, dim=1)  # (b, seq) -> (b)
        pred_ends = torch.argmax(out.end_logits, dim=1)  # (b, seq) -> (b)
        dataset: JsquadDataset = self.trainer.val_dataloaders.dataset
        metric_kwargs: dict[str, Any] = {k: v for k, v in batch.items() if k in ("example_ids", "input_ids")}
        metric_kwargs.update(pred_starts=pred_starts, pred_ends=pred_ends, dataset=dataset)
        self.metric.update(**metric_kwargs)

    def on_validation_epoch_end(self) -> None:
        metrics = self.metric.compute()
        self.metric.reset()
        self.log_dict({f"valid/{key}": value / 100.0 for key, value in metrics.items()})

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        out: QuestionAnsweringModelOutput = self({k: v for k, v in batch.items() if k in self.MODEL_ARGS})
        pred_starts = torch.argmax(out.start_logits, dim=1)  # (b, seq) -> (b)
        pred_ends = torch.argmax(out.end_logits, dim=1)  # (b, seq) -> (b)
        dataset: JsquadDataset = self.trainer.test_dataloaders.dataset
        metric_kwargs: dict[str, Any] = {k: v for k, v in batch.items() if k in ("example_ids", "input_ids")}
        metric_kwargs.update(pred_starts=pred_starts, pred_ends=pred_ends, dataset=dataset)
        self.metric.update(**metric_kwargs)

    def on_test_epoch_end(self) -> None:
        metrics = self.metric.compute()
        self.metric.reset()
        self.log_dict({f"test/{key}": value / 100.0 for key, value in metrics.items()})
