from typing import TYPE_CHECKING, ClassVar

import torch
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForQuestionAnswering, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from metrics import JSQuADMetric
from modules.base import BaseModule

if TYPE_CHECKING:
    from datamodule.datasets.jsquad import JSQuADDataset


class JSQuADModule(BaseModule):
    MODEL_ARGS: ClassVar[list[str]] = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "start_positions",
        "end_positions",
    ]

    def __init__(self, hparams: DictConfig) -> None:
        super().__init__(hparams)
        config: PretrainedConfig = AutoConfig.from_pretrained(
            hparams.model.model_name_or_path,
            finetuning_task="JSQuAD",
        )
        self.model: PreTrainedModel = AutoModelForQuestionAnswering.from_pretrained(
            hparams.model.model_name_or_path,
            config=config,
        )
        self.metric = JSQuADMetric()

    def forward(self, batch: dict[str, torch.Tensor]) -> QuestionAnsweringModelOutput:
        return self.model(**{k: v for k, v in batch.items() if k in self.MODEL_ARGS})

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out: QuestionAnsweringModelOutput = self(batch)
        assert out.loss is not None
        self.log("train/loss", out.loss)
        return out.loss

    def validation_step(self, batch: dict[str, torch.Tensor]) -> None:
        out: QuestionAnsweringModelOutput = self(batch)
        assert out.start_logits is not None
        assert out.end_logits is not None
        dataloader = self.trainer.val_dataloaders
        assert dataloader is not None
        dataset: JSQuADDataset = dataloader.dataset
        self.metric.update(
            batch["example_ids"],  # ty: ignore[invalid-argument-type]
            out.start_logits,
            out.end_logits,
            dataset,
        )

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            {
                f"valid/{key}": value
                for key, value in self.metric.compute().items()  # ty: ignore[missing-argument]
            }
        )
        self.metric.reset()

    def test_step(self, batch: dict[str, torch.Tensor]) -> None:
        out: QuestionAnsweringModelOutput = self(batch)
        assert out.start_logits is not None
        assert out.end_logits is not None
        dataloader = self.trainer.test_dataloaders
        assert dataloader is not None
        dataset: JSQuADDataset = dataloader.dataset
        self.metric.update(
            batch["example_ids"],  # ty: ignore[invalid-argument-type]
            out.start_logits,
            out.end_logits,
            dataset,
        )

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            {
                f"test/{key}": value
                for key, value in self.metric.compute().items()  # ty: ignore[missing-argument]
            }
        )
        self.metric.reset()
