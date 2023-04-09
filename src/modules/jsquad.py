import torch
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForQuestionAnswering, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from datamodule.datasets.jsquad import JSQuADDataset
from metrics import JSQuADMetric
from modules.base import BaseModule


class JSQuADModule(BaseModule):
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
        return self.model(**{k: v for k, v in batch.items() if k in self.MODEL_ARGS})

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out: QuestionAnsweringModelOutput = self(batch)
        self.log("train/loss", out.loss)
        return out.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        out: QuestionAnsweringModelOutput = self(batch)
        dataset: JSQuADDataset = self.trainer.val_dataloaders.dataset
        self.metric.update(batch["example_ids"], out.start_logits, out.end_logits, dataset)

    def on_validation_epoch_end(self) -> None:
        self.log_dict({f"valid/{key}": value for key, value in self.metric.compute().items()})
        self.metric.reset()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        out: QuestionAnsweringModelOutput = self(batch)
        dataset: JSQuADDataset = self.trainer.test_dataloaders.dataset
        self.metric.update(batch["example_ids"], out.start_logits, out.end_logits, dataset)

    def on_test_epoch_end(self) -> None:
        self.log_dict({f"test/{key}": value for key, value in self.metric.compute().items()})
        self.metric.reset()
