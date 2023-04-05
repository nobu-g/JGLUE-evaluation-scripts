import torch
from omegaconf import DictConfig
from torchmetrics import SQuAD
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from datamodule.datasets.jsquad import JsquadDataset
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
        self.metric = SQuAD()

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
        preds = []
        target = []
        for example_id, input_ids, pred_start, pred_end in zip(
            batch["example_ids"].tolist(), batch["input_ids"].tolist(), pred_starts.tolist(), pred_ends.tolist()
        ):
            example = dataset.hf_dataset[example_id]
            preds.append(
                {
                    "prediction_text": self._get_text(input_ids, pred_start, pred_end, dataset.tokenizer),
                    "id": example_id,
                }
            )
            target.append(
                {
                    "answers": {
                        "text": [answer["text"].rstrip("。").rstrip() for answer in example["answers"]],
                        "answer_start": [answer["answer_start"] for answer in example["answers"]],
                    },
                    "id": example_id,
                }
            )
        self.metric.update(preds, target)

    def on_validation_epoch_end(self) -> None:
        metrics = self.metric.compute()
        self.metric.reset()
        self.log_dict({f"valid/{key}": value / 100.0 for key, value in metrics.items()})

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        out: QuestionAnsweringModelOutput = self({k: v for k, v in batch.items() if k in self.MODEL_ARGS})
        pred_starts = torch.argmax(out.start_logits, dim=1)  # (b, seq) -> (b)
        pred_ends = torch.argmax(out.end_logits, dim=1)  # (b, seq) -> (b)
        dataset: JsquadDataset = self.trainer.test_dataloaders.dataset
        preds = []
        target = []
        for example_id, input_ids, pred_start, pred_end in zip(
            batch["example_ids"].tolist(), batch["input_ids"].tolist(), pred_starts.tolist(), pred_ends.tolist()
        ):
            example = dataset.hf_dataset[example_id]
            preds.append(
                {
                    "prediction_text": self._get_text(input_ids, pred_start, pred_end, dataset.tokenizer),
                    "id": example_id,
                }
            )
            target.append(
                {
                    "answers": {
                        "text": [answer["text"].rstrip("。").rstrip() for answer in example["answers"]],
                        "answer_start": [answer["answer_start"] for answer in example["answers"]],
                    },
                    "id": example_id,
                }
            )
        self.metric.update(preds, target)

    def on_test_epoch_end(self) -> None:
        metrics = self.metric.compute()
        self.metric.reset()
        self.log_dict({f"test/{key}": value / 100.0 for key, value in metrics.items()})

    @staticmethod
    def _get_text(
        input_ids: list[int], start_position: int, end_position: int, tokenizer: PreTrainedTokenizerBase
    ) -> str:
        """トークンの開始位置と終了位置から対応する文字列を取得"""
        token_span = slice(start_position, end_position + 1)
        token_ids = input_ids[token_span]
        # 文末に句点がある場合は除く
        text = tokenizer.decode(token_ids).rstrip("。").rstrip()
        return text
