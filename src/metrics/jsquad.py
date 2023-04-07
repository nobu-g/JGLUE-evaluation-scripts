import torch
from torchmetrics import Metric, SQuAD
from transformers import PreTrainedTokenizerBase

from datamodule.datasets import JsquadDataset


class JSQuADMetric(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.squad = SQuAD()

    def update(
        self,
        example_ids: torch.Tensor,  # (b)
        input_ids: torch.Tensor,  # (b, seq)
        pred_starts: torch.Tensor,  # (b)
        pred_ends: torch.Tensor,  # (b)
        dataset: JsquadDataset,
    ) -> None:
        preds = []
        target = []
        for example_id, input_id, pred_start, pred_end in zip(
            example_ids.tolist(), input_ids.tolist(), pred_starts.tolist(), pred_ends.tolist()
        ):
            example = dataset.hf_dataset[example_id]
            preds.append(
                {
                    "prediction_text": self._postprocess_text(
                        self._get_text_span(input_id, pred_start, pred_end, dataset.tokenizer)
                    ),
                    "id": example_id,
                }
            )
            target.append(
                {
                    "answers": {
                        "text": [self._postprocess_text(answer["text"]) for answer in example["answers"]],
                        "answer_start": [answer["answer_start"] for answer in example["answers"]],
                    },
                    "id": example_id,
                }
            )
        self.squad.update(preds, target)

    def compute(self) -> dict[str, torch.Tensor]:
        return self.squad.compute()

    @staticmethod
    def _get_text_span(
        input_ids: list[int], start_position: int, end_position: int, tokenizer: PreTrainedTokenizerBase
    ) -> str:
        """トークンの開始位置と終了位置から対応する文字列を取得"""
        token_span = slice(start_position, end_position + 1)
        token_ids = input_ids[token_span]
        return tokenizer.decode(token_ids)

    @staticmethod
    def _postprocess_text(text: str) -> str:
        """句点を除去し，文字単位に分割"""
        return " ".join(text.replace(" ", "").rstrip("。"))
