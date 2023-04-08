import os
from typing import Any

from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from datamodule.datasets.base import BaseDataset
from datamodule.util import SequenceClassificationFeatures, batch_segment


class JSTSDataset(BaseDataset[SequenceClassificationFeatures]):
    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        limit_examples: int = -1,
    ) -> None:
        super().__init__("JSTS", split, tokenizer, max_seq_length, limit_examples)

        self.hf_dataset = self.hf_dataset.map(
            lambda x: {"segmented1": batch_segment(x["sentence1"]), "segmented2": batch_segment(x["sentence2"])},
            batched=True,
            batch_size=100,
            num_proc=os.cpu_count(),
        ).map(
            lambda x: self.tokenizer(
                x["segmented1"],
                x["segmented2"],
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=True,
                max_length=self.max_seq_length,
            ),
            batched=True,
        )

    def __getitem__(self, index: int) -> SequenceClassificationFeatures:
        example: dict[str, Any] = self.hf_dataset[index]
        return SequenceClassificationFeatures(
            input_ids=example["input_ids"],
            attention_mask=example["attention_mask"],
            token_type_ids=example["token_type_ids"],
            labels=example["label"],
        )
