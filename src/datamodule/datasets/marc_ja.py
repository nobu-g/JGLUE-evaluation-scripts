import os
from dataclasses import dataclass
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset
from rhoknp import Jumanpp
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass(frozen=True)
class MarkJaFeatures:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    labels: int


class MarkJaDataset(Dataset[MarkJaFeatures]):
    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        limit_examples: int = -1,
    ) -> None:
        super().__init__()
        self.split: str = split
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_seq_length: int = max_seq_length

        # NOTE: JGLUE does not provide test set.
        if self.split == "test":
            self.split = "validation"
        dataset = load_dataset("shunk031/JGLUE", name="MARC-ja", split=self.split)
        if limit_examples > 0:
            dataset = dataset.select(range(limit_examples))
        self.hf_dataset: HFDataset = dataset.map(
            lambda x: {"segmented": batch_segment(x["sentence"])},
            batched=True,
            batch_size=100,
            num_proc=os.cpu_count(),
        ).map(
            lambda x: self.tokenizer(
                x["segmented"],
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=True,
                max_length=self.max_seq_length - 2,  # +2 for [CLS] and [SEP]
            ),
            batched=True,
        )

    def __getitem__(self, index: int) -> MarkJaFeatures:
        example: dict[str, Any] = self.hf_dataset[index]
        return MarkJaFeatures(
            input_ids=example["input_ids"],
            attention_mask=example["attention_mask"],
            token_type_ids=example["token_type_ids"],
            labels=example["label"],
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)


def batch_segment(texts: list[str]) -> list[str]:
    jumanpp = Jumanpp()
    return [" ".join(m.text for m in jumanpp.apply_to_sentence(text).morphemes) for text in texts]
