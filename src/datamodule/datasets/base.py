from abc import ABC
from typing import Generic, TypeVar

from datasets import Dataset as HFDataset  # type: ignore[attr-defined]
from datasets import load_dataset  # type: ignore[attr-defined]
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

FeatureType = TypeVar("FeatureType")


class BaseDataset(Dataset[FeatureType], ABC, Generic[FeatureType]):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        limit_examples: int = -1,
    ) -> None:
        self.split: str = split
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_seq_length: int = max_seq_length

        # NOTE: JGLUE does not provide test set.
        if self.split == "test":
            self.split = "validation"
        # columns: id, title, context, question, answers, is_impossible
        self.hf_dataset: HFDataset = load_dataset(
            "shunk031/JGLUE", name=dataset_name, split=self.split, trust_remote_code=True
        )
        if limit_examples > 0:
            self.hf_dataset = self.hf_dataset.select(range(limit_examples))

    def __getitem__(self, index: int) -> FeatureType:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.hf_dataset)
