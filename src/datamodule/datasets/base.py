from abc import ABC
from typing import Generic, TypeVar

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from datamodule.datasets.loader import load_jglue

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
        hf_dataset = load_jglue(dataset_name, self.split)
        assert isinstance(hf_dataset, HFDataset)
        self.hf_dataset = hf_dataset
        if limit_examples > 0:
            self.hf_dataset = self.hf_dataset.select(range(limit_examples))

    def __getitem__(self, index: int) -> FeatureType:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.hf_dataset)
