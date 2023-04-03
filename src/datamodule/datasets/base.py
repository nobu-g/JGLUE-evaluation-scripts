import logging
from abc import ABC
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

ExampleType = TypeVar("ExampleType")
FeatureType = TypeVar("FeatureType")


@dataclass(frozen=True)
class SequenceClassificationFeatures:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    labels: int


@dataclass(frozen=True)
class MultipleChoiceFeatures:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    token_type_ids: list[list[int]]
    labels: int


@dataclass(frozen=True)
class QuestionAnsweringFeatures:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    start_positions: int
    end_positions: int
    start_positions_all: Optional[list[int]]
    end_positions_all: Optional[list[int]]


class BaseDataset(Dataset[FeatureType], Generic[ExampleType, FeatureType], ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_seq_length: int = max_seq_length
        self.examples: List[ExampleType] = []

    def __getitem__(self, index) -> FeatureType:
        return self.encode(self.examples[index])

    def __len__(self) -> int:
        return len(self.examples)

    def encode(self, example: ExampleType) -> FeatureType:
        raise NotImplementedError
