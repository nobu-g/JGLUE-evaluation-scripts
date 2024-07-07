import os
from itertools import chain
from typing import Any

from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from datamodule.datasets.base import BaseDataset
from datamodule.datasets.util import MultipleChoiceFeatures, batch_segment

CHOICE_NAMES = ["choice0", "choice1", "choice2", "choice3", "choice4"]
NUM_CHOICES = len(CHOICE_NAMES)


class JCommonsenseQADataset(BaseDataset[MultipleChoiceFeatures]):
    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        segmenter_kwargs: DictConfig,
        limit_examples: int = -1,
    ) -> None:
        super().__init__("JCommonsenseQA", split, tokenizer, max_seq_length, limit_examples)

        def preprocess_function(examples: dict[str, list]) -> dict[str, list[list[Any]]]:
            # (example, 5)
            first_sentences: list[list[str]] = [[question] * NUM_CHOICES for question in examples["question"]]
            second_sentences: list[list[str]] = [
                [examples[name][i] for name in CHOICE_NAMES] for i in range(len(examples["question"]))
            ]
            # Tokenize
            tokenized_examples = self.tokenizer(
                list(chain(*first_sentences)),
                list(chain(*second_sentences)),
                truncation=True,
                max_length=self.max_seq_length,
                padding=PaddingStrategy.MAX_LENGTH,
            )
            # Un-flatten
            return {
                k: [v[i : i + NUM_CHOICES] for i in range(0, len(v), NUM_CHOICES)]
                for k, v in tokenized_examples.items()
            }

        self.hf_dataset = self.hf_dataset.map(
            lambda x: {
                key: batch_segment(x[key], **segmenter_kwargs) for key in ["question", *CHOICE_NAMES]  # type: ignore[misc]
            },
            batched=True,
            batch_size=100,
            num_proc=os.cpu_count(),
        ).map(
            preprocess_function,
            batched=True,
        )

    def __getitem__(self, index: int) -> MultipleChoiceFeatures:
        example: dict[str, Any] = self.hf_dataset[index]
        return MultipleChoiceFeatures(
            input_ids=example["input_ids"],
            attention_mask=example["attention_mask"],
            token_type_ids=example["token_type_ids"],
            labels=example["label"],
        )
