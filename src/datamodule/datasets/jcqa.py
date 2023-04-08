import os
from itertools import chain
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from datamodule.util import MultipleChoiceFeatures, batch_segment

CHOICE_NAMES = ["choice0", "choice1", "choice2", "choice3", "choice4"]
NUM_CHOICES = len(CHOICE_NAMES)


class JCommonsenseQADataset(Dataset[MultipleChoiceFeatures]):
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
        dataset = load_dataset("shunk031/JGLUE", name="JCommonsenseQA", split=self.split)
        if limit_examples > 0:
            dataset = dataset.select(range(limit_examples))

        def preprocess_function(examples) -> dict[str, list[list[Any]]]:
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

        self.hf_dataset: HFDataset = dataset.map(
            lambda x: {
                "question": batch_segment(x["question"]),
                "choice0": batch_segment(x["choice0"]),
                "choice1": batch_segment(x["choice1"]),
                "choice2": batch_segment(x["choice2"]),
                "choice3": batch_segment(x["choice3"]),
                "choice4": batch_segment(x["choice4"]),
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

    def __len__(self) -> int:
        return len(self.hf_dataset)
