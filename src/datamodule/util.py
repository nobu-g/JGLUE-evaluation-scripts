from dataclasses import dataclass

from rhoknp import Jumanpp


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
    example_ids: int
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    start_positions: int
    end_positions: int


def batch_segment(texts: list[str]) -> list[str]:
    jumanpp = Jumanpp()
    return [" ".join(m.text for m in jumanpp.apply_to_sentence(text).morphemes) for text in texts]
