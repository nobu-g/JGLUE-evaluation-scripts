from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset
from rhoknp import Jumanpp
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from datamodule.datasets.base import QuestionAnsweringFeatures

ANSWER_COLUMN_NAME = "answer"


@dataclass
class Answer:
    text: str
    start: int  # character index


@dataclass
class JsquadExample:
    id: str
    title: str
    context: str
    question: str
    answers: list[Answer]
    is_impossible: bool

    @classmethod
    def from_hf_dataset(cls, hf_example: dict[str, Any]) -> "JsquadExample":
        return cls(
            id=hf_example["id"],
            title=hf_example["title"],
            context=hf_example["context"],
            question=hf_example["question"],
            answers=[Answer(*x) for x in zip(hf_example["answers"]["text"], hf_example["answers"]["answer_start"])],
            is_impossible=hf_example["is_impossible"],
        )


class JsquadDataset(Dataset[QuestionAnsweringFeatures]):
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
        # columns: id, title, context, question, answers, is_impossible
        dataset: HFDataset = load_dataset("shunk031/JGLUE", name="JSQuAD", split=self.split)
        if limit_examples > 0:
            dataset = dataset.select(range(limit_examples))

        self.examples: list[JsquadExample] = []
        for hf_example in dataset:
            example = JsquadExample.from_hf_dataset(hf_example)
            preprocess(example)
            self.examples.append(example)

    def __getitem__(self, index: int) -> QuestionAnsweringFeatures:
        example: JsquadExample = self.examples[index]
        inputs = self.tokenizer(
            example.question,
            example.context,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation="only_second",
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
            # return_tensors="pt",
        )
        start_positions, end_positions = self._get_token_span(inputs, example.context, example.answers[0])
        if self.split == "train":
            start_positions_all = None
            end_positions_all = None
        else:
            # 評価時は答えの候補が 2つ または 3つ ある
            start_positions_all, end_positions_all = [], []
            for answer in example.answers:
                start_position, end_position = self._get_token_span(inputs, example.context, answer)
                start_positions_all.append(start_position)
                end_positions_all.append(end_position)
            if len(example.answers) == 2:  # 候補が2つの時はpadding
                start_positions_all.append(-100)
                end_positions_all.append(-100)

        return QuestionAnsweringFeatures(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            start_positions=start_positions,
            end_positions=end_positions,
            start_positions_all=start_positions_all,
            end_positions_all=end_positions_all,
        )

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _get_token_span(inputs: BatchEncoding, context: str, answer: Answer) -> tuple[int, int]:
        """スパンの位置について、文字単位からトークン単位に変換"""
        context_length = 0
        # sequence_ids の内容は
        #   None ... 特殊トークン
        #   0 ... 1番目の入力(=`question`)のトークン
        #   1 ... 2番目の入力(=`context`)のトークン
        # であることを表す
        sequence_ids = inputs.sequence_ids()
        # トークンのインデックスと文字のスパンのマッピングを保持した変数
        # "京都 大学" -> ["[CLS]", "▁京都", "▁大学", "[SEP]"] のように分割された場合、
        #   [(0, 0), (0, 2), (2, 5), (0, 0)]
        offset_mapping: np.ndarray = np.array(inputs["offset_mapping"])  # (N, 2)
        assert offset_mapping.shape[0] == len(sequence_ids)
        for i, (sequence_id, offset) in enumerate(zip(sequence_ids, offset_mapping)):
            char_start, _ = offset
            if sequence_id == 1:
                context_length += 1
                # 時折半角スペースが無視されていない時があるため、その場合はマッピングを1つずらす
                if context[char_start] == " ":
                    offset_mapping[i, 0] += 1
            else:
                # 対象外のトークンは`answer_start`の検索時に引っかからないように -100 をセット
                offset_mapping[i, :] = -100

        answer_end = answer.start + len(answer.text)
        try:
            # `answer_start`に対応するトークンのインデックスを検索
            token_start = (offset_mapping[:, 0] == answer.start).nonzero()[0].item()
            token_end = (offset_mapping[:, 1] == answer_end).nonzero()[0].item()
        except ValueError:
            # 見つからなければ先頭のトークン([CLS])を指すように設定
            token_start, token_end = 0, 0

        return token_start, token_end


def batch_segment(texts: list[str]) -> list[str]:
    jumanpp = Jumanpp()
    return [" ".join(m.text for m in jumanpp.apply_to_sentence(text).morphemes) for text in texts]


def preprocess(example: JsquadExample):
    """前処理を適用"""
    title, body = example.context.split(" [SEP] ")
    segmented_title, segmented_body, segmented_question = batch_segment([title, body, example.question])
    example.context = f"{segmented_title} [SEP] {segmented_body}"
    example.question = segmented_question
    for answer in example.answers:
        segmented_answer_text, answer_start = find_segmented_answer(example.context, answer, len(title))
        # 答えの文字列が単語区切りに沿わない場合は分かち書きを適用しない
        if answer_start is None:
            answer.start = -1
            continue
        assert segmented_answer_text is not None
        answer.text = segmented_answer_text
        answer.start = answer_start


def find_segmented_answer(segmented_context: str, answer: Answer, sep_index) -> tuple[Optional[str], Optional[int]]:
    """単語区切りの文脈から単語区切りの答えのスパンを探す"""
    words = segmented_context.split(" ")
    # 単語のインデックスと文字のインデックスの対応関係を保持
    word_index2char_index = [0]
    for word in words:
        # [SEP]だけ前後の半角スペースを考慮する必要があるため+2する
        char_length = len(word) + 2 if word == "[SEP]" else len(word)
        word_index2char_index.append(word_index2char_index[-1] + char_length)

    # 答えのスパンの開始位置が単語区切りに沿うかチェック
    if answer.start in word_index2char_index:
        word_index = word_index2char_index.index(answer.start)
        buf = []
        for word in words[word_index:]:
            buf.append(word)
            # 分かち書きしても答えのスパンが見つかる場合
            if "".join(buf) == answer.text:
                offset = 2 if answer.start >= sep_index else 0
                return " ".join(buf), answer.start + word_index - offset
    return None, None
