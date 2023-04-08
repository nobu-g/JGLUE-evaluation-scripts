import os
from typing import Any, Optional

from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from datamodule.util import QuestionAnsweringFeatures, batch_segment


class JSQuADDataset(Dataset[QuestionAnsweringFeatures]):
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

        self.hf_dataset: HFDataset = dataset.map(
            preprocess,
            batched=True,
            batch_size=100,
            num_proc=os.cpu_count(),
        ).map(
            lambda x: self.tokenizer(
                x["question"],
                x["context"],
                padding=PaddingStrategy.MAX_LENGTH,
                truncation="only_second",
                max_length=self.max_seq_length,
                return_offsets_mapping=True,
            ),
            batched=True,
        )

        # skip invalid examples for training
        if self.split == "train":
            self.hf_dataset = self.hf_dataset.filter(
                lambda example: any(answer["answer_start"] >= 0 for answer in example["answers"])
            )

    def __getitem__(self, index: int) -> QuestionAnsweringFeatures:
        example: dict[str, Any] = self.hf_dataset[index]
        start_positions = end_positions = 0
        for answer in example["answers"]:
            start_positions, end_positions = self._get_token_span(example, answer["text"], answer["answer_start"])
            if start_positions > 0 or end_positions > 0:
                break

        return QuestionAnsweringFeatures(
            example_ids=index,
            input_ids=example["input_ids"],
            attention_mask=example["attention_mask"],
            token_type_ids=example["token_type_ids"],
            start_positions=start_positions,
            end_positions=end_positions,
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)

    @staticmethod
    def _get_token_span(example: dict[str, Any], answer_text: str, answer_start: int) -> tuple[int, int]:
        """スパンの位置について、文字単位からトークン単位に変換"""
        # token_type_ids:
        #   0: 1番目の入力(=`question`)のトークン or パディング
        #   1: 2番目の入力(=`context`)のトークン
        token_type_ids: list[int] = example["token_type_ids"]
        # トークンのインデックスと文字のスパンのマッピングを保持した変数
        # "京都 大学" -> ["[CLS]", "▁京都", "▁大学", "[SEP]"] のように分割された場合、
        #   [(0, 0), (0, 2), (2, 5), (0, 0)]
        offset_mapping: list[tuple[int, int]] = example["offset_mapping"]
        context: str = example["context"]
        assert len(offset_mapping) == len(token_type_ids)
        token_to_char_start_index = [x[0] for x in offset_mapping]
        token_to_char_end_index = [x[1] for x in offset_mapping]
        answer_end = answer_start + len(answer_text)
        token_start_index = token_end_index = 0
        for token_index, (token_type_id, char_start_index, char_end_index) in enumerate(
            zip(token_type_ids, token_to_char_start_index, token_to_char_end_index)
        ):
            if token_type_id != 1 or char_start_index == char_end_index == 0:
                continue
            # 半角スペースが無視されていない時があるため、その場合はマッピングを1つずらす
            if context[char_start_index] == " ":
                char_start_index += 1
            if answer_start == char_start_index:
                token_start_index = token_index
            if answer_end == char_end_index:
                token_end_index = token_index
        return token_start_index, token_end_index


def preprocess(examples):
    titles: list[str]
    bodies: list[str]
    titles, bodies = zip(*[context.split(" [SEP] ") for context in examples["context"]])
    segmented_titles = batch_segment(titles)
    segmented_bodies = batch_segment(bodies)
    segmented_contexts = [f"{title} [SEP] {body}" for title, body in zip(segmented_titles, segmented_bodies)]
    segmented_questions = batch_segment(examples["question"])
    batch_answers: list[list[dict]] = []
    for answers, segmented_context, title in zip(examples["answers"], segmented_contexts, titles):
        processed_answers: list[dict] = []
        for answer_text, answer_start in zip(answers["text"], answers["answer_start"]):
            segmented_answer_text, answer_start = find_segmented_answer(
                segmented_context, answer_text, answer_start, len(title)
            )
            if answer_start is None:
                processed_answers.append(dict(text=batch_segment([answer_text])[0], answer_start=-1))
                continue
            assert segmented_answer_text is not None
            processed_answers.append(dict(text=segmented_answer_text, answer_start=answer_start))
        batch_answers.append(processed_answers)
    return {"context": segmented_contexts, "question": segmented_questions, "answers": batch_answers}


def find_segmented_answer(
    segmented_context: str, answer_text: str, answer_start: int, sep_index: int
) -> tuple[Optional[str], Optional[int]]:
    """単語区切りされた context から単語区切りされた answer のスパンを探索

    Args:
        segmented_context: 単語区切りされた context
        answer_text: answer の文字列
        answer_start: answer の文字単位開始インデックス
        sep_index: [SEP] の文字単位開始インデックス

    Returns:
        Optional[str]: 単語区切りされた answer（見つからなければ None）
        Optional[int]: 単語区切りされた context における answer の文字単位開始インデックス（見つからなければ None）
    """
    words = segmented_context.split(" ")
    char_to_word_index = {}
    char_index = 0
    for word_index, word in enumerate(words):
        char_to_word_index[char_index] = word_index
        # [SEP]だけ前後の半角スペースを考慮する必要があるため+2する
        char_length = len(word) + 2 if word == "[SEP]" else len(word)
        char_index += char_length

    # 答えのスパンの開始位置が単語区切りに沿うかチェック
    if answer_start in char_to_word_index:
        word_index = char_to_word_index[answer_start]
        buf = []
        for word in words[word_index:]:
            buf.append(word)
            # 分かち書きしても答えのスパンが見つかる場合
            if "".join(buf) == answer_text:
                offset = 2 if answer_start >= sep_index else 0
                return " ".join(buf), answer_start + word_index - offset
    return None, None
