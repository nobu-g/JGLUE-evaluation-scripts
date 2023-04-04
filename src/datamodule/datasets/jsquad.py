import os
from typing import Any, Optional

import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from datamodule.util import QuestionAnsweringFeatures, batch_segment


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

        self.hf_dataset: HFDataset = dataset.map(
            preprocess,
            batched=True,
            batch_size=100,
            num_proc=os.cpu_count(),
        )

    def __getitem__(self, index: int) -> QuestionAnsweringFeatures:
        example: dict[str, Any] = self.hf_dataset[index]
        inputs = self.tokenizer(
            example["question"],
            example["context"],
            padding=PaddingStrategy.MAX_LENGTH,
            truncation="only_second",
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
        )
        answer = example["answers"][0]
        start_positions, end_positions = self._get_token_span(
            inputs, example["context"], answer["text"], answer["answer_start"]
        )

        return QuestionAnsweringFeatures(
            example_ids=index,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            start_positions=start_positions,
            end_positions=end_positions,
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)

    @staticmethod
    def _get_token_span(inputs: BatchEncoding, context: str, answer_text: str, answer_start: int) -> tuple[int, int]:
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

        answer_end = answer_start + len(answer_text)
        try:
            # `answer_start`に対応するトークンのインデックスを検索
            token_start = (offset_mapping[:, 0] == answer_start).nonzero()[0].item()
            token_end = (offset_mapping[:, 1] == answer_end).nonzero()[0].item()
        except ValueError:
            # 見つからなければ先頭のトークン([CLS])を指すように設定
            token_start, token_end = 0, 0

        return token_start, token_end


def preprocess(examples):
    titles, bodies = zip(*[context.split(" [SEP] ") for context in examples["context"]])
    segmented_titles = batch_segment(titles)
    segmented_bodies = batch_segment(bodies)
    segmented_contexts = [f"{title} [SEP] {body}" for title, body in zip(segmented_titles, segmented_bodies)]
    segmented_questions = batch_segment(examples["question"])
    batch_answers: list[list[dict]] = []
    for answers, segmented_context, segmented_title in zip(examples["answers"], segmented_contexts, segmented_titles):
        processed_answers: list[dict] = []
        for answer_text, answer_start in zip(answers["text"], answers["answer_start"]):
            segmented_answer_text, answer_start = find_segmented_answer(
                segmented_context, answer_text, answer_start, len(segmented_titles)
            )
            # 答えの文字列が単語区切りに沿わない場合は分かち書きを適用しない
            if answer_start is None:
                processed_answers.append(dict(text=answer_text, answer_start=-1))
                continue
            assert segmented_answer_text is not None
            processed_answers.append(dict(text=segmented_answer_text, answer_start=answer_start))
        batch_answers.append(processed_answers)
    return {"context": segmented_contexts, "question": segmented_questions, "answers": batch_answers}


def find_segmented_answer(
    segmented_context: str, answer_text: str, answer_start: int, sep_index: int
) -> tuple[Optional[str], Optional[int]]:
    """単語区切りの文脈から単語区切りの答えのスパンを探す"""
    words = segmented_context.split(" ")
    # 単語のインデックスと文字のインデックスの対応関係を保持
    word_index2char_index = [0]
    for word in words:
        # [SEP]だけ前後の半角スペースを考慮する必要があるため+2する
        char_length = len(word) + 2 if word == "[SEP]" else len(word)
        word_index2char_index.append(word_index2char_index[-1] + char_length)

    # 答えのスパンの開始位置が単語区切りに沿うかチェック
    if answer_start in word_index2char_index:
        word_index = word_index2char_index.index(answer_start)
        buf = []
        for word in words[word_index:]:
            buf.append(word)
            # 分かち書きしても答えのスパンが見つかる場合
            if "".join(buf) == answer_text:
                offset = 2 if answer_start >= sep_index else 0
                return " ".join(buf), answer_start + word_index - offset
    return None, None
