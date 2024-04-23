from dataclasses import dataclass
from typing import Optional

import jaconv
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


def batch_segment(
    texts: list[str], analyzer: Optional[str], h2z: bool = True, mecab_dic_dir: Optional[str] = None
) -> list[str]:
    if analyzer is None:
        return texts
    segmenter = WordSegmenter(analyzer, h2z, mecab_dic_dir)
    return [segmenter.get_segmented_string(text) for text in texts]


class WordSegmenter:
    def __init__(self, analyzer: str, h2z: bool, mecab_dic_dir: Optional[str] = None):
        self._analyzer: str = analyzer
        self._h2z: bool = h2z

        if self._analyzer == "jumanpp":
            self._jumanpp = Jumanpp()
        elif self._analyzer == "mecab":
            tagger_options = []
            if mecab_dic_dir is not None:
                tagger_options += f"-d {mecab_dic_dir}".split()
            import MeCab

            self._mecab = MeCab.Tagger(" ".join(tagger_options))

    def get_words(self, string: str) -> list[str]:
        words: list[str] = []

        if self._analyzer == "jumanpp":
            sentence = self._jumanpp.apply_to_sentence(string)

            for morpheme in sentence.morphemes:
                words.append(morpheme.text)
        elif self._analyzer == "mecab":
            self._mecab.parse("")
            node = self._mecab.parseToNode(string)
            while node:
                word = node.surface
                if node.feature.split(",")[0] != "BOS/EOS":
                    words.append(word)
                node = node.next
        elif self._analyzer == "char":
            for char in string:
                words.append(char)
        else:
            NotImplementedError(f"unknown analyzer: {self._analyzer}")

        return words

    def get_segmented_string(self, string: str) -> str:
        if self._h2z is True:
            string = jaconv.h2z(string)
        words = self.get_words(string)
        return " ".join(word for word in words)
