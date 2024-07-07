import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture()
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-tiny-japanese")


@pytest.fixture()
def deberta_v3_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("ku-nlp/deberta-v3-base-japanese")
