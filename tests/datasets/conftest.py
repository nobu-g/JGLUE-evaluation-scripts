import os
import sys

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))


@pytest.fixture()
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-tiny-japanese")


@pytest.fixture()
def deberta_v3_tokenizer() -> PreTrainedTokenizerBase:
    # TODO: remove `use_auth_token=True` when the model is public
    return AutoTokenizer.from_pretrained("ku-nlp/deberta-v3-base-japanese", use_auth_token=True)
