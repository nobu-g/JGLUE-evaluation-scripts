from transformers import PreTrainedTokenizerBase

from datamodule.datasets import JsquadDataset


def test_init(tokenizer: PreTrainedTokenizerBase):
    _ = JsquadDataset("train", tokenizer, max_seq_length=128, limit_examples=3)


def test_getitem(tokenizer: PreTrainedTokenizerBase):
    max_seq_length: int = 128
    dataset = JsquadDataset("train", tokenizer, max_seq_length, limit_examples=3)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
        assert len(feature.token_type_ids) == max_seq_length
        assert isinstance(feature.start_positions, int)
        assert isinstance(feature.end_positions, int)
