from transformers import PreTrainedTokenizerBase

from datamodule.datasets.jsquad import Answer, JsquadDataset, JsquadExample


def test_init(tokenizer: PreTrainedTokenizerBase):
    _ = JsquadDataset("train", tokenizer, max_seq_length=128, limit_examples=3)


def test_examples_0(tokenizer: PreTrainedTokenizerBase):
    dataset = JsquadDataset("validation", tokenizer, max_seq_length=128, limit_examples=1)
    example = dataset.examples[0]
    for answer in example.answers:
        assert example.context[answer.start :].startswith(answer.text)


def test_getitem(tokenizer: PreTrainedTokenizerBase):
    max_seq_length = 128
    dataset = JsquadDataset("train", tokenizer, max_seq_length, limit_examples=3)
    for i in range(len(dataset)):
        feature = dataset[i]
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
        assert len(feature.token_type_ids) == max_seq_length
        assert isinstance(feature.start_positions, int)
        assert isinstance(feature.end_positions, int)


def test_features_0(tokenizer: PreTrainedTokenizerBase):
    max_seq_length = 128
    dataset = JsquadDataset("validation", tokenizer, max_seq_length, limit_examples=1)
    example = JsquadExample(
        id="a10336p0q0",
        title="梅雨",
        context="梅雨 [SEP] 梅雨 （ つゆ 、 ばいう ） は 、 北海道 と 小笠原 諸島 を 除く 日本 、 朝鮮 半島 南部 、 中国 の 南部 から 長江 流域 に かけて の 沿海 部 、 および 台湾 など 、 東 アジア の 広範囲に おいて み られる 特有の 気象 現象 で 、 5 月 から 7 月 に かけて 来る 曇り や 雨 の 多い 期間 の こと 。 雨季 の 一種 である 。",
        question="日本 で 梅雨 が ない の は 北海道 と どこ か 。",
        answers=[
            Answer(text="小笠原 諸島", start=35),
            Answer(text="小笠原 諸島 を 除く 日本", start=35),
            Answer(text="小笠原 諸島", start=35),
        ],
        is_impossible=False,
    )
    features = dataset[0]
    question_tokens: list[str] = tokenizer.tokenize(example.question)
    context_tokens: list[str] = tokenizer.tokenize(example.context)
    input_tokens = (
        [tokenizer.cls_token] + question_tokens + [tokenizer.sep_token] + context_tokens + [tokenizer.sep_token]
    )
    padded_input_tokens = input_tokens + [tokenizer.pad_token] * (max_seq_length - len(input_tokens))
    assert features.input_ids == tokenizer.convert_tokens_to_ids(padded_input_tokens)
    assert features.attention_mask == [1] * len(input_tokens) + [0] * (max_seq_length - len(input_tokens))
    assert features.token_type_ids == [0] * (len(question_tokens) + 2) + [1] * (len(context_tokens) + 1) + [0] * (
        max_seq_length - len(input_tokens)
    )

    assert 0 <= features.start_positions <= features.end_positions < max_seq_length
    answer_span = slice(features.start_positions, features.end_positions + 1)
    tokenized_answer_text: str = tokenizer.decode(features.input_ids[answer_span])
    assert tokenized_answer_text == example.answers[0].text
