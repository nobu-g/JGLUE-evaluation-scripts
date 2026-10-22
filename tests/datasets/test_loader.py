import pytest

from datamodule.datasets.loader import flatten_jsquad, load_jglue


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize(
    ("name", "columns", "label_count"),
    [
        ("JNLI", ["sentence1", "sentence2"], 3),
        ("JSTS", ["sentence1", "sentence2"], None),
        ("JCommonsenseQA", ["question", "choice0", "choice4"], 5),
        ("JCoLA", ["sentence"], 2),
        ("MARC-ja", ["sentence"], 3),
    ],
)
def test_load_classification_dataset(name, columns, label_count, split):
    dataset = load_jglue(name, split)
    assert len(dataset) > 0
    for column in columns:
        assert isinstance(dataset[0][column], str)
    if label_count is None:
        assert isinstance(dataset[0]["label"], float)
    else:
        assert set(dataset["label"]) <= set(range(label_count))
        assert dataset.features["label"].num_classes == label_count


def test_flatten_jsquad_preserves_answers():
    batch = {
        "title": ["Title"],
        "paragraphs": [
            [
                {
                    "context": "one two",
                    "qas": [
                        {
                            "id": "a",
                            "question": "Which?",
                            "is_impossible": False,
                            "answers": [{"text": "one", "answer_start": 0}, {"text": "two", "answer_start": 4}],
                        },
                        {"id": "b", "question": "Missing?", "is_impossible": True, "answers": []},
                    ],
                }
            ]
        ],
    }
    rows = flatten_jsquad(batch)
    assert rows["id"] == ["a", "b"]
    assert rows["title"] == ["Title", "Title"]
    assert rows["context"] == ["one two", "one two"]
    assert rows["is_impossible"] == [False, True]
    assert rows["answers"] == [
        {"text": ["one", "two"], "answer_start": [0, 4]},
        {"text": [], "answer_start": []},
    ]
