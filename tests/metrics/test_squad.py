from typing import Any

import pytest
from torchmetrics.text import SQuAD

CASES = [
    {
        "preds": [
            {"prediction_text": "1976", "id": "000"},
        ],
        "target": [{"answers": {"answer_start": [97], "text": ["1976"]}, "id": "000"}],
        "exact_match": 1.0,
        "f1": 1.0,  # precision: 1 / 1, recall: 1 / 1
    },
    {
        "preds": [
            {"prediction_text": "2 時間 21 分", "id": "001"},
        ],
        "target": [{"answers": {"answer_start": [10], "text": ["2 時間"]}, "id": "001"}],
        "exact_match": 0.0,
        "f1": 2 / 3,  # precision: 2 / 4, recall: 2 / 2
    },
    {
        "preds": [
            {"prediction_text": "2 時間 21 分", "id": "001"},
        ],
        "target": [{"answers": {"answer_start": [10, 10], "text": ["2 時間", "2 時間 21 分"]}, "id": "001"}],
        "exact_match": 1.0,
        "f1": 1.0,  # precision: 4 / 4, recall: 4 / 4
    },
    {
        "preds": [
            {"prediction_text": "2 時間 21 分", "id": "001"},
        ],
        "target": [{"answers": {"answer_start": [10, 12], "text": ["2 時間", "時間 21 分"]}, "id": "001"}],
        "exact_match": 0.0,
        "f1": 6 / 7,  # precision: 3 / 4, recall: 3 / 3
    },
    {
        "preds": [
            {"prediction_text": "2 時 間 2 1 分", "id": "001"},
        ],
        "target": [{"answers": {"answer_start": [10], "text": ["2 時 間"]}, "id": "001"}],
        "exact_match": 0.0,
        "f1": 2 / 3,  # precision: 3 / 6, recall: 3 / 3
    },
]


@pytest.mark.parametrize("case", CASES)
def test_jsquad(case: dict[str, Any]):
    metric = SQuAD()
    metrics = metric(case["preds"], case["target"])
    assert metrics["exact_match"].item() / 100.0 == pytest.approx(case["exact_match"])
    assert metrics["f1"].item() / 100.0 == pytest.approx(case["f1"])
