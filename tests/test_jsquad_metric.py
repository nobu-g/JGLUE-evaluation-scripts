import pytest
from torchmetrics import SQuAD

CASES = [
    {
        "preds": [
            {"prediction_text": "1976", "id": "000"},
        ],
        "target": [{"answers": {"answer_start": [97], "text": ["1976"]}, "id": "000"}],
        "exact_match": 1.0,
        "f1": 1.0,
    },
    {
        "preds": [
            {"prediction_text": "2 時間 21 分", "id": "001"},
        ],
        "target": [{"answers": {"answer_start": [10], "text": ["2 時間"]}, "id": "001"}],
        "exact_match": 0.0,
        "f1": 2 / 3,  # precision: 2 / 2, recall: 2 / 4
    },
]


@pytest.mark.parametrize("case", CASES)
def test_jsquad(case: dict):
    metric = SQuAD()
    metrics = metric(case["preds"], case["target"])
    assert metrics["exact_match"].item() / 100.0 == pytest.approx(case["exact_match"])
    assert metrics["f1"].item() / 100.0 == pytest.approx(case["f1"])
