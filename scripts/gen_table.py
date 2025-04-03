from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import wandb
from prettytable import PrettyTable

if TYPE_CHECKING:
    from wandb.apis.public import Run, Sweep

TASKS = {
    "marc_ja/accuracy": "MARC-ja/acc",
    "jcola/accuracy": "JCoLA/acc",
    "jsts/pearson": "JSTS/pearson",
    "jsts/spearman": "JSTS/spearman",
    "jnli/accuracy": "JNLI/acc",
    "jsquad/exact_match": "JSQuAD/EM",
    "jsquad/f1": "JSQuAD/F1",
    "jcqa/accuracy": "JComQA/acc",
}
MODELS = {
    "roberta_base": "Waseda RoBERTa base",  # nlp-waseda/roberta-base-japanese
    "roberta_large": "Waseda RoBERTa large (seq512)",  # nlp-waseda/roberta-large-japanese-seq512
    "deberta_base": "DeBERTaV2 base",  # ku-nlp/deberta-v2-base-japanese
    "deberta_large": "DeBERTaV2 large",  # ku-nlp/deberta-v2-large-japanese
    "deberta_v3_base": "DeBERTaV3 base",  # ku-nlp/deberta-v3-base-japanese
}


@dataclass(frozen=True)
class RunSummary:
    metric: float
    lr: float
    max_epochs: int
    batch_size: int


def create_table(headers: list[str], align: list[str]) -> PrettyTable:
    table = PrettyTable()
    table.field_names = headers
    for header, a in zip(headers, align):
        table.align[header] = a
    return table


def main() -> None:
    api = wandb.Api()
    name_to_sweep_path: dict[str, str] = {
        line.split()[0]: line.split()[1] for line in Path("sweep_status.txt").read_text().splitlines()
    }
    results: list[list[Optional[RunSummary]]] = []
    for model in MODELS:
        items: list[Optional[RunSummary]] = []
        for task_and_metric in TASKS:
            task, metric_name = task_and_metric.split("/")
            sweep: Sweep = api.sweep(name_to_sweep_path[f"{task}-{model}"])
            if sweep.state == "FINISHED":
                run: Optional[Run] = sweep.best_run()
                assert run is not None
                metric_name = "valid/" + metric_name
                items.append(
                    RunSummary(
                        metric=run.summary[metric_name],
                        lr=run.config["lr"],
                        max_epochs=run.config["max_epochs"],
                        batch_size=run.config["effective_batch_size"],
                    )
                )
            else:
                items.append(None)
        results.append(items)

    headers = ["Model", *TASKS.values()]
    align = ["l"] + ["r"] * len(TASKS)

    # スコアのテーブル
    print("Scores of best runs:")
    score_table = create_table(headers, align)
    for model, items in zip(MODELS.values(), results):
        row = [model] + [f"{item.metric:.3f}" if item else "-" for item in items]
        score_table.add_row(row)
    print(score_table)
    print()

    # 学習率のテーブル
    print("Learning rates of best runs:")
    lr_table = create_table(headers, align)
    for model, items in zip(MODELS.values(), results):
        row = [model] + [str(item.lr) if item else "-" for item in items]
        lr_table.add_row(row)
    print(lr_table)
    print()

    # エポック数のテーブル
    print("Training epochs of best runs:")
    epoch_table = create_table(headers, align)
    for model, items in zip(MODELS.values(), results):
        row = [model] + [str(item.max_epochs) if item else "-" for item in items]
        epoch_table.add_row(row)
    print(epoch_table)


if __name__ == "__main__":
    main()
