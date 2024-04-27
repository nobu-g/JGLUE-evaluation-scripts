from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import wandb
from tabulate import tabulate
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


def main():
    api = wandb.Api()
    name_to_sweep_path: dict[str, str] = {
        line.split()[0]: line.split()[1] for line in Path("sweep_status.txt").read_text().splitlines()
    }
    table: list[list[Optional[RunSummary]]] = []
    for model in MODELS.keys():
        items: list[Optional[RunSummary]] = []
        for task_and_metric in TASKS.keys():
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
        table.append(items)
    print("Scores of best runs:")
    print(
        tabulate(
            [
                [model] + [item.metric if item else "-" for item in items]
                for model, items in zip(MODELS.values(), table)
            ],
            headers=["Model", *TASKS.values()],
            tablefmt="github",
            floatfmt=".3f",
            colalign=["left"] + ["right"] * len(TASKS),
        )
    )
    print()
    print("Learning rates of best runs:")
    print(
        tabulate(
            [[model] + [item.lr if item else "-" for item in items] for model, items in zip(MODELS.values(), table)],
            headers=["Model", *TASKS.values()],
            tablefmt="github",
            colalign=["left"] + ["right"] * len(TASKS),
        )
    )
    print()
    print("Training epochs of best runs:")
    print(
        tabulate(
            [
                [model] + [item.max_epochs if item else "-" for item in items]
                for model, items in zip(MODELS.values(), table)
            ],
            headers=["Model", *TASKS.values()],
            tablefmt="github",
            colalign=["left"] + ["right"] * len(TASKS),
        )
    )


if __name__ == "__main__":
    main()
