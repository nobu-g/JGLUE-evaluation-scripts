from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import wandb
from tabulate import tabulate
from wandb.apis.public import Run, Sweep

TASKS = ["marc_ja/accuracy", "jsts/spearman", "jnli/accuracy", "jsquad/exact_match", "jsquad/f1", "jcqa/accuracy"]
MODELS = ["roberta_base", "roberta_large", "deberta_base", "deberta_large"]


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
    for model in MODELS:
        items: list[Optional[RunSummary]] = []
        for task in TASKS:
            task, metric_name = task.split("/")
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
            [[model] + [item.metric if item else "-" for item in items] for model, items in zip(MODELS, table)],
            headers=["model"] + TASKS,
            tablefmt="github",
            floatfmt=".3f",
            colalign=["left"] + ["right"] * len(TASKS),
        )
    )
    print("Learning rates of best runs:")
    print(
        tabulate(
            [[model] + [item.lr if item else "-" for item in items] for model, items in zip(MODELS, table)],
            headers=["model"] + TASKS,
            tablefmt="github",
            colalign=["left"] + ["right"] * len(TASKS),
        )
    )
    print("Training epochs of best runs:")
    print(
        tabulate(
            [[model] + [item.max_epochs if item else "-" for item in items] for model, items in zip(MODELS, table)],
            headers=["model"] + TASKS,
            tablefmt="github",
            colalign=["left"] + ["right"] * len(TASKS),
        )
    )


if __name__ == "__main__":
    main()
