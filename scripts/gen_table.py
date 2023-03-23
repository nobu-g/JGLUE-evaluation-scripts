from pathlib import Path
from typing import Optional

import wandb
from tabulate import tabulate
from wandb.apis.public import Run, Sweep

TASKS = ["marc_ja", "jsts", "jnli", "jcqa"]
MODELS = ["roberta_base", "roberta_large", "deberta_base", "deberta_large"]


def main():
    api = wandb.Api()
    name_to_sweep_path: dict[str, str] = {
        line.split()[0]: line.split()[1] for line in Path("sweep_status.txt").read_text().splitlines()
    }
    table = []
    for model in MODELS:
        items: list[str] = [model]
        for task in TASKS:
            sweep: Sweep = api.sweep(name_to_sweep_path[f"{task}-{model}"])
            run: Optional[Run] = sweep.best_run()
            if run is None:
                items.append("-")
            else:
                try:
                    metric_name = sweep.config["metric"]["name"]
                    items.append(f"{run.summary[metric_name]:.3f}")
                except KeyError:
                    items.append("-")
        table.append(items)
    print(tabulate(table, headers=["model"] + TASKS, tablefmt="github"))


if __name__ == "__main__":
    main()
