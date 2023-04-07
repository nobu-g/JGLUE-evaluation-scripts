#!/usr/bin/env bash

set -euo pipefail

for sweep in "$@"; do
  while read -r task_model sweep_id _; do
    if [[ "${sweep}" = "${task_model}" ]]; then
      wandb agent "${sweep_id}"
      break
    fi
  done < sweep_status.txt
done
