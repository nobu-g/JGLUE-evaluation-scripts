#!/usr/bin/env bash

set -euo pipefail

for sweep in "$@"; do
  while read -r task_model sweep_id status; do
    if [[ "${sweep}" = "${task_model}" && "${status}" = "0" ]]; then
      sed -i "s|${task_model} ${sweep_id} 0|${task_model} ${sweep_id} 1|" sweep_status.txt
      wandb agent "${sweep_id}"
      break
    fi
  done < sweep_status.txt
done
