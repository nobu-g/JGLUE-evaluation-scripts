#!/usr/bin/env bash

set -euo pipefail

for task in marc_ja jcola jsts jnli jsquad jcqa; do
  for model in roberta_base roberta_large deberta_base deberta_large; do
    sweep_id=$(wandb sweep --name="${task}-${model}" "sweeps/${task}/${model}.yaml" 2>&1 | tail -1 | cut -d' ' -f8)
    echo "${task}-${model}" "${sweep_id}" 0 | tee -a sweep_status.txt
  done
done
