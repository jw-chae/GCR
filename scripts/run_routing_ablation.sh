#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_routing_ablation.sh /path/to/mvtec_ad
DATA_ROOT="${1:?Usage: $0 /path/to/mvtec_ad}"

# Common settings
MEMORY_BANK_SIZE=196

python cli/run.py \
  --data_root "${DATA_ROOT}" \
  --save_root ./results/gs_an_continual_score_k196 \
  --continual \
  --memory_bank_size ${MEMORY_BANK_SIZE} \
  --routing_rule score

python cli/run.py \
  --data_root "${DATA_ROOT}" \
  --save_root ./results/gs_an_continual_geom_k196 \
  --continual \
  --memory_bank_size ${MEMORY_BANK_SIZE} \
  --routing_rule geometry
