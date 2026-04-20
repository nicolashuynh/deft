#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SEEDS=(42 43 44 45 46)
if [[ -n "${DEFT_SEEDS:-}" ]]; then
  # Space-separated list, e.g. DEFT_SEEDS="42 43"
  # shellcheck disable=SC2206
  SEEDS=(${DEFT_SEEDS})
fi

for seed in "${SEEDS[@]}"; do
  echo "[DEFT_perf polymerase | deft_perf_depth_10] seed=${seed}"
  uv run python ../../experiments/deft_tree.py \
    --config-name=config_deft_perf \
    params_generation.n_prompts=1 \
    params_generation.population_size=10 \
    params_generation.n_parents_per_prompt=10 \
    random_state="${seed}" \
    type_dataset="Plus" \
    max_depth=10 \
    name_experiment="deft_perf_depth_10" \
    "$@"
done
