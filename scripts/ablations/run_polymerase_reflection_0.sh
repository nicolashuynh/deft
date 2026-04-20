#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS=(42 43 44 45 46)
if [[ -n "${DEFT_SEEDS:-}" ]]; then
  # Space-separated list, e.g. DEFT_SEEDS="42 43"
  # shellcheck disable=SC2206
  SEEDS=(${DEFT_SEEDS})
fi

for seed in "${SEEDS[@]}"; do
  echo "[Ablation reflection_0 | polymerase] seed=${seed}"
  uv run python experiments/deft_tree.py \
    params_generation.n_prompts=1 \
    params_generation.population_size=10 \
    params_generation.n_parents_per_prompt=10 \
    params_generation.n_reflections=0 \
    random_state="${seed}" \
    type_dataset="Plus" \
    max_depth=10 \
    name_experiment="reflection_0" \
    "$@"
done
