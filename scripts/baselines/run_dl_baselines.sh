#!/usr/bin/env bash
# Run DL baselines (CNN + Transformer) for all three target datasets.

set -euo pipefail

cd "$(dirname "$0")"

SEEDS="${SEEDS:-[42,43,44,45,46]}"
PROGRESS="${PROGRESS:-false}"

echo "=========================================="
echo "Running DL baselines on all datasets"
echo "=========================================="
echo "Seeds: ${SEEDS}"
echo ""

echo "1/3: polymerase"
uv run python ../../experiments/run_dl_baselines.py \
  dataset=polymerase \
  "seed_list=${SEEDS}" \
  "enable_progress_bar=${PROGRESS}"
echo ""

echo "2/3: promoters"
uv run python ../../experiments/run_dl_baselines.py \
  dataset=promoters \
  "seed_list=${SEEDS}" \
  "enable_progress_bar=${PROGRESS}"
echo ""

echo "3/3: mpra_enhancers"
uv run python ../../experiments/run_dl_baselines.py \
  dataset=mpra_enhancers \
  "seed_list=${SEEDS}" \
  "enable_progress_bar=${PROGRESS}"
echo ""

echo "=========================================="
echo "Done."
echo "Results written to:"
echo "  - ../../results/polymerase/baselines/"
echo "  - ../../results/promoters/baselines/"
echo "  - ../../results/mpra_enhancers/baselines/"
echo "=========================================="
