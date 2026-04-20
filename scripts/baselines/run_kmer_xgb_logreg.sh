#!/usr/bin/env bash


set -euo pipefail

cd "$(dirname "$0")"

SEEDS="${SEEDS:-[42,43,44,45,46]}"
MAX_DEPTH="${MAX_DEPTH:-6}"
KMER_SIZE="${KMER_SIZE:-2}"

LOGREG_C_POLYMERASE="${LOGREG_C_POLYMERASE:-100.0}"
LOGREG_C_PROMOTERS="${LOGREG_C_PROMOTERS:-1.0}"
LOGREG_C_MPRA_ENHANCERS="${LOGREG_C_MPRA_ENHANCERS:-0.1}"

run_dataset() {
  local dataset="$1"
  local logreg_c="$2"

  echo "------------------------------------------"
  echo "Dataset: ${dataset}"
  echo "  Seeds: ${SEEDS}"
  echo "  Max depth (XGBoost): ${MAX_DEPTH}"
  echo "  K-mer size: ${KMER_SIZE}"
  echo "  LogReg C: ${logreg_c}"
  echo "------------------------------------------"

  echo "  [1/2] XGBoost + 2-mers"
  uv run python ../../experiments/run_baselines.py \
    dataset="${dataset}" \
    baseline_model=xgboost \
    featurizer=kmer_count \
    "seed_list=${SEEDS}" \
    "baseline_model.max_depth=${MAX_DEPTH}" \
    "featurizer.k=${KMER_SIZE}" \
    "output_file_template=xgboost__kmer_count__k${KMER_SIZE}.csv"

  echo "  [2/2] LogReg + 2-mers"
  uv run python ../../experiments/run_baselines.py \
    dataset="${dataset}" \
    baseline_model=logreg \
    featurizer=kmer_count \
    "seed_list=${SEEDS}" \
    "baseline_model.C_values=[${logreg_c}]" \
    "baseline_model.C=${logreg_c}" \
    "featurizer.k=${KMER_SIZE}" \
    "output_file_template=logreg__kmer_count__k${KMER_SIZE}.csv"

  echo ""
}

echo "=========================================="
echo "Running 2-mer XGBoost + LogReg baselines"
echo "=========================================="
echo ""

run_dataset polymerase "${LOGREG_C_POLYMERASE}"
run_dataset promoters "${LOGREG_C_PROMOTERS}"
run_dataset mpra_enhancers "${LOGREG_C_MPRA_ENHANCERS}"

echo "=========================================="
echo "Done. Results written under:"
echo "  - ../../results/polymerase/baselines/"
echo "  - ../../results/promoters/baselines/"
echo "  - ../../results/mpra_enhancers/baselines/"
echo "Files per dataset:"
echo "  - xgboost__kmer_count__k${KMER_SIZE}.csv"
echo "  - logreg__kmer_count__k${KMER_SIZE}.csv"
echo "=========================================="
