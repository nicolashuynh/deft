#!/usr/bin/env bash


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS:-[42,43,44,45,46]}"
MAX_DEPTH="${MAX_DEPTH:-6}"

OC1_TRIES_POLYMERASE="${OC1_TRIES_POLYMERASE:-50}"
OC1_TRIES_OTHER="${OC1_TRIES_OTHER:-20}"

# Number of dataset jobs to run concurrently.
if command -v nproc >/dev/null 2>&1; then
  DEFAULT_WORKERS="$(nproc)"
else
  DEFAULT_WORKERS="2"
fi
DATASET_WORKERS="${DATASET_WORKERS:-$DEFAULT_WORKERS}"
if (( DATASET_WORKERS < 1 )); then
  DATASET_WORKERS=1
fi

LOG_DIR="${LOG_DIR:-../../results/_logs/baselines_aistats_2026_parallel_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

declare -a PIDS=()
declare -a DATASETS=()
declare -a LOGS=()

cleanup_on_error() {
  local code=$?
  if (( code != 0 )); then
    echo "Error detected. Stopping remaining background jobs..."
    if jobs -pr >/dev/null 2>&1; then
      jobs -pr | xargs -r kill || true
    fi
  fi
}
trap cleanup_on_error EXIT

run_dataset() {
  local dataset="$1"
  local oc1_tries="$2"
  local oc1_seed_mode="$3"

  echo "------------------------------------------"
  echo "Dataset: ${dataset}"
  echo "  Seeds: ${SEEDS}"
  echo "  Max depth: ${MAX_DEPTH}"
  echo "  OC1 tries: ${oc1_tries}"
  echo "  OC1 random_state_from_seed: ${oc1_seed_mode}"
  echo "------------------------------------------"

  echo "  [1/3] CART (identity featurizer)"
  uv run python ../../experiments/run_baselines.py \
    dataset="${dataset}" \
    baseline_model=cart \
    featurizer=identity \
    "seed_list=${SEEDS}" \
    "baseline_model.max_depth=${MAX_DEPTH}" \
    "output_file_template=cart__identity.csv"

  echo "  [2/3] OC1 (identity featurizer)"
  uv run python ../../experiments/run_baselines.py \
    dataset="${dataset}" \
    baseline_model=oc1 \
    featurizer=identity \
    "seed_list=${SEEDS}" \
    "baseline_model.max_depth=${MAX_DEPTH}" \
    "baseline_model.num_tries=${oc1_tries}" \
    "baseline_model.random_state_from_seed=${oc1_seed_mode}" \
    "output_file_template=oc1__identity.csv"

  echo "  [3/3] 2-mer tree baseline (CART + kmer_count, k=2)"
  uv run python ../../experiments/run_baselines.py \
    dataset="${dataset}" \
    baseline_model=cart \
    featurizer=kmer_count \
    "seed_list=${SEEDS}" \
    "baseline_model.max_depth=${MAX_DEPTH}" \
    featurizer.k=2 \
    "output_file_template=cart__kmer_count__k2.csv"

  echo "Dataset ${dataset}: done"
}

wait_one_oldest() {
  local pid="${PIDS[0]}"
  local dataset="${DATASETS[0]}"
  local logfile="${LOGS[0]}"

  if wait "$pid"; then
    echo "[OK] ${dataset} completed. Log: ${logfile}"
  else
    echo "[FAIL] ${dataset} failed. Log: ${logfile}"
    echo "---- Last 80 log lines (${dataset}) ----"
    tail -n 80 "$logfile" || true
    exit 1
  fi

  PIDS=("${PIDS[@]:1}")
  DATASETS=("${DATASETS[@]:1}")
  LOGS=("${LOGS[@]:1}")
}

wait_for_slot() {
  while (( ${#PIDS[@]} >= DATASET_WORKERS )); do
    wait_one_oldest
  done
}

launch_dataset() {
  local dataset="$1"
  local oc1_tries="$2"
  local oc1_seed_mode="$3"
  local logfile="${LOG_DIR}/${dataset}.log"

  (
    run_dataset "$dataset" "$oc1_tries" "$oc1_seed_mode"
  ) >"$logfile" 2>&1 &

  local pid=$!
  echo "[START] ${dataset} (pid=${pid}) log=${logfile}"
  PIDS+=("$pid")
  DATASETS+=("$dataset")
  LOGS+=("$logfile")
}

echo "=========================================="
echo "Running unified baselines for AISTATS plots (parallel)"
echo "=========================================="
echo "Workers: ${DATASET_WORKERS}"
echo "Logs: ${LOG_DIR}"
echo ""

launch_dataset polymerase "${OC1_TRIES_POLYMERASE}" false
wait_for_slot

launch_dataset promoters "${OC1_TRIES_OTHER}" true
wait_for_slot
launch_dataset mpra_easy "${OC1_TRIES_OTHER}" true
wait_for_slot
launch_dataset mpra_enhancers "${OC1_TRIES_OTHER}" true
wait_for_slot

# Wait remaining jobs.
while (( ${#PIDS[@]} > 0 )); do
  wait_one_oldest
done

echo "=========================================="
echo "Done. Results written under:"
echo "  - ../../results/polymerase/baselines/"
echo "  - ../../results/promoters/baselines/"
echo "  - ../../results/mpra_easy/baselines/"
echo "  - ../../results/mpra_enhancers/baselines/"
echo "Files per dataset:"
echo "  - cart__identity.csv"
echo "  - oc1__identity.csv"
echo "  - cart__kmer_count__k2.csv"
echo "Logs: ${LOG_DIR}"
echo "=========================================="
