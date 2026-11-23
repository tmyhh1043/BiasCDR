#!/usr/bin/env bash
# Train popularity-only Selective-CL for six available source->target directions.
# Run from any directory. Set SEED, PYTHON_BIN, GPU_ID, or DRY_RUN as needed.
set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

seed="${SEED:-2022}"
python_bin="${PYTHON_BIN:-python3}"
gpu_id="${GPU_ID:-0}"
dry_run="${DRY_RUN:-0}"
log_dir="log/proposed_popularity_only_cl/seed${seed}"

if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    printf 'SEED must be a nonnegative integer: %s\n' "$seed" >&2
    exit 2
fi
if [[ "$dry_run" != 0 && "$dry_run" != 1 ]]; then
    printf 'DRY_RUN must be 0 or 1: %s\n' "$dry_run" >&2
    exit 2
fi

directions=(
    AmazonPhone_AmazonSport
    AmazonSport_AmazonPhone
    AmazonSport_AmazonToy
    AmazonToy_AmazonSport
    AmazonElec_AmazonCloth
    AmazonCloth_AmazonElec
)

for direction in "${directions[@]}"; do
    config="recbole_cdr/properties/dataset/${direction}.yaml"
    interactions="dataset/${direction}/${direction}.inter"
    if [[ ! -f "$config" || ! -f "$interactions" ]]; then
        printf 'Missing input for %s: %s or %s\n' "$direction" "$config" "$interactions" >&2
        exit 1
    fi
done

if [[ "$dry_run" == 0 ]]; then
    if ! command -v "$python_bin" >/dev/null 2>&1; then
        printf 'Python executable not found: %s\n' "$python_bin" >&2
        exit 1
    fi
    if ! "$python_bin" -c 'import torch, scipy, recbole, yaml' >/dev/null 2>&1; then
        printf 'Required training packages are missing from %s (torch, scipy, recbole, PyYAML).\n' "$python_bin" >&2
        exit 1
    fi
    mkdir -p "$log_dir"
fi

printf 'Popularity-only CL: six directions, seed %s\n' "$seed"
for index in "${!directions[@]}"; do
    direction="${directions[$index]}"
    command=(
        "$python_bin" run_recbole_cdr.py
        --model Proposed
        --dataset "$direction"
        --seed "$seed"
        --gpu_id "$gpu_id"
        --popularity_only_cl
    )
    printf '[%d/%d] %s\n' "$((index + 1))" "${#directions[@]}" "$direction"
    if [[ "$dry_run" == 1 ]]; then
        printf '  '
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}" 2>&1 | tee "$log_dir/${direction}.log"
    fi
done

if [[ "$dry_run" == 0 ]]; then
    printf 'Completed. Metrics: results/proposed_popularity_only_cl/test_metrics.csv\n'
    printf 'Checkpoints: saved/proposed_popularity_only_cl/\n'
    printf 'Logs: %s/\n' "$log_dir"
fi
