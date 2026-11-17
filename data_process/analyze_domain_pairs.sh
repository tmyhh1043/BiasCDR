#!/usr/bin/env bash
set -uo pipefail

# Analyze the same 13 unordered domain pairs as preprocess_domain_pairs.sh.
# Usage: ./analyze_domain_pairs.sh [--dry-run] [--head-fraction N] [--random-state N]
# Results: cdr_eda_output/<source>_<target>/ (override with OUTPUT_ROOT).

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
datasets_dir="$script_dir/../datasets"
output_root="${OUTPUT_ROOT:-$script_dir/cdr_eda_output}"
python_bin="${PYTHON_BIN:-python3}"
dry_run=false
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi
analysis_args=("$@")

media_domains=(movie music book)
other_domains=(phone cloth elec sport toy)
total=0
failed=0

analyze_group() {
    local domains=("$@")
    local i j source target pair source_file target_file
    for ((i = 0; i < ${#domains[@]}; i++)); do
        for ((j = i + 1; j < ${#domains[@]}; j++)); do
            source="${domains[i]}"
            target="${domains[j]}"
            pair="${source}_${target}"
            source_file="$datasets_dir/$pair/$source/${source}_inter.csv"
            target_file="$datasets_dir/$pair/$target/${target}_inter.csv"
            ((total += 1))
            printf 'Analyzing %s\n' "$pair"

            if [[ $dry_run == true ]]; then
                printf '  source: %s\n  target: %s\n  output: %s\n' \
                    "$source_file" "$target_file" "$output_root/$pair"
                continue
            fi
            if [[ ! -f $source_file || ! -f $target_file ]]; then
                printf 'Missing interaction file for %s\n' "$pair" >&2
                ((failed += 1))
                continue
            fi
            if ! "$python_bin" "$script_dir/data_analysis.py" \
                "$source_file" "$target_file" "${analysis_args[@]}" \
                --output-dir "$output_root/$pair" \
                --source-label "$source" --target-label "$target"; then
                printf 'Analysis failed for %s\n' "$pair" >&2
                ((failed += 1))
            fi
        done
    done
}

analyze_group "${media_domains[@]}"
analyze_group "${other_domains[@]}"

if [[ $dry_run == false ]]; then
    printf 'Completed: %d/%d pairs\n' "$((total - failed))" "$total"
    ((failed == 0))
fi
