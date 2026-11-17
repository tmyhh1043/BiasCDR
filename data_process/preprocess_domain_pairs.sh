#!/usr/bin/env bash
set -euo pipefail

# Usage: ./preprocess_domain_pairs.sh [--dry-run] [data_preprocess.py options]
# Each unordered pair is processed once; the first domain is the source.
# k-core defaults to 5 users / 5 items; --k, --k_user, and --k_item can override it.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
dry_run=false
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi

preprocess_args=("$@")
media_domains=(movie music book)
other_domains=(phone cloth elec sport toy)

process_group() {
    local domains=("$@")
    local i j source target
    for ((i = 0; i < ${#domains[@]}; i++)); do
        for ((j = i + 1; j < ${#domains[@]}; j++)); do
            source="${domains[i]}"
            target="${domains[j]}"
            printf 'Processing %s_%s\n' "$source" "$target"
            if [[ $dry_run == false ]]; then
                "$python_bin" "$script_dir/data_preprocess.py" \
                    --source "$source" --target "$target" \
                    --k_user 5 --k_item 5 "${preprocess_args[@]}"
            fi
        done
    done
}

process_group "${media_domains[@]}"
process_group "${other_domains[@]}"
