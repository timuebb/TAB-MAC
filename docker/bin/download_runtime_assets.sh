#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  docker/bin/download_runtime_assets.sh dataset URL [TARGET_DIR]
  docker/bin/download_runtime_assets.sh llm-checkpoints URL [TARGET_DIR]
  docker/bin/download_runtime_assets.sh pretrain-checkpoints URL [TARGET_DIR]

The URL must point to a downloadable .tar.gz, .tgz, .zip, or plain file.
The target directory is created if it does not exist.
EOF
}

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    usage >&2
    exit 2
fi

asset_type="$1"
url="$2"

case "$asset_type" in
    dataset)
        target_dir="${3:-/app/dataset}"
        ;;
    llm-checkpoints)
        target_dir="${3:-/app/ts_benchmark/baselines/LLM/checkpoints}"
        ;;
    pretrain-checkpoints)
        target_dir="${3:-/app/ts_benchmark/baselines/pre_train/checkpoints}"
        ;;
    *)
        echo "Unknown asset type: $asset_type" >&2
        usage >&2
        exit 2
        ;;
esac

tmp_file="/tmp/$(basename "${url%%\?*}")"
if [ "$tmp_file" = "/tmp/" ] || [ "$tmp_file" = "/tmp/download" ]; then
    tmp_file="/tmp/${asset_type}.download"
fi

mkdir -p "$target_dir"

echo "Downloading $asset_type to $target_dir"
curl --fail --location --retry 5 --connect-timeout 30 --output "$tmp_file" "$url"

case "$tmp_file" in
    *.tar.gz|*.tgz)
        tar -xzf "$tmp_file" -C "$target_dir"
        ;;
    *.zip)
        python -m zipfile -e "$tmp_file" "$target_dir"
        ;;
    *)
        mv "$tmp_file" "$target_dir/"
        tmp_file=""
        ;;
esac

if [ -n "$tmp_file" ]; then
    rm -f "$tmp_file"
fi

echo "Done."
