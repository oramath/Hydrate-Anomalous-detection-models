#!/usr/bin/env bash
# Download 3W 2.0.0 (about 1.8 GB of Parquet files) from the Petrobras GitHub repository.
# The same data are archived on Figshare: https://doi.org/10.6084/m9.figshare.29205836.v1
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST" && cd "$DEST"
git -c gc.auto=0 clone --depth 1 --filter=blob:none --sparse https://github.com/petrobras/3W.git 3W
cd 3W
git -c gc.auto=0 sparse-checkout set dataset
grep -E "^DATASET" dataset/dataset.ini
echo "export H3W_DATA=$(pwd)/dataset"
