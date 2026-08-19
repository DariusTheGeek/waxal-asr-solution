#!/bin/bash
set -e

# Run the full pipeline: audio in, submission out.
#
# Usage:
#   bash run_inference.sh
#
# Reads runtime settings from:
#   configs/inference.yaml
#
# Writes submission to:
#   outputs/submissions/final_submission.csv
#
# Stages, each under the environment its dependencies require:
#   0  joint tag-free decode + text language ID   -> outputs/route/route.csv
#   1  per-model decode, per language lane        -> outputs/decodes/*.csv
#   2  TTIA profile matching                      -> outputs/ttia/keys_*.csv
#   3  TTIA (Lingala) and medoid (Shona) fusion   -> outputs/fused/*.csv
#   4  normalisation and submission write         -> outputs/submissions/
#
# Model weights must be present first:
#   python models/download_models.py

DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -x "$DIR/.venvs/fuse/bin/python" ]]; then
    echo "Missing environments. Run: bash install.sh"; exit 1
fi

"$DIR/.venvs/fuse/bin/python" "$DIR/inference/predict.py" "$@"
