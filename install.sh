#!/bin/bash
set -euo pipefail

# Build every environment this solution needs, then verify them.
#
# Usage:
#   bash install.sh
#
# Creates three isolated virtual environments under .venvs/:
#   omni  Torch 2.8.0 + fairseq2      -- the six OmniASR models
#   hf    Torch 2.5.1 + transformers  -- the two MMS-1B models
#   fuse  CPU only                    -- routing, fusion, post-processing
#
# They cannot be merged: omni and hf pin incompatible Torch builds.
#
# Requires uv and Python 3.11.13. If uv is absent:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   uv python install 3.11.13

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Building environments ==="
bash "$DIR/env/build_envs.sh" all

echo "=== Verifying environments ==="
bash "$DIR/env/verify_envs.sh"

echo ">>> Install complete. Evidence in env/health/"
