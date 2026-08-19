#!/usr/bin/env bash
set -euo pipefail

# Probe every environment and write the reproducibility evidence under env/health/.
#
# Usage:
#   bash env/verify_envs.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
health_dir="${repo_root}/env/health"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
uv_cache="${UV_CACHE_DIR:-$HOME/.cache/uv}"
mkdir -p "${health_dir}"

status=0
for profile in omni hf fuse; do
  python_bin="${repo_root}/.venvs/${profile}/bin/python"
  [[ -x "${python_bin}" ]] || { echo "missing environment: ${profile} (run env/build_envs.sh)" >&2; exit 2; }
  "${python_bin}" "${repo_root}/env/healthcheck.py" \
    --profile "${profile}" --output "${health_dir}/${profile}_health.json" || status=1
  "${uv_bin}" --cache-dir "${uv_cache}" pip freeze --python "${python_bin}" \
    | LC_ALL=C sort > "${health_dir}/${profile}_installed.txt"
done

{
  echo "created_at=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
  echo "python=$("${repo_root}/.venvs/fuse/bin/python" --version 2>&1)"
  echo "uv=$("${uv_bin}" --version 2>&1)"
  echo "ffmpeg=$(ffmpeg -version 2>/dev/null | head -n 1)"
  echo "kernel=$(uname -srmo)"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
} > "${health_dir}/system_inventory.txt"

(
  cd "${repo_root}"
  sha256sum env/requirements-*.txt env/locks/*.lock.txt \
    env/health/*_health.json env/health/*_installed.txt \
    env/health/system_inventory.txt \
    | LC_ALL=C sort > env/health/SHA256SUMS
)

echo ">>> Evidence written to env/health/"
exit "${status}"
