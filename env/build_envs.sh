#!/usr/bin/env bash
set -euo pipefail

# Build the three isolated environments from their resolved lock files.
#
# Usage:
#   bash env/build_envs.sh [PROFILE]
#
# PROFILE:
#   omni | hf | fuse | all
#   default: all
#
# Examples:
#   bash env/build_envs.sh
#     -> build all three
#
#   bash env/build_envs.sh omni
#     -> build only the fairseq2/OmniASR environment

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
python_bin="${PYTHON311_BIN:-$HOME/.local/bin/python3.11}"
uv_cache_dir="${UV_CACHE_DIR:-$HOME/.cache/uv}"
profile="${1:-all}"

if [[ ! -x "${uv_bin}" ]]; then
  echo "uv not found at ${uv_bin}; install from https://docs.astral.sh/uv/ or set UV_BIN" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "python3.11 not found at ${python_bin}; run '${uv_bin} python install 3.11.13' or set PYTHON311_BIN" >&2
  exit 2
fi

case "${profile}" in
  omni|hf|fuse|all) ;;
  *) echo "usage: $0 {omni|hf|fuse|all}" >&2; exit 2 ;;
esac

mkdir -p "${repo_root}/.venvs" "${uv_cache_dir}"

build_profile() {
  local name="$1"
  local target="${repo_root}/.venvs/${name}"
  local lock="${repo_root}/env/locks/${name}.lock.txt"

  case "${target}" in
    "${repo_root}/.venvs/"*) ;;
    *) echo "unsafe environment target: ${target}" >&2; exit 3 ;;
  esac
  [[ -s "${lock}" ]] || { echo "missing lock: ${lock}" >&2; exit 3; }

  echo ">>> Building ${name}"
  if [[ ! -x "${target}/bin/python" ]]; then
    "${uv_bin}" --cache-dir "${uv_cache_dir}" venv --python "${python_bin}" "${target}"
  fi

  # --link-mode copy: the uv cache and this repository generally sit on
  # different filesystems, where hardlinking silently degrades.
  if [[ "${name}" == "omni" ]]; then
    "${uv_bin}" --cache-dir "${uv_cache_dir}" pip sync \
      --python "${target}/bin/python" "${lock}" \
      --index-strategy unsafe-best-match --link-mode copy
  else
    "${uv_bin}" --cache-dir "${uv_cache_dir}" pip sync \
      --python "${target}/bin/python" "${lock}" \
      --index https://download.pytorch.org/whl/cu124 \
      --index-strategy unsafe-best-match --link-mode copy
  fi
}

if [[ "${profile}" == "all" ]]; then
  for name in omni hf fuse; do build_profile "${name}"; done
else
  build_profile "${profile}"
fi

echo ">>> Environments built. Verify with: bash env/verify_envs.sh"
