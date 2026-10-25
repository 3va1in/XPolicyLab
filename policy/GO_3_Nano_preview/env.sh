#!/usr/bin/env bash
# Shared runtime setup. Keep the sourced configuration outside the checkout.
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_config="${ROBOTWIN_RUNTIME_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/robotwin/runtime_paths.sh}"
if [[ -f "$runtime_config" ]]; then set -a; source "$runtime_config"; set +a; fi
if [[ -n "${SFT_RUNTIME_CONFIG:-}" ]]; then set -a; source "$SFT_RUNTIME_CONFIG"; set +a; fi
POLICY_PYTHON="${POLICY_ENV:?Set POLICY_ENV to the Python environment directory}/bin/python"
[[ -x "$POLICY_PYTHON" ]] || { echo 'POLICY_ENV/bin/python is not executable' >&2; exit 1; }
export PYTHONPATH="${POLICY_DIR}:${POLICY_DIR}/go3_nano_preview${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export STAT_PATH="${POLICY_DIR}/assets/stats/robotwin.json"
