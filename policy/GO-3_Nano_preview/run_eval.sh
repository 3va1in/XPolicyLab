#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_CONFIG="${ROBOTWIN_RUNTIME_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/robotwin/runtime_paths.sh}"
if [[ -f "$MACHINE_CONFIG" ]]; then set -a; source "$MACHINE_CONFIG"; set +a; fi
for name in ROBOTWIN_ROOT ROBOTWIN_ACTIVATE_SCRIPT EVAL_ENV MODEL_SERVER_HOST EVAL_ENV_CFG_TYPE; do
  [[ -n "${!name:-}" ]] || { echo "Set $name in the private runtime configuration script." >&2; exit 1; }
done
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${CACHE_DIR:-${TMPDIR:-/tmp}/robotwin-eval-${UID}}"
export __GL_SHADER_DISK_CACHE_PATH="$XDG_CACHE_HOME/nvidia"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
mkdir -p "$__GL_SHADER_DISK_CACHE_PATH" "$CUDA_CACHE_PATH"
source "$ROBOTWIN_ACTIVATE_SCRIPT"
RUN_ROOT="${RUN_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/robotwin}"
RUN_DIR="$RUN_ROOT/eval_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR"
export RUN_DIR MODEL_SERVER_HOST
export EVAL_TASK="${EVAL_TASK:-adjust_bottle}"
export MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-18081}"
export EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
"$EVAL_ENV/bin/python" - <<'PY'
import os
from pathlib import Path
import yaml
cfg = {'gpu_ids': os.environ['EVAL_GPU_ID'], 'jobs_per_gpu': 1, 'num_workers': 1,
       'enable_remote': True, 'policy_server_ip': os.environ['MODEL_SERVER_HOST'],
       'policy_server_port': [int(os.environ['MODEL_SERVER_PORT'])],
       'tasks': [os.environ['EVAL_TASK']]}
(Path(os.environ['RUN_DIR']) / 'single_task.yml').write_text(yaml.safe_dump(cfg))
PY
cd "$ROBOTWIN_ROOT"
exec bash scripts/eval_policy.sh multitask --config "$RUN_DIR/single_task.yml" \
  --policy-name "$(basename "$ROOT")" --ckpt-name standalone_wam --env-cfg-type "$EVAL_ENV_CFG_TYPE" --action-type joint \
  --eval-env-conda-env "$EVAL_ENV" --task-config "${TASK_CONFIG:-demo_randomized}" \
  --test-num "${EPISODES:-10}" --seed "${EVAL_SEED:-0}" --output-dir "$RUN_DIR/results" --stream-output "$@"
