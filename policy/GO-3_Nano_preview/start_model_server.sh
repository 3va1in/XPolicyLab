#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-start}"
MACHINE_CONFIG="${ROBOTWIN_RUNTIME_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/robotwin/runtime_paths.sh}"
if [[ -f "$MACHINE_CONFIG" ]]; then
  set -a
  source "$MACHINE_CONFIG"
  set +a
fi
PYTHON="${POLICY_ENV:-}/bin/python"
CFG="${CFG:-$ROOT/configs/robotwin_infer.py}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$ROOT/configs/deploy.yml}"
STAT_PATH="$ROOT/assets/stats/robotwin.json"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18081}"
SEED="${SEED:-42}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
RUN_ROOT="${RUN_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/robotwin}"
CACHE_DIR="${CACHE_DIR:-${TMPDIR:-/tmp}/robotwin-cache-${UID}}"
LOG_DIR="$RUN_ROOT/logs"
PID_FILE="$LOG_DIR/model_server.pid"
LOG_FILE="$LOG_DIR/model_server.log"
ENTRY="$ROOT/go3_nano_preview/serve_GO-3.py"
mkdir -p "$LOG_DIR"

owned_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid command_line
  read -r pid < "$PID_FILE"
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  if [[ "$command_line" == *"$ROOT/start_model_server.sh foreground"* || "$command_line" == *"$ENTRY"* ]]; then
    printf '%s\n' "$pid"
  else
    return 1
  fi
}

health() {
  "$PYTHON" - "$HOST" "$PORT" <<'PY'
import sys, urllib.request
host = '127.0.0.1' if sys.argv[1] in ('0.0.0.0', 'localhost') else sys.argv[1]
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://{host}:{sys.argv[2]}/healthz', timeout=2) as response:
        assert response.status == 200 and response.read().strip() == b'OK'
except Exception:
    raise SystemExit(1)
PY
}

check_files() {
  for name in POLICY_ENV CKPT_PATH VIDEO_PRETRAIN_PATH ACTION_PRETRAIN_PATH VLM_MODEL_PATH VAE_PATH STATS_TRACKER_PATH; do
    [[ -n "${!name:-}" ]] || { echo "Set $name in your private runtime configuration script." >&2; exit 1; }
  done
  [[ -x "$PYTHON" ]] || { echo 'POLICY_ENV must contain an executable bin/python.' >&2; exit 1; }
  for file in "$ENTRY" "$CFG" "$DEPLOY_CONFIG" "$STAT_PATH" \
      "$CKPT_PATH/config.json" "$CKPT_PATH/diffusion_pytorch_model.bin" \
      "$CKPT_PATH/action_model/config.json" "$CKPT_PATH/action_model/diffusion_pytorch_model.bin" \
      "$VLM_MODEL_PATH/config.json" "$VAE_PATH/config.json" \
      "$VIDEO_PRETRAIN_PATH" "$ACTION_PRETRAIN_PATH" "$STATS_TRACKER_PATH"; do
    [[ -s "$file" ]] || { echo "Missing or empty file: $file" >&2; exit 1; }
  done
  "$PYTHON" "$ROOT/go3_nano_preview/check_checkpoint.py" --config "$CFG" --ckpt "$CKPT_PATH"
}

case "$MODE" in
  check) check_files; echo 'Files and checkpoint configuration OK.' ;;
  status)
    if pid="$(owned_pid)"; then
      echo "Running: PID=$pid; log=$LOG_FILE"
      if health; then echo "Ready: ws://$HOST:$PORT"; else echo 'Loading or unhealthy.'; fi
    else echo 'Stopped.'; exit 1; fi
    ;;
  stop)
    if pid="$(owned_pid)"; then
      kill -TERM "$pid"
      for ((i=0; i<30; i++)); do
        if ! owned_pid >/dev/null; then break; fi
        sleep 1
      done
      if owned_pid >/dev/null; then echo 'Server has not exited yet.' >&2; exit 1; fi
      rm -f -- "$PID_FILE"
      echo "Stopped PID $pid."
    else echo 'No server owned by this script is running.'; fi
    ;;
  foreground)
    check_files
    export VIDEO_PRETRAIN_PATH ACTION_PRETRAIN_PATH VLM_MODEL_PATH VAE_PATH STATS_TRACKER_PATH STAT_PATH
    export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    # Reuse dependency bytecode; PYTHONDONTWRITEBYTECODE prevents repository caches.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export PYTHONPATH="$ROOT:$ROOT/go3_nano_preview${PYTHONPATH:+:${PYTHONPATH}}"
    # Debug video is optional and decoded on CPU. Action inference stays on GPU.
    case "${SAVE_DEBUG_VIDEO:-0}" in
      0) unset WAM_policy_OPENLOOP_DEBUG_VIDEO_DIR ;;
      1)
        export WAM_policy_OPENLOOP_DEBUG_VIDEO_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
        mkdir -p "$WAM_policy_OPENLOOP_DEBUG_VIDEO_DIR"
        ;;
      *) echo 'SAVE_DEBUG_VIDEO must be 0 or 1.' >&2; exit 2 ;;
    esac
    export WAM_policy_INFER_LEGACY_STATE_LAYOUT=identity
    export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_DIR/cuda}"
    mkdir -p "$CUDA_CACHE_PATH"
    cd "$ROOT"
    exec env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$ENTRY" \
      --deploy-config "$DEPLOY_CONFIG" --cfg-path "$CFG" --ckpt-path "$CKPT_PATH" \
      --stat-path "$STAT_PATH" --robot-name "${ROBOT_NAME:-robotwin}" --dense-action-steps 52 \
      --host "$HOST" --port "$PORT" --seed "$SEED"
    ;;
  start)
    check_files
    if pid="$(owned_pid)"; then echo "Already running: PID=$pid"; exit 0; fi
    "$PYTHON" - "$HOST" "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind((sys.argv[1], int(sys.argv[2])))
PY
    [[ ! -f "$LOG_FILE" ]] || mv -- "$LOG_FILE" "$LOG_DIR/model_server_$(date +%Y%m%d_%H%M%S)_$$.log"
    export ROBOTWIN_RUNTIME_CONFIG="$MACHINE_CONFIG"
    export GPU_ID HOST PORT SEED STARTUP_TIMEOUT RUN_ROOT CACHE_DIR CFG DEPLOY_CONFIG STAT_PATH
    nohup bash "$ROOT/start_model_server.sh" foreground > "$LOG_FILE" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$PID_FILE"
    echo "Starting PID=$pid; log=$LOG_FILE"
    started=$SECONDS
    while (( SECONDS-started < STARTUP_TIMEOUT )); do
      if ! kill -0 "$pid" 2>/dev/null; then tail -n 50 "$LOG_FILE" >&2; exit 1; fi
      if health; then echo "Ready: ws://$HOST:$PORT"; exit 0; fi
      sleep 2
    done
    if owned_pid >/dev/null; then kill -TERM "$pid"; fi
    echo 'Readiness timeout; server stopped.' >&2
    exit 1
    ;;
  *) echo 'Usage: bash start_model_server.sh [start|foreground|status|stop|check]'; exit 2 ;;
esac
