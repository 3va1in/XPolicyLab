#!/usr/bin/env bash
set -euo pipefail
if (( $# < 6 )); then
  echo 'Usage: train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [trainer arguments]' >&2
  exit 2
fi
bench_name=$1; ckpt_name=$2; env_cfg_type=$3; action_type=$4; seed=$5; gpu_id=$6
shift 6
[[ "${bench_name,,}" == robotwin && "$action_type" == joint ]] || { echo 'Supported: RoboTwin / joint' >&2; exit 2; }
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
[[ "$setting" != */* && "$setting" != *..* ]] || { echo 'Use a simple checkpoint name' >&2; exit 2; }
output="${SFT_OUTPUT_DIR:-${RUN_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/robotwin}/checkpoints/${setting}}"
export CUDA_VISIBLE_DEVICES="$gpu_id"
exec "$POLICY_PYTHON" -m torch.distributed.run --standalone --nproc-per-node="${SFT_NPROC:-1}" \
  -m go3_nano_preview.sft.train --config "${SFT_CONFIG:-${POLICY_DIR}/configs/robotwin_sft.py}" \
  --output "$output" --steps "${SFT_STEPS:-10000000}" --seed "$seed" "$@"
