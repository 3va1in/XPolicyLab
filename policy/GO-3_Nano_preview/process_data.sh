#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 )); then
  echo 'Usage: process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expected_episode_count]' >&2
  exit 2
fi
[[ "${1,,}" == robotwin && "$4" == joint ]] || { echo 'Supported: processed RoboTwin / joint' >&2; exit 2; }
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
args=()
if [[ -n "${5:-}" ]]; then args+=(--expected-episodes "$5"); fi
exec "$POLICY_PYTHON" -m go3_nano_preview.sft.prepare_data \
  --jsonl "${SFT_JSONL_PATH:?Set SFT_JSONL_PATH}" --data-root "${SFT_DATA_ROOT:?Set SFT_DATA_ROOT}" \
  --output "${SFT_PREPARED_DIR:?Set SFT_PREPARED_DIR outside the checkout}" "${args[@]}"
