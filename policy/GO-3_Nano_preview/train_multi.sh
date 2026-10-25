#!/usr/bin/env bash
set -euo pipefail
# Keep this entry point independent of XPolicyLab's root scripts and cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
exec "$POLICY_PYTHON" -m go3_nano_preview.sft.launch \
  --env GO-3_Nano_preview --mode train \
  --run_name cotrain_robotwin_sft_ka_robotype \
  --overwrite True --debug false --seed 1016 "$@"
