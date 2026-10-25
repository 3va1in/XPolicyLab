#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"
source "$POLICY_DIR/configs/runtime.conf"
if (( $# < 9 || $# > 10 )); then
    echo 'Usage: setup_eval_policy_server.sh <bench> <task> <ckpt> <profile> <action> <seed> <gpu> <policy_env> <port> [host]' >&2
    exit 2
fi

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

if [[ -x "${policy_conda_env}/bin/python" ]]; then
    policy_python="${policy_conda_env}/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    policy_python="$(command -v python)"
fi
export PYTHONPATH="${XPL_ROOT}/..:${SCRIPT_DIR}:${SCRIPT_DIR}/go3_nano_preview${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export STAT_PATH="${SCRIPT_DIR}/assets/stats/robotwin.json"
export WAM_policy_INFER_LEGACY_STATE_LAYOUT=identity
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false


exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${policy_python}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}"
