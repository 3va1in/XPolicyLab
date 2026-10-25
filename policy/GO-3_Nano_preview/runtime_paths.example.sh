# Copy this template OUTSIDE the repository and fill in your machine paths.
# Default location: ${XDG_CONFIG_HOME:-$HOME/.config}/robotwin/runtime_paths.sh
# Alternatively set ROBOTWIN_RUNTIME_CONFIG to your private script.
export POLICY_ENV=""
export CKPT_PATH=""
export VIDEO_PRETRAIN_PATH=""
export ACTION_PRETRAIN_PATH=""
export VLM_MODEL_PATH=""
export VAE_PATH=""
# VAE latent statistics; distinct from bundled state/action statistics.
export STATS_TRACKER_PATH=""
# Training, validation and inference use assets/stats/robotwin.json directly.
# Only robotwin is supported; omit this variable to use the same default.
export ROBOT_NAME="robotwin"
# Optional overrides: CFG, DEPLOY_CONFIG, GPU_ID, HOST, PORT, SEED,
# RUN_ROOT, OUTPUT_DIR, CACHE_DIR, STARTUP_TIMEOUT, SAVE_DEBUG_VIDEO (0 by default; 1 enables CPU VAE debug video).
# Optional dataset metadata (unused by inference): TRAIN_JSONL_PATHS,
# DATA_VIDEO_ROOTS (JSON arrays); TASK_RECAP_PATH, STEP_RECAP_PATH.
# Optional evaluation paths: ROBOTWIN_ROOT, ROBOTWIN_ACTIVATE_SCRIPT, EVAL_ENV,
# MODEL_SERVER_HOST, MODEL_SERVER_PORT, EVAL_TASK, EPISODES, EVAL_GPU_ID, EVAL_SEED.
# EVAL_ENV_CFG_TYPE: the profile filename stem provided by your external simulator.

# SFT paths: source weights are distinct from the evaluation checkpoint.
export SFT_VIDEO_PRETRAIN_PATH=""
export SFT_ACTION_PRETRAIN_PATH=""
export SFT_JSONL_PATH=""
export SFT_DATA_ROOT=""
export SFT_VAL_JSONL_PATH=""
export SFT_VAL_DATA_ROOT=""
export TASK_RECAP_PATH=""
export STEP_RECAP_PATH=""
export SFT_PREPARED_DIR=""
# Optional: SFT_RUNTIME_CONFIG (a second private shell file), SFT_OUTPUT_DIR,
# SFT_STEPS, SFT_NPROC, SFT_CONFIG, XPL_ROOT, DEBUG_EPISODES.
