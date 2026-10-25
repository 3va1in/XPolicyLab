# GO-3 Nano Preview

Python source code is stored directly in `policy/GO_3_Nano_preview/go3_nano_preview/`, which is also the `go3_nano_preview` import package; no separate forwarding package is needed. The standalone server file remains `serve_GO-3.py`, and the XPolicyLab `policy_name` is `GO_3_Nano_preview`. The model's internal `policy_core.*.WAM_policy` class names, subdirectories, and configuration variables retain their original meanings.

See [RENAME_VALIDATION.md](RENAME_VALIDATION.md) for runtime validation after the rename, the locations of changes, and the latest logs.

Model inference and supervised fine-tuning (SFT) code for RoboTwin, including a standalone WebSocket inference server, an XPolicyLab adapter, and training entry points for multiple GPUs on a single machine and multiple nodes managed by a platform. When submitting to XPolicyLab, only the complete `policy/GO_3_Nano_preview/` directory needs to be included; weights, training data, and machine-specific path configuration are prepared separately.

The commands below assume they are run from the **XPolicyLab root directory**. A standalone copy of the code uses the same `policy/GO_3_Nano_preview/` layout and supports installing dependencies, launching the standalone inference server, and training directly; simulation evaluation requires the complete XPolicyLab/RoboTwin environment.

```text
policy/GO_3_Nano_preview/
├── README.md                     # Installation, inference, and fine-tuning instructions
├── model.py / deploy.py          # XPolicyLab model and evaluation interfaces
├── deploy.yml                   # XPolicyLab policy configuration
├── install.sh / env.sh          # Dependency installation and external path loading
├── eval.sh                      # Standard XPolicyLab evaluation entry point
├── setup_eval_policy_server.sh  # Official policy server launcher
├── setup_eval_env_client.sh     # Simulation or debug client launcher
├── train.sh / process_data.sh   # Standard XPolicyLab positional-argument entry points
├── train_multi.sh               # Multi-GPU training entry point with automatic GPU detection
├── start_model_server.sh        # Standalone inference server management
├── run_eval.sh                  # External RoboTwin evaluation using a remote server
├── runtime_paths.example.sh     # Template for path configuration outside the repository
├── configs/                     # Inference, training, and checkpoint architecture configurations
├── assets/stats/                # Robotwin normalization statistics
├── go3_nano_preview/            # Python package, implementation and dependency lists
├── tests/                       # Interface and training regression tests
└── tools/privacy_scan.py        # Privacy scan of the policy directory
```

## Environment Setup

### Install Dependencies

The runtime environment consists of Linux, Python 3.12, an NVIDIA GPU, and a driver compatible with the installed PyTorch version. Training dependencies include inference dependencies; choose `infer` when deploying only the inference server.

```bash
python3.12 -m venv "$HOME/.venvs/WAM_policy"
export PYTHON_BIN="$HOME/.venvs/WAM_policy/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip

# Install the inference environment
bash policy/GO_3_Nano_preview/install.sh infer

# If training is required, the installation command below is sufficient
# bash policy/GO_3_Nano_preview/install.sh train

```


The training dependencies include the inference dependencies and `decord`, which is required for CPU video reading. The two dependency lists are located at `go3_nano_preview/requirements-infer.txt` and `go3_nano_preview/requirements-train.txt`.

### Set Up External Path Configuration

Write the environment, weight, data, and output paths to a shell file outside the repository. Both inference and training select this file through `ROBOTWIN_RUNTIME_CONFIG` to avoid loading different configurations through different entry points.

```bash
mkdir -p "$HOME/.config/robotwin"
# Copy for initial setup; edit an existing configuration directly, preserving its contents
cp -n policy/GO_3_Nano_preview/runtime_paths.example.sh \
  "$HOME/.config/robotwin/runtime_paths.sh"

export ROBOTWIN_RUNTIME_CONFIG="$HOME/.config/robotwin/runtime_paths.sh"
```

Edit this file to set the variables required below. Replace every `/path/to/...` in the examples with an actual path on the current server; `POLICY_ENV` must point to the environment directory, not the path to `bin/python`.

```bash
# Add to the external runtime_paths.sh
export POLICY_ENV="$HOME/.venvs/WAM_policy"
export RUN_ROOT="$HOME/robotwin-runs"
export ROBOT_NAME="robotwin"
```

The scripts load the configuration automatically. Before referencing variables from the configuration on the command line, such as `"$RUN_ROOT"`, run:

```bash
source "$ROBOTWIN_RUNTIME_CONFIG"
```

Set `ROBOTWIN_RUNTIME_CONFIG` again whenever you open a new terminal, or set it in the task platform's launch command. Path configuration and model weights should not be added to Git.

## Model Inference

### Prepare Weights

The inference checkpoint directory must follow the structure below. The current launch script checks for `.bin` weight files:

```text
step_N/
├── config.json
├── diffusion_pytorch_model.bin
└── action_model/
    ├── config.json
    └── diffusion_pytorch_model.bin
```

Add the following to the external `runtime_paths.sh`:

```bash
export CKPT_PATH="/path/to/step_N"
export VIDEO_PRETRAIN_PATH="$CKPT_PATH/diffusion_pytorch_model.bin"
export ACTION_PRETRAIN_PATH="$CKPT_PATH/action_model/diffusion_pytorch_model.bin"
export VLM_MODEL_PATH="/path/to/vlm"
export VAE_PATH="/path/to/vae"
export STATS_TRACKER_PATH="/path/to/vae/latent_stats.pth"

export GPU_ID=0
export HOST="0.0.0.0"
export PORT=18081
```

| Variable | Description |
| --- | --- |
| `CKPT_PATH` | Checkpoint directory to deploy, containing the video and action models |
| `VIDEO_PRETRAIN_PATH`, `ACTION_PRETRAIN_PATH` | Initialization weights used when building the model; these can point to the corresponding files in the same checkpoint |
| `VLM_MODEL_PATH` | Directory of the corresponding Qwen 3.5 2B model |
| `VAE_PATH` | Directory of the corresponding VAE configuration and weights |
| `STATS_TRACKER_PATH` | VAE latent statistics file |
| `GPU_ID`, `HOST`, `PORT` | Defaults are `0`, `0.0.0.0`, and `18081`, respectively |
| `RUN_ROOT` | Root directory for logs and run outputs |

Training, validation, and inference all read [assets/stats/robotwin.json](assets/stats/robotwin.json) in this directory, so there is no need to set the state/action statistics file location in the external configuration. `STATS_TRACKER_PATH` is the VAE latent statistics file, serves a different purpose, and still needs to be configured separately.

Models are loaded offline at runtime, so all assets must be prepared in advance. The installation script does not download model weights. The model configuration and weight architecture must match; the pre-launch checks verify some key architecture parameters.

### Start, Inspect, and Stop the Server

```bash
# Check the configuration and asset files
bash policy/GO_3_Nano_preview/start_model_server.sh check

# Start in the background and wait for the health check to pass
bash policy/GO_3_Nano_preview/start_model_server.sh start

# Check the process and server readiness
bash policy/GO_3_Nano_preview/start_model_server.sh status

# Stop the server managed by this launch script
bash policy/GO_3_Nano_preview/start_model_server.sh stop
```

For foreground debugging, use `bash policy/GO_3_Nano_preview/start_model_server.sh foreground`. With the default configuration, the local health check URL is `http://127.0.0.1:18081/healthz`, and clients connect through `ws://<server-address>:18081`. `0.0.0.0` is the listening address; remote clients should use the actual server address.

See [deploy.yml](configs/deploy.yml) for the inference configuration.


### Evaluate Through XPolicyLab

The official adapter uses `model.py`, `deploy.py`, and the root-level [deploy.yml](deploy.yml) in this directory. This configuration serves a different purpose from the standalone server's `configs/deploy.yml`, and both must be retained.

Set `XPL_ROOT` (the XPolicyLab root directory), `EVAL_ENV` (the simulation environment directory), and `EVAL_ENV_CFG_TYPE` (the external simulator configuration filename) in the external path configuration, then run:

```bash
source "$ROBOTWIN_RUNTIME_CONFIG"
export EVAL_ENV_TYPE=sim
bash policy/GO_3_Nano_preview/eval.sh \
  robotwin adjust_bottle "$CKPT_PATH" "$EVAL_ENV_CFG_TYPE" joint \
  42 0 1 "$POLICY_ENV" "$EVAL_ENV"
```

The example specifies, in order, the benchmark, task, checkpoint, simulation configuration, action type, random seed, policy GPU, environment GPU, policy Python environment, and simulation Python environment. Adjust the GPU IDs to match the available resources. `eval.sh` starts the official policy server and client; there is no need to start the standalone `start_model_server.sh` server beforehand.

## Model Fine-Tuning

### Prepare Data and Training Paths

Training uses processed Robotwin data, with the following file layout for each episode:

```text
<SFT_DATA_ROOT>/<clip>/
├── head.mp4
├── left_hand.mp4
├── right_hand.mp4
└── actions.pt
```

`actions.pt` contains action data of shape `[num_frames,14]`, ordered as the 6 left-arm joints, left gripper, 6 right-arm joints, and right gripper. Each JSONL line corresponds to one episode and includes fields such as `clip`, `task_caption`, `video_frames`, `action_frames`, and `step_caption`; `clip` is relative to the data root directory. Step captions contain `start_frame`, `end_frame`, and `action_text`, with the end frame serving as an exclusive upper bound. Preserve the other caption and sampling fields in the original data.

Add the following paths to the external `runtime_paths.sh`, retaining `POLICY_ENV` and `RUN_ROOT` from the Environment Setup section, as well as `VLM_MODEL_PATH`, `VAE_PATH`, and `STATS_TRACKER_PATH` from the inference section:

```bash
export SFT_VIDEO_PRETRAIN_PATH="/path/to/initial_checkpoint/diffusion_pytorch_model.bin"
export SFT_ACTION_PRETRAIN_PATH="/path/to/initial_checkpoint/action_model/diffusion_pytorch_model.bin"

export SFT_JSONL_PATH="/path/to/train.jsonl"
export SFT_DATA_ROOT="/path/to/train_episodes"
export TASK_RECAP_PATH="/path/to/task_recap.json"
export STEP_RECAP_PATH="/path/to/step_recap.json"

export SFT_VAL_JSONL_PATH="/path/to/validation.jsonl"
export SFT_VAL_DATA_ROOT="/path/to/validation_episodes"
```

`SFT_*_PRETRAIN_PATH` is used for training initialization and is configured separately from the inference `CKPT_PATH`. `TASK_RECAP_PATH` and `STEP_RECAP_PATH` are mapping files for rewritten task and step text. The training and validation JSONL files must match their respective data root directories; even if a short test skips validation, the validation path variables must be set because the configuration module reads them at import time.


### Launch Training on a Single Machine or Multiple Platform-Managed Nodes

The current multi-GPU entry point is `policy/GO_3_Nano_preview/train_multi.sh`, using the following KA step configuration:

```bash
bash policy/GO_3_Nano_preview/train_multi.sh --env WAM_policy \
  policy/GO_3_Nano_preview/configs/robotwin_sft_ka_step.py \
  --mode train \
  --run_name cotrain_robotwin_sft_ka_robotype \
  --overwrite True --debug false --seed 1016
```

The default configuration is already located in the policy directory, so you can also launch directly or only check the topology:

```bash
bash policy/GO_3_Nano_preview/train_multi.sh
# Only resolve the configuration path and GPU/node topology, without loading the model or starting training
bash policy/GO_3_Nano_preview/train_multi.sh --dry-run
```

On a single machine, the script automatically detects the number of visible GPUs, so the GPU count does not need to be specified in the command. Set `CUDA_VISIBLE_DEVICES` to use only specific GPUs. Append `--dry-run` to the full `train_multi.sh` command to view only the resolved topology and launch command.

For multi-node training, the task platform dispatches the same command to every node in one operation and must provide the following variables. All nodes should have access to the same code, dependencies, data, and shared output directory.

| Variable | Meaning in the multi-GPU launch entry point |
| --- | --- |
| `WORLD_SIZE` | Number of nodes; for example, set to 8 for 8 nodes |
| `RANK` | Current node index, injected by the platform |
| `MASTER_ADDR` | Training network address of the master node |
| `MASTER_PORT` | Rendezvous port; defaults to 12345 |
| `NPROC_PER_NODE` | Optional number of GPU processes per node; by default, visible GPUs are detected automatically |

For example, with 8 nodes and 8 GPUs per node, totaling 64 GPUs, `WORLD_SIZE` at the entry point is **8**. Only after torchrun launches the workers does `WORLD_SIZE` inside each worker represent the total process count of 64. The script relies on platform dispatch and does not automatically launch other nodes over SSH from a regular single-machine terminal. All nodes must use consistent code paths, external configuration, and shared output paths.


When checking the training pipeline for the first time, you can explicitly reduce the batch size and run one iteration:

```bash
bash policy/GO_3_Nano_preview/train_multi.sh --env WAM_policy \
  policy/GO_3_Nano_preview/configs/robotwin_sft_ka_step.py \
  --mode train --run_name robotwin_smoke --seed 1016 \
  --steps 1 --batch-size 1 --workers 0 --skip-validation --skip-save
```

These command-line arguments override settings only for the current run and do not modify the configuration file. Add `--synchronize-timing` when CUDA synchronization is needed for timing; add `--gradient-checkpointing` to further reduce GPU memory used by activations. The test above uses all visible GPUs; to test a single GPU, set `CUDA_VISIBLE_DEVICES=0` before the command.

### Outputs and Resuming Training

Outputs are written to `$RUN_ROOT/checkpoints/<run_name>` by default; use `SFT_OUTPUT_DIR` to specify another directory. The main outputs are:

| File or Directory | Contents |
| --- | --- |
| `run_settings.json` | Effective batch size, process count, step count, test override parameters, and other settings for this run |
| `train_rank<N>.jsonl` | Per-rank records of loss, gradients, learning rate, GPU memory usage, and timing |
| `validation.jsonl` | Fixed validation results when validation is enabled |
| `step_N/` | Video and action model weights that can be loaded for inference |
| `training_state/` | DeepSpeed model, optimizer, and scheduler states |

`--overwrite True` renames the existing output directory as a backup and starts over; it does not resume previous training. To resume, use a new output name and keep the same number of GPU processes as when the checkpoint was saved:

```bash
bash policy/GO_3_Nano_preview/train_multi.sh --env WAM_policy \
  policy/GO_3_Nano_preview/configs/robotwin_sft_ka_step.py \
  --mode train --run_name robotwin_resumed --seed 1016 \
  --resume "/path/to/previous_run/training_state" --steps 10000
```

The final step count must be greater than the number of steps already completed in the checkpoint. Resuming restores the model, optimizer, and scheduler, but restarts the random streams for data sampling and augmentation. Setting only the initialization weight paths creates a new optimizer and restarts warmup. After training, point `CKPT_PATH` in the inference configuration to the generated `step_N/` directory.

Existing validation includes regression tests, real-data reading, short training runs on two H100 GPUs with a batch size of 1 per GPU, and inference checks with existing weights. These checks do not establish validation of a fresh environment installation, a full training run on 64 GPUs, or complete RoboTwin success-rate evaluation results.

### Directory and Interface Checks

In an environment with the training dependencies installed, run the following from the XPolicyLab root directory:

```bash
source policy/GO_3_Nano_preview/env.sh
export PYTHONPATH="$(pwd)/..:$PYTHONPATH"
"$POLICY_PYTHON" -B -m unittest discover -s policy/GO_3_Nano_preview/tests
"$POLICY_PYTHON" policy/GO_3_Nano_preview/tools/privacy_scan.py
```

The current 25 tests cover the model adapter, action format, captions, CPU decoding, training objectives, distributed sampling, and launch parsing after relocating the policy directory. The directory relocation tests copy only `policy/GO_3_Nano_preview/` and check the default configuration, command-line overrides, and a simulated 8×8 node topology from other working directories without starting actual training. By default, the privacy tool scans only the policy directory.

See [TESTING.md](TESTING.md) for the regression tests from 2026-10-24, plain/encoded-image Debug logs with real weights, environment versions, weight hashes, and reproduction commands. This record covers interface and regression validation and does not include RoboTwin simulation success-rate results.
