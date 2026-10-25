"""Reference CLI for a scheduler that runs one command instance per node.

WORLD_SIZE/RANK here mean node count/node rank at the train_multi.sh entry point.
torchrun replaces them with GPU process world size/global rank for the worker.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def boolean(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('Expected true or false')
    return value.lower() == 'true'


def gpu_count(env):
    if 'CUDA_VISIBLE_DEVICES' in env:
        devices = [d.strip() for d in env['CUDA_VISIBLE_DEVICES'].split(',') if d.strip()]
        return 0 if devices == ['-1'] else len(devices)
    result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True, check=True)
    return len(result.stdout.splitlines())


def build_command(args, env):
    if 'LOCAL_RANK' in env:
        raise ValueError('Run train_multi.sh once per node, not inside an existing torchrun GPU worker')
    nodes = int(env.get('WORLD_SIZE', '1'))
    node_rank = int(env.get('RANK', '0'))
    per_node = int(env['NPROC_PER_NODE']) if 'NPROC_PER_NODE' in env else gpu_count(env)
    if nodes < 1 or per_node < 1 or not 0 <= node_rank < nodes:
        raise ValueError('Invalid WORLD_SIZE/RANK/NPROC_PER_NODE node topology')
    if nodes > 1 and (not env.get('MASTER_ADDR') or 'RANK' not in env):
        raise ValueError('Multi-node jobs require scheduler-provided MASTER_ADDR and node RANK')
    master = env.get('MASTER_ADDR', '127.0.0.1')
    port = int(env.get('MASTER_PORT', '12345'))
    if not 1 <= port <= 65535:
        raise ValueError('MASTER_PORT must be between 1 and 65535')
    config = Path(args.config).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    name = args.run_name or config.stem
    if not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('run_name must be a simple directory name')
    run_root = Path(env.get('RUN_ROOT', str(Path.home() / '.local/state/robotwin')))
    output = Path(env.get('SFT_OUTPUT_DIR', str(run_root / 'checkpoints' / name))).resolve()
    command = [sys.executable, '-m', 'torch.distributed.run',
        f'--nnodes={nodes}', f'--nproc-per-node={per_node}', f'--node-rank={node_rank}',
        f'--master-addr={master}', f'--master-port={port}',
        '-m', 'go3_nano_preview.sft.train', '--config', str(config), '--output', str(output),
        '--seed', str(args.seed)]
    for attr, flag in [('steps','--steps'),('batch_size','--batch-size'),('workers','--workers'),
                       ('save_every','--save-every'),('resume','--resume')]:
        value = getattr(args, attr)
        if value is not None:
            command.extend([flag, str(value)])
    for attr in ('overwrite','data_only','skip_validation','skip_save','gradient_checkpointing','synchronize_timing'):
        if getattr(args, attr):
            command.append('--' + attr.replace('_','-'))
    return command, {'nodes':nodes,'node_rank':node_rank,'gpus_per_node':per_node,
                     'world_size':nodes*per_node,'output':str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?',
        default=str(Path(__file__).resolve().parents[2] / 'configs' / 'robotwin_sft_ka_step.py'),
        help='Training config; relative paths are resolved from the calling directory')
    parser.add_argument('--env', choices=['GO_3_Nano_preview', 'go3_nano_preview', 'WAM_policy'], required=True)
    parser.add_argument('--mode', choices=['train'], default='train')
    parser.add_argument('--run_name')
    parser.add_argument('--overwrite', type=boolean, default=False)
    parser.add_argument('--debug', type=boolean, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--resume')
    for option in ('data-only','skip-validation','skip-save','gradient-checkpointing','synchronize-timing','dry-run'):
        parser.add_argument('--' + option, action='store_true')
    args = parser.parse_args()
    command, topology = build_command(args, os.environ)
    print(json.dumps({'event':'distributed_launch', **topology, 'command':command}), flush=True)
    if args.dry_run:
        return
    if args.debug:
        os.environ.setdefault('TORCH_DISTRIBUTED_DEBUG', 'DETAIL')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
