"""Check the distributed launcher after copying only the policy directory."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Linux/Bash launcher')
class PortableLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='wam-layout-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.policy = cls.root / 'XPolicyLab workspace' / 'policy' / 'GO_3_Nano_preview'
        shutil.copytree(Path(__file__).resolve().parents[1], cls.policy,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        cls.caller = cls.root / 'caller'
        cls.caller.mkdir()
        environment = cls.root / 'python environment'
        (environment / 'bin').mkdir(parents=True)
        (environment / 'bin' / 'python').symlink_to(sys.executable)
        runtime = cls.root / 'runtime paths.sh'
        runtime.write_text(f'export POLICY_ENV="{environment}"\n')
        cls.env = dict(os.environ)
        for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'NPROC_PER_NODE',
                    'MASTER_ADDR', 'MASTER_PORT', 'PYTHONPATH', 'SFT_RUNTIME_CONFIG'):
            cls.env.pop(key, None)
        cls.env.update(ROBOTWIN_RUNTIME_CONFIG=str(runtime),
                       CUDA_VISIBLE_DEVICES='0,1', PYTHONDONTWRITEBYTECODE='1',
                       RUN_ROOT=str(cls.root / 'runs'))

    def launch(self, *arguments, extra_env=None):
        result = subprocess.run(
            ['bash', str(self.policy / 'train_multi.sh'), *arguments, '--dry-run'],
            cwd=self.caller, env={**self.env, **(extra_env or {})},
            text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_default_config_and_gpu_detection_without_root_files(self):
        self.assertFalse((self.policy.parents[1] / 'train.sh').exists())
        result = self.launch()
        self.assertEqual(result['world_size'], 2)
        command = result['command']
        self.assertEqual(command[command.index('--config') + 1],
                         str(self.policy / 'configs' / 'robotwin_sft_ka_step.py'))
        self.assertIn('go3_nano_preview.sft.train', command)

    def test_explicit_relative_config_and_cli_overrides(self):
        config = self.caller / 'custom.py'
        config.write_text('# dry-run only\n')
        command = self.launch('custom.py', '--run_name', 'custom-run',
                              '--overwrite', 'False', '--seed', '7')['command']
        self.assertEqual(command[command.index('--config') + 1], str(config))
        self.assertEqual(command[command.index('--seed') + 1], '7')
        self.assertNotIn('--overwrite', command)
        self.assertTrue(command[command.index('--output') + 1].endswith('custom-run'))

    def test_scheduler_topology_after_relocation(self):
        result = self.launch(extra_env={'WORLD_SIZE': '8', 'RANK': '3',
            'MASTER_ADDR': 'training-leader', 'NPROC_PER_NODE': '8'})
        self.assertEqual(result['world_size'], 64)
        self.assertIn('--nnodes=8', result['command'])
        self.assertIn('--node-rank=3', result['command'])


if __name__ == '__main__':
    unittest.main()
