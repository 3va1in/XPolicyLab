import argparse
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from go3_nano_preview.sft.launch import build_command
from go3_nano_preview.sft.sampler import DistributedWeightedSampler, balanced_partitions


class DistributedTrainingTests(unittest.TestCase):
    def test_reference_balanced_pool_example(self):
        pools, mass = balanced_partitions(torch.tensor([9., 8., 7., 6., 5., 4.]), 2)
        self.assertEqual(pools.tolist(), [[0, 3, 4], [1, 2, 5]])
        self.assertEqual(mass.tolist(), [20., 19.])

    def test_disjoint_pools_equal_lengths_and_repeatable_draws(self):
        weights = torch.arange(1, 102, dtype=torch.float64)
        samplers = [DistributedWeightedSampler(weights, 4, rank=r, world=4, seed=1016) for r in range(4)]
        self.assertEqual([len(s) for s in samplers], [24]*4)
        pools = [set(s.indices.tolist()) for s in samplers]
        self.assertEqual(len(set.union(*pools)), 101)
        for i in range(4):
            for j in range(i):
                self.assertFalse(pools[i] & pools[j])
            draws = list(itertools.islice(iter(samplers[i]), 100))
            self.assertTrue(set(draws) <= pools[i])
            self.assertEqual(draws, list(itertools.islice(iter(samplers[i]), 100)))

    def test_sampling_never_allocates_training_horizon(self):
        sampler = DistributedWeightedSampler(torch.ones(2500), 16, rank=0, world=64, seed=1, chunk_size=17)
        original = torch.multinomial
        requests = []
        def sample(weights, count, **kwargs):
            requests.append(count)
            return original(weights, count, **kwargs)
        with patch('torch.multinomial', side_effect=sample):
            list(itertools.islice(iter(sampler), 1000))
        self.assertTrue(requests)
        self.assertLessEqual(max(requests), 17)
        self.assertEqual(len(sampler), 32)

    def test_unbalanced_shards_match_single_global_draw_stream(self):
        weights = torch.arange(1, 65, dtype=torch.float64)
        single = DistributedWeightedSampler(weights, 4, rank=0, world=1, seed=5, balanced=False)
        global_indices = list(itertools.islice(single, 64))
        for rank in range(4):
            sampler = DistributedWeightedSampler(weights, 4, rank=rank, world=4, seed=5, balanced=False)
            expected = [v for i,v in enumerate(global_indices) if (i//4)%4 == rank]
            self.assertEqual(list(itertools.islice(sampler,16)), expected)

    def test_invalid_weight_or_too_small_dataset_fails(self):
        for weights in ([0.,0.], [1.,float('nan')], [-1.,2.]):
            with self.assertRaises(ValueError):
                DistributedWeightedSampler(weights, 1, rank=0, world=1, seed=1)
        with self.assertRaises(ValueError):
            DistributedWeightedSampler(torch.ones(63), 1, rank=0, world=64, seed=1)

    def test_scheduler_node_environment_becomes_64_process_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'config.py'; config.write_text('')
            args = argparse.Namespace(config=str(config),run_name='test',seed=1016,steps=None,
                batch_size=None,workers=None,save_every=None,resume=None,overwrite=True,
                data_only=False,skip_validation=False,skip_save=False,
                gradient_checkpointing=False,synchronize_timing=False)
            for rank in range(8):
                command, topology = build_command(args, {'WORLD_SIZE':'8','RANK':str(rank),
                    'NPROC_PER_NODE':'8','MASTER_ADDR':'training-leader','MASTER_PORT':'29500','RUN_ROOT':tmp})
                self.assertEqual(topology['world_size'],64)
                self.assertIn(f'--node-rank={rank}',command)
                self.assertIn('--nnodes=8',command)
                self.assertNotIn('--standalone',command)
                self.assertIn('--overwrite',command)
            with self.assertRaises(ValueError):
                build_command(args, {'LOCAL_RANK':'0','WORLD_SIZE':'64','NPROC_PER_NODE':'8'})


if __name__ == '__main__':
    unittest.main()
