"""Run with python -m unittest discover -s tests -v."""
import copy
import unittest
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKL

from go3_nano_preview.policy_core.utils.WAM_policy.cpu_video_decoder import CpuVideoDecoder
from go3_nano_preview.policy_core.utils.WAM_policy.vae_normal_utils import LatentStatsTracker


class Tiny3DVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.rand(()))
        self.config = SimpleNamespace(scaling_factor=1.0, timestep_conditioning=True)
        self.latents_mean = torch.zeros(2)
        self.latents_std = torch.ones(2)
        self.vae_encoding_mode = '3dvae'

    @classmethod
    def from_config(cls, config):
        return cls()

    def decode(self, latents, temb, return_dict=False):
        assert latents.dtype == torch.float32 and latents.device.type == 'cpu'
        assert temb.dtype == torch.float32 and temb.device.type == 'cpu'
        return (latents * self.scale,)


class CpuVideoDecoderTests(unittest.TestCase):
    def test_real_2d_vae_matches_direct_decode_and_preserves_source(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(source_device=device):
                source = AutoencoderKL(
                    in_channels=3, out_channels=3,
                    down_block_types=('DownEncoderBlock2D',),
                    up_block_types=('UpDecoderBlock2D',),
                    block_out_channels=(8,), layers_per_block=1,
                    latent_channels=2, norm_num_groups=4, sample_size=8,
                ).to(device).eval()
                source.vae_encoding_mode = '2dvae'
                stats = LatentStatsTracker(num_channels=2).to(device)
                stats.running_mean.fill_(0.1)
                stats.running_std.fill_(0.7)
                before = {k: v.detach().clone() for k, v in source.state_dict().items()}
                latents = torch.randn(1, 2, 2, 4, 4, device=device)
                cpu_reference = copy.deepcopy(source).cpu()
                flat = latents.cpu().permute(0, 2, 1, 3, 4).reshape(2, 2, 4, 4)
                with torch.no_grad():
                    expected = cpu_reference.decode(flat * 0.7 + 0.1, return_dict=False)[0]
                expected = expected.reshape(1, 2, 3, 4, 4).permute(0, 2, 1, 3, 4).clamp(-1, 1)
                rng = torch.random.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
                decoder = CpuVideoDecoder(source, stats)
                actual = decoder.decode(latents)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(actual.device.type, 'cpu')
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                if cuda_rng is not None:
                    self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
                self.assertEqual(next(source.parameters()).device.type, device)
                for k, v in source.state_dict().items():
                    torch.testing.assert_close(v, before[k], rtol=0, atol=0)
                self.assertEqual(stats.running_mean.device.type, device)

    def test_3d_decode_uses_float32_and_preserves_rng(self):
        source = Tiny3DVAE().to(dtype=torch.bfloat16)
        latents = torch.randn(1, 2, 2, 3, 3).to(torch.bfloat16)
        rng = torch.random.get_rng_state().clone()
        decoder = CpuVideoDecoder(source)
        result = decoder.decode(latents)
        torch.testing.assert_close(result, (latents.float() * source.scale.float()).clamp(-1, 1))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(source.scale.dtype, torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
