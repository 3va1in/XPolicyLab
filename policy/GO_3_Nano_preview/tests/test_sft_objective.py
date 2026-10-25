"""Regression tests for rejection, member/view alignment and weighted gradients."""
import unittest
from types import SimpleNamespace
import torch

from go3_nano_preview.sft.loss import compute_action_loss
from go3_nano_preview.sft.helpers.two_step_cotrain import (
    TwoStepConfig, apply_filter, select_topk_members, confidence_weights,
    repeat_members, gather_member_noise,
)
from go3_nano_preview.sft.helpers.combined_cotrain import parse_combined_cotrain, build_micro_steps


class SFTObjectiveTests(unittest.TestCase):
    def test_noise_replay_preserves_member_and_view_order(self):
        raw = torch.arange(6).reshape(6,1)
        repeated = repeat_members(raw,4,3)
        selected = torch.tensor([2,5])
        replay = gather_member_noise(repeated,selected,8,3)
        torch.testing.assert_close(replay,raw)

    def test_collapsed_group_has_zero_confidence(self):
        cfg = TwoStepConfig(filter_type='separability',sep_threshold=2,sep_sigma_floor=.3,
                            weight_mode='gt_anchored')
        rewards = torch.full((2,4),-1.0)
        valid, _, _ = apply_filter(rewards,cfg)
        # Source keeps non-separable groups; the GT-anchored margin is zero.
        self.assertTrue(valid.all())
        weights = confidence_weights(rewards,torch.ones(2),valid,select_topk_members(rewards,1),cfg)
        self.assertEqual(torch.count_nonzero(weights).item(),0)

    def test_confidence_matches_gt_anchor_formula(self):
        rewards = torch.tensor([[-1.,-4.,-5.,-6.],[-2.,-3.,-4.,-7.]])
        energy = torch.tensor([2.,4.])
        cfg = TwoStepConfig(weight_mode='gt_anchored',weight_temp=3.,eref_floor=.1)
        selected = select_topk_members(rewards,1)
        actual = confidence_weights(rewards,energy,torch.ones_like(rewards,dtype=torch.bool),selected,cfg)
        expected = torch.tanh(torch.tensor([3.5,1.5])/(energy+1e-6)/3)
        torch.testing.assert_close(actual,expected)
        self.assertEqual(selected.tolist(),[0,4])

    def test_zero_weight_keeps_backward_graph(self):
        pred = torch.ones(2,4,32,requires_grad=True)
        mask = torch.zeros(2,1,32); mask[:,:,:14]=1
        out = dict(actions=torch.zeros_like(pred),pred_action=pred,
            target_action=torch.zeros_like(pred),action_sigmas=torch.tensor([.3,.8]),
            action_padding_mask=mask,sparse_length=1,sample_weights=torch.zeros(2),paths=None)
        loss = compute_action_loss(out)['loss_action']
        self.assertEqual(loss.item(),0)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertEqual(torch.count_nonzero(pred.grad).item(),0)

    def test_weighted_loss_normalizes_by_surviving_elements(self):
        pred = torch.ones(2,4,32,requires_grad=True)
        mask = torch.zeros(2,1,32); mask[0,:,:14]=1; mask[1,:,:7]=1
        out = dict(actions=torch.zeros_like(pred),pred_action=pred,
            target_action=torch.zeros_like(pred),action_sigmas=torch.tensor([.3,.8]),
            action_padding_mask=mask,sparse_length=1,sample_weights=torch.tensor([1.,0.]),paths=None)
        loss = compute_action_loss(out)['loss_action']
        self.assertAlmostEqual(loss.item(),(3+.001)/4,places=6)
        loss.backward()
        self.assertEqual(torch.count_nonzero(pred.grad[1]).item(),0)
        self.assertGreater(pred.grad[0].norm().item(),0)

    def test_reference_accumulation_window(self):
        cfg = SimpleNamespace(cotrain=SimpleNamespace(combined_cotrain=dict(enabled=True,
            order=['vgen','ae','cotrain'],weights=dict(vgen=1,ae=.1,cotrain=1),
            share_batch=True,share_features=True,warmup_steps=0)))
        combined = parse_combined_cotrain(cfg)
        self.assertEqual(combined.grad_accum,3)
        self.assertEqual([(m.name,m.weight) for m in build_micro_steps(0,combined)],
                         [('vgen',1),('ae',.1),('cotrain',1)])


if __name__ == '__main__':
    unittest.main()
