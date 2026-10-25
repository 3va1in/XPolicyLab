import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from policy_core.utils.WAM_policy.body_embedding import (
    DEFAULT_BODY_EMBEDDING_MAPPING, DEFAULT_NUM_BODY_EMBEDDINGS,
    resolve_body_emb_id, infer_body_type_from_dataset_name, validate_body_embedding_mapping,
)
from policy_core.data.WAM_policy.unified_dataset.embodiment_cue import (
    EMBODIMENT_FACTORS, compose_embodiment_cue, condition_caption,
)
from go3_nano_preview.sft.data.unified_dataset.base_dataset import MixBase
from go3_nano_preview.sft.data.unified_dataset.action_utils import get_processed_actions
from policy_core.model.WAM_policy.task.WAM_policy_task_flow import WAM_policyTaskFlow

class RobotwinConditioningTests(unittest.TestCase):
    def test_only_robotwin_and_checkpoint_slot_preserved(self):
        self.assertEqual(DEFAULT_BODY_EMBEDDING_MAPPING, {'robotwin':11})
        self.assertEqual(DEFAULT_NUM_BODY_EMBEDDINGS, 14)
        self.assertEqual(set(EMBODIMENT_FACTORS), {'robotwin'})
        self.assertFalse(EMBODIMENT_FACTORS['robotwin'].is_sim)
        self.assertIsNone(EMBODIMENT_FACTORS['robotwin'].gripper)
        self.assertEqual(resolve_body_emb_id('robotwin'), 11)
        with self.assertRaises(ValueError): resolve_body_emb_id('unsupported')
        with self.assertRaises(ValueError): infer_body_type_from_dataset_name('unsupported')
        with self.assertRaises(ValueError): validate_body_embedding_mapping({'robotwin':0},14)

    def test_caption_prefix_is_fixed_and_idempotent(self):
        self.assertEqual(compose_embodiment_cue('robotwin'), 'Agilex robot.')
        for caption in ['Pick up the bottle.', '', '  Move left. ', 'Agilex robot. Move left.']:
            text = condition_caption(caption)
            self.assertTrue(text.startswith('Agilex robot.'))
            self.assertEqual(condition_caption(text), text)
        dummy = SimpleNamespace(body_type_list=['robotwin'], embodiment_cue_prob=0.0)
        self.assertEqual(MixBase._embodiment_cue(dummy,0), 'Agilex robot.')
        flow = SimpleNamespace(_deploy_body_type='robotwin')
        actual = WAM_policyTaskFlow._condition_deploy_captions(flow, ['Pick up.', ''], 2)
        self.assertEqual(actual, ['Agilex robot. Pick up.', 'Agilex robot.'])

    def test_processed_actions_preserve_order_and_padding(self):
        data = np.arange(56, dtype=np.float32).reshape(4,14)
        with patch('torch.load', return_value=data):
            action, state, joints = get_processed_actions('unused', 1, 1, [0,1,2,3,3], norm_action=False, filter_action=False)
        self.assertEqual(joints, 6)
        np.testing.assert_array_equal(state, data[:1])
        np.testing.assert_array_equal(action, np.concatenate([data[1:], data[-1:]]))

if __name__ == '__main__': unittest.main()
