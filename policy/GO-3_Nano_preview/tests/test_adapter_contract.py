"""Contract checks without loading multi-billion-parameter weights."""
import unittest
import numpy as np
from model import Model


class FakeInferrer:
    def infer(self, *, data):
        return {'result': {
            'left_arm': {'values': [[1]*6, [2]*6]},
            'left_effector': [[3], [4]],
            'right_arm': {'values': [[5]*6, [6]*6]},
            'right_effector': [[7], [8]],
        }}


class AdapterContract(unittest.TestCase):
    def setUp(self):
        self.model = Model.__new__(Model)
        self.model.keys = ('left_arm_joint_state','left_ee_joint_state',
                           'right_arm_joint_state','right_ee_joint_state')
        self.model.widths = (6,1,6,1)
        self.model.camera_keys = ('cam_head','cam_left_wrist','cam_right_wrist')
        self.model.config = {'image_size':[2,3]}
        self.model.horizon = 52
        self.model.steps = 5
        self.model.inferrer = FakeInferrer()
        self.model.reset()
        self.obs = dict(instruction='pick up the bottle',
            state={k:np.full(w,i+1) for i,(k,w) in enumerate(zip(self.model.keys,self.model.widths))},
            vision={k:{'color':np.full((2,3,3),[11+i,22+i,33+i],np.uint8)}
                    for i,k in enumerate(self.model.camera_keys)})

    def test_state_and_rgb_camera_order(self):
        self.model.update_obs(self.obs)
        np.testing.assert_array_equal(self.model.obs['state'],[1]*6+[2]+[3]*6+[4])
        np.testing.assert_array_equal(self.model.obs['obs'][:,0,0],[[11,22,33],[12,23,34],[13,24,35]])
        self.assertEqual(self.model.obs['prompt'],'pick up the bottle')

    def test_action_order_and_sparse_environment_ids(self):
        self.model.update_obs_batch([self.obs,self.obs])
        actions=self.model.get_action_batch([2,9])
        self.assertEqual(len(actions),2)
        self.assertEqual(len(actions[0]),2)
        for key,expected in zip(self.model.keys,[2,4,6,8]):
            np.testing.assert_array_equal(actions[0][1][key],np.full(self.model.widths[self.model.keys.index(key)],expected))
        self.model.reset()
        with self.assertRaises(RuntimeError): self.model.get_action()
        with self.assertRaises(RuntimeError): self.model.get_action_batch([2,9])

    def test_invalid_state_fails_before_inference(self):
        self.obs['state']['left_ee_joint_state']=[float('nan')]
        with self.assertRaises(ValueError):self.model.update_obs(self.obs)


if __name__ == '__main__':unittest.main()
