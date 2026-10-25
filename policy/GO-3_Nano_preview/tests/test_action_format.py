import unittest
import numpy as np
from policy_core.data.WAM_policy.transforms.WAM_policy_infer_transforms import WAM_policyCorobotFormat

class ActionFormatTests(unittest.TestCase):
    def test_robotwin_six_joint_profile(self):
        formatter = WAM_policyCorobotFormat('robotwin', 14, 52)
        result = formatter({'actions':np.arange(14)[None,:]})['result']
        self.assertEqual(result['left_arm']['values'], [[0,1,2,3,4,5]])
        self.assertEqual(result['left_effector'], [[6]])
        self.assertEqual(result['right_arm']['values'], [[7,8,9,10,11,12]])
        self.assertEqual(result['right_effector'], [[13]])

    def test_unsupported_type_and_dimensions_rejected(self):
        for robot, dimension, joints in [('unsupported',14,6), ('robotwin',16,7)]:
            with self.subTest(robot=robot, dimension=dimension), self.assertRaises(ValueError):
                WAM_policyCorobotFormat(robot,dimension,52,joint_dim=joints)({'actions':np.arange(dimension)[None,:]})

if __name__ == '__main__': unittest.main()
