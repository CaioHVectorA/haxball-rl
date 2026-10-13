import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
from dataclasses import replace
import numpy as np
from environment import STADIUM, motion_scale, MotionPlayerToBall, MotionBallToGoal, RepeatKickAction
from train import create_eval_env


class EnvironmentChecks(unittest.TestCase):
    def setUp(self):
        self.env = create_eval_env(1,1,1,STADIUM,8)
        self.env.reset()
        self.state = self.env.engine.snapshot()

    def test_wall_keeps_ball_on_court(self):
        for y, vy in [(150.,10.),(-150.,-10.)]:
            self.env.engine.set_state(np.array([[0.,y]]),np.array([[0.,vy]]),
                np.array([[[-200.,0.],[200.,0.]]]),np.zeros((1,2,2)))
            positions=[]
            for _ in range(30):
                st=self.env.engine.step(np.zeros((1,2,3),dtype=np.int64))
                positions.append(st.ball_pos[0,1])
            self.assertLess(max(abs(y) for y in positions),201.)
            self.assertTrue(any(y*vy<0 for y in positions))

    def test_stationary_player_cannot_farm_reported_velocity(self):
        after=replace(self.state,player_vel=np.full((1,2,2),2.))
        rew=MotionPlayerToBall(motion_scale(STADIUM,8))
        np.testing.assert_array_equal(rew.get_rewards(after,self.state,None,None),np.zeros((1,2)))

    def test_ball_roundtrip_and_finite_goal(self):
        r=MotionBallToGoal(motion_scale(STADIUM,8))
        a=replace(self.state,ball_pos=np.array([[0.,0.]]))
        b=replace(a,ball_pos=np.array([[100.,0.]]))
        np.testing.assert_allclose(r.get_rewards(b,a,None,None)+r.get_rewards(a,b,None,None),0)
        far=replace(a,ball_pos=np.array([[100.,200.]]))
        self.assertGreater(r.distance(far)[0,0],r.distance(b)[0,0])

    def test_zeroing_critic_preserves_action_logits(self):
        import torch
        from model import Policy
        model = Policy(16)
        obs = torch.as_tensor(self.env.obs_builder.build_obs(self.state)[:,0])
        with torch.no_grad():
            before, _ = model._logits(obs)
            torch.nn.init.zeros_(model.value.weight)
            torch.nn.init.zeros_(model.value.bias)
            after, value = model._logits(obs)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(value, torch.zeros_like(value))

    def test_repeat_kick_keeps_two_policy_bins(self):
        parser=RepeatKickAction();a=np.array([[[2,1,1]]]);b=parser.parse_actions(a)
        self.assertEqual(parser.action_space(),(3,3,2))
        np.testing.assert_array_equal(b,np.array([[[1,0,2]]]))
        self.assertEqual(a[0,0,2],1)

    def test_approach_weight_changes_only_one_reward(self):
        from train import build_reward
        control=build_reward(approach_weight=.6)
        reduced=build_reward(approach_weight=.05)
        self.assertEqual([weight for _,weight in control.terms],[.6,.08,25.])
        self.assertEqual([weight for _,weight in reduced.terms],[.05,.08,25.])

    def test_motion_scale_matches_free_run(self):
        for _ in range(50):st=self.env.engine.step(np.array([[[1,0,0],[-1,0,0]]]))
        # Walls interfere with positions; velocities are still bounded by the free speed.
        self.assertAlmostEqual(motion_scale(STADIUM,8),21.12,places=6)
        self.assertLessEqual(np.linalg.norm(st.player_vel,axis=-1).max(),2.65)

if __name__=='__main__':unittest.main()
