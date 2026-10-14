import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import torch
from train import create_eval_env,build_reward
from environment import STADIUM,DiscountedPocketPotential,DiscountedBallGoalPotential
from model import Policy
from gradient_metrics import objective_gradient_metrics
from opponents import OpponentCurriculum
from training_checkpoint import atomic_torch_save,random_state,restore_random_state


class PBRSChecks(unittest.TestCase):
    def setUp(self):
        self.env=create_eval_env(2,1,1,STADIUM,8)
        self.env.reset();self.state=self.env.prev_state

    def test_discounted_telescoping_with_terminal_boundary(self):
        states=[replace(self.state,ball_pos=np.tile([x,20.],(2,1))) for x in [0,30,-70,100]]
        for reward in [DiscountedPocketPotential(21.12,.9),DiscountedBallGoalPotential(21.12,.9)]:
            total=np.zeros((2,2))
            for t in range(3):
                terminal=np.array([t==2,False])
                total+=.9**t*reward.get_rewards(states[t+1],states[t],terminal,np.array([False,t==2]))
            expected=-reward.potential(states[0])+.9**3*reward.potential(states[3])
            expected[0]=-reward.potential(states[0])[0]
            np.testing.assert_allclose(total,expected,atol=2e-6)

    def test_timeout_preserves_potential_but_goal_dominates_simultaneity(self):
        reward=DiscountedPocketPotential(21.12,.9)
        r=reward.get_rewards(self.state,self.state,np.array([False,True]),np.array([True,True]))
        phi=reward.potential(self.state)
        np.testing.assert_allclose(r[0],(.9-1)*phi[0],rtol=1e-6)
        np.testing.assert_allclose(r[1],-phi[1],rtol=1e-6)

    def test_ball_potential_is_antisymmetric_for_1v1(self):
        state=replace(self.state,ball_pos=np.array([[90.,50.],[-120.,-20.]]))
        phi=DiscountedBallGoalPotential(21.12,.99).potential(state)
        np.testing.assert_allclose(phi[:,0],-phi[:,1],atol=1e-12)

    def test_pocket_uses_reachable_kick_band_for_each_team(self):
        state=replace(self.state,ball_pos=np.zeros((2,2)),
           player_pos=np.tile([[[-22.,0.],[22.,0.]]],(2,1,1)))
        reward=DiscountedPocketPotential(21.12,.99)
        np.testing.assert_allclose(reward.potential(state),0.,atol=1e-12)
        wrong=replace(state,player_pos=-state.player_pos)
        self.assertTrue((reward.potential(wrong)<0).all())

    def test_table_uses_same_gamma_and_explicit_weights(self):
        reward=build_reward('pbrs',gamma=.91)
        self.assertEqual([w for f,w in reward.terms],[.15,.08,25.])
        self.assertEqual(reward.terms[0][0].gamma,.91)
        self.assertEqual(reward.terms[1][0].gamma,.91)


class ArchitectureChecks(unittest.TestCase):
    def test_transfer_preserves_actor_and_shared_value_when_requested(self):
        torch.manual_seed(2);old=Policy(16,hidden=8)
        new=Policy(16,hidden=8,separate_critic=True)
        new.load_compatible(old.state_dict(),keep_critic=True)
        obs=torch.randn(15,16)
        for a,b in zip(old.actor_logits(obs),new.actor_logits(obs)):torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(old.get_value(obs),new.get_value(obs),rtol=0,atol=0)
        self.assertFalse(set(map(id,new.actor_parameters())) & set(map(id,new.critic_parameters())))

    def test_only_critic_update_cannot_move_actor(self):
        model=Policy(16,hidden=8,separate_critic=True)
        before=model.actor_state_dict()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        loss=(model.get_value(torch.randn(12,16))-100).square().mean()
        loss.backward();optimizer.step()
        for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)

    def test_objective_metrics_support_disjoint_trunks(self):
        model=Policy(16,hidden=8,separate_critic=True)
        logp,ent,value=model.evaluate(torch.randn(12,16),torch.zeros(12,3,dtype=torch.long))
        metrics=objective_gradient_metrics(model,-logp.mean(),.25*(value-1).square().mean())
        self.assertIsNone(metrics['trunk_grad_cosine'])
        self.assertGreater(metrics['grad_norm_actor_pre_clip'],0)
        self.assertGreater(metrics['grad_norm_critic_pre_clip'],0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_invalid_checkpoint_rejected_without_partial_actor_change(self):
        source=Policy(16,hidden=8);target=Policy(16,hidden=8,separate_critic=True)
        before={k:v.clone() for k,v in target.state_dict().items()}
        broken=dict(source.state_dict());broken['value.bias']=torch.zeros(2)
        with self.assertRaises(ValueError):target.load_compatible(broken,keep_critic=True)
        for k,v in before.items():torch.testing.assert_close(target.state_dict()[k],v,rtol=0,atol=0)
        broken=dict(source.state_dict());del broken['head_k.bias']
        with self.assertRaises(ValueError):target.load_compatible(broken)

    def test_partial_separate_critic_must_not_silently_copy_actor(self):
        source=Policy(16,hidden=8,separate_critic=True);broken=dict(source.state_dict())
        del broken['critic_trunk.0.bias']
        with self.assertRaises(ValueError):source.load_compatible(broken,keep_critic=True)
        with self.assertRaises(RuntimeError):Policy.from_state_dict(broken)

    def test_factory_and_legacy_state_compatibility(self):
        for separate in [False,True]:
            model=Policy(16,hidden=8,separate_critic=separate)
            restored=Policy.from_state_dict(model.state_dict())
            self.assertEqual(restored.separate_critic,separate)
            self.assertEqual(set(restored.state_dict()),set(model.state_dict()))


class CheckpointAndCurriculumChecks(unittest.TestCase):
    def test_curriculum_changes_opponent_only_for_reset_episodes(self):
        curriculum=OpponentCurriculum(16,1.,[Policy(16,hidden=8)],12)
        before=curriculum.assignment.copy();mask=np.arange(16)%2==0
        curriculum.reset_episodes(mask)
        np.testing.assert_array_equal(before[~mask],curriculum.assignment[~mask])
        self.assertEqual(curriculum.weights[-1],0)
        env=create_eval_env(16,1,1,STADIUM,8);obs=env.reset()
        actions=curriculum.act(env.prev_state,torch.tensor(obs[:,1],dtype=torch.float32),Policy(16,hidden=8))
        self.assertEqual(actions.shape,(16,3))
        self.assertTrue(((actions>=0)&(actions<=np.array([2,2,1]))).all())

    def test_random_streams_are_restored(self):
        rng=np.random.default_rng(12);state=random_state(rng)
        draws=(torch.rand(5),np.random.rand(5),rng.random(5))
        restore_random_state(state,rng)
        torch.testing.assert_close(torch.rand(5),draws[0],rtol=0,atol=0)
        np.testing.assert_array_equal(np.random.rand(5),draws[1])
        np.testing.assert_array_equal(rng.random(5),draws[2])

    def test_failed_save_leaves_last_checkpoint_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resume.pt';atomic_torch_save({'iteration':4},path)
            def fail(value,handle):handle.write(b'partial');raise OSError('simulated failure')
            with patch('training_checkpoint.torch.save',fail):
                with self.assertRaises(OSError):atomic_torch_save({'iteration':5},path)
            self.assertEqual(torch.load(path,weights_only=True)['iteration'],4)
            atomic_torch_save({'iteration':6},path)
            self.assertEqual(torch.load(path,weights_only=True)['iteration'],6)

if __name__=='__main__':unittest.main()
