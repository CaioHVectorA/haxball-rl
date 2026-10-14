import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import numpy as np
import torch
from haxballgym import TimeoutCondition
from ppo_returns import gae_returns
from gradient_metrics import objective_gradient_metrics
from model import Policy
from telemetry import RewardMeter
from train import create_eval_env
from environment import STADIUM


class ReturnChecks(unittest.TestCase):
    def test_mixed_boundaries_and_no_cross_episode_trace(self):
        # Columns: ordinary, goal, timeout, simultaneous goal + timeout.
        rewards = torch.tensor([[1., 2., 3., 4.], [100., 100., 100., 100.]])
        values = torch.tensor([[.5]*4, [10.]*4])
        term = torch.tensor([[False, True, False, True], [False]*4])
        done = torch.tensor([[False, True, True, True], [False]*4])
        final = torch.tensor([[0., 999., 7., 999.], [0.]*4])
        adv, ret = gae_returns(rewards, values, term, done, final, torch.ones(4)*20, .9, .8)
        torch.testing.assert_close(adv[1], torch.ones(4)*108)
        torch.testing.assert_close(adv[0], torch.tensor([1+9-.5+.9*.8*108, 1.5, 8.8, 3.5]))
        torch.testing.assert_close(ret, adv+values)
        legacy, _ = gae_returns(rewards, values, term, done, final, torch.ones(4)*20, .9, .8, False)
        self.assertAlmostEqual(float(legacy[0, 2]), 2.5)

    def test_timeout_on_last_rollout_step_uses_final_not_reset(self):
        for enabled, expected in [(True, 5.3), (False, -.1)]:
            adv, _ = gae_returns(torch.tensor([[.3]]), torch.tensor([[.4]]),
                torch.tensor([[False]]), torch.tensor([[True]]), torch.tensor([[6.]]),
                torch.tensor([999.]), .9, .95, enabled)
            self.assertAlmostEqual(float(adv[0, 0]), expected, places=5)

    def test_randomized_gae_matches_explicit_forward_sum(self):
        rng = torch.Generator().manual_seed(123)
        time, batch = 19, 7
        rewards = torch.randn(time, batch, generator=rng, dtype=torch.float64)
        values = torch.randn(time, batch, generator=rng, dtype=torch.float64)
        term = torch.rand(time, batch, generator=rng) < .13
        done = term | (torch.rand(time, batch, generator=rng) < .2)
        final = torch.randn(time, batch, generator=rng, dtype=torch.float64)
        last = torch.randn(batch, generator=rng, dtype=torch.float64)
        for enabled in (False, True):
            actual, _ = gae_returns(rewards, values, term, done, final, last, .9923, .95, enabled)
            expected = torch.zeros_like(actual)
            for start in range(time):
                for b in range(batch):
                    factor = 1.
                    for t in range(start, time):
                        successor = last[b] if t == time-1 else values[t+1, b]
                        if done[t, b]:
                            successor = final[t, b]
                        if term[t, b] or (not enabled and done[t, b]):
                            successor = 0.
                        expected[start, b] += factor*(rewards[t, b]+.9923*successor-values[t, b])
                        if done[t, b]:
                            break
                        factor *= .9923*.95
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_actual_env_captures_before_autoreset(self):
        env = create_eval_env(2, 1, 1, STADIUM, 8)
        env.truncation_cond = TimeoutCondition(8)
        meter = RewardMeter(env.reward_fn)
        env.reward_fn = meter
        env.reset()
        actions = np.tile([2, 1, 0], (2, 2, 1))
        reset_obs, _, term, trunc = env.step(actions)
        self.assertTrue(trunc.all())
        self.assertFalse(term.any())
        final_obs = env.obs_builder.build_obs(meter.last_state)
        self.assertGreater(float(np.abs(final_obs-reset_obs).max()), 0)
        saved = meter.last_state.player_pos.copy()
        env.engine.step(np.zeros((2, 2, 3), dtype=np.int32))
        np.testing.assert_array_equal(meter.last_state.player_pos, saved)

    def test_gradient_diagnostics_do_not_change_update_gradients(self):
        torch.manual_seed(7)
        model = Policy(16, hidden=8)
        obs = torch.randn(6, 16)
        actions = torch.zeros(6, 3, dtype=torch.long)
        logp, ent, val = model.evaluate(obs, actions)
        actor = -logp.mean()-.02*ent.mean()
        critic = .25*(val-2).square().mean()
        metrics = objective_gradient_metrics(model, actor, critic)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        expected = torch.autograd.grad(actor+critic, tuple(model.parameters()), retain_graph=True)
        (actor+critic).backward()
        for p, g in zip(model.parameters(), expected):
            torch.testing.assert_close(p.grad, g)
        self.assertGreater(metrics['grad_norm_actor_pre_clip'], 0)
        self.assertGreater(metrics['grad_norm_critic_pre_clip'], 0)
        self.assertLessEqual(abs(metrics['trunk_grad_cosine']), 1.000001)

    def test_zero_value_head_has_undefined_trunk_cosine(self):
        model = Policy(16, hidden=8)
        torch.nn.init.zeros_(model.value.weight)
        torch.nn.init.zeros_(model.value.bias)
        logp, ent, val = model.evaluate(torch.randn(4, 16), torch.zeros(4, 3, dtype=torch.long))
        metrics = objective_gradient_metrics(model, -logp.mean(), (val-1).square().mean())
        self.assertIsNone(metrics['trunk_grad_cosine'])
        self.assertGreater(metrics['grad_norm_critic_pre_clip'], 0)

if __name__ == '__main__':
    unittest.main()
