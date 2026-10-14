"""Behavioural analysis of checkpoints vs ChaseBot (goals for/against, kick rate, touches, freezing)."""
import sys
import numpy as np
import torch
from bot import ChaseBot
from haxballgym import BLUE, RED, DefaultObs, DiscreteAction, Env, GoalCondition, TimeoutCondition, TransitionEngine
from model import Policy
from train import KickoffSetter
from haxballgym import CombinedReward, GoalReward


def analyze(path, n=128, steps=600):
    eng = TransitionEngine(n, 1, 1, step_limit=2000, tick_skip=8, stadium="futsal-classic")
    env = Env(engine=eng, obs_builder=DefaultObs(pos_coef="auto"), action_parser=DiscreteAction(kick_values=2),
              reward_fn=CombinedReward((GoalReward(), 1.0)), termination_cond=GoalCondition(),
              truncation_cond=TimeoutCondition(2000), state_mutator=KickoffSetter())
    m = Policy.from_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    chase = ChaseBot(seed=0)
    obs = env.reset()
    gf = ga = contacts = kicks = far = idle = touch_steps = 0
    kick_steps = 0
    own_half = 0
    tot = 0
    with torch.no_grad():
        for _ in range(steps):
            s = env.prev_state
            d = np.linalg.norm(s.player_pos[:, 0] - s.ball_pos, axis=-1)
            sp = np.linalg.norm(s.player_vel[:, 0], axis=-1)
            act = m.act(torch.as_tensor(obs[:, 0], dtype=torch.float32)).cpu().numpy()
            c = d < 34
            contacts += int(c.sum()); kicks += int((c & (act[:, 2] == 1)).sum())
            kick_steps += int((act[:, 2] == 1).sum())
            f = d > 250
            far += int(f.sum()); idle += int((f & (sp < 0.2)).sum())
            own_half += int((s.ball_pos[:, 0] < 0).sum()); tot += n
            full = np.zeros((n, 2, 3), dtype=np.int64)
            full[:, 0] = act
            full[:, 1] = chase.act(s, np.array([1]), n_red=1).reshape(n, 3)
            obs, _, _, _ = env.step(full)
            sc = env.prev_state.scored
            gf += int((sc == BLUE).sum()); ga += int((sc == RED).sum())
    return dict(gf=gf, ga=ga, kick_on_contact=round(100 * kicks / max(1, contacts), 1),
                contact_frac=round(100 * contacts / tot, 1), kick_press=round(100 * kick_steps / tot, 1),
                idle_far=round(100 * idle / max(1, far), 1), ball_own_half=round(100 * own_half / tot, 1))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, analyze(p), flush=True)
