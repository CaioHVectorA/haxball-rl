import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import unittest
import numpy as np
from dataclasses import replace
from telemetry import Telemetry, RewardMeter
from train import create_eval_env
from environment import STADIUM
from bot import PositionBot
from live_evaluation import balanced_score


class TelemetryChecks(unittest.TestCase):
    def test_atomic_state_and_valid_append_records(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=Telemetry(directory,{'iters':2})
            writer.update(completed_iterations=1,phase='updating')
            writer.append('training.jsonl',{'iteration':1,'score':None})
            state=json.loads((Path(directory)/'live.json').read_text())
            self.assertEqual(state['completed_iterations'],1)
            self.assertIsNone(json.loads((Path(directory)/'training.jsonl').read_text())['score'])
            self.assertFalse((Path(directory)/'live.json.tmp').exists())

    def test_meter_preserves_reward_and_goal_count(self):
        env=create_eval_env(1,1,1,STADIUM,8);env.reset();state=env.prev_state
        goal=replace(state,scored=np.array([4]))
        original=env.reward_fn;meter=RewardMeter(original)
        np.testing.assert_allclose(meter.get_rewards(goal,state,np.array([True]),np.array([False])),original.get_rewards(goal,state,np.array([True]),np.array([False])))
        self.assertEqual(meter.gf,1);self.assertEqual(meter.ga,0)

    def test_no_completed_episode_is_unknown_not_zero(self):
        self.assertIsNone(balanced_score([{'score':None}]))
        self.assertEqual(balanced_score([{'score':0.5},{'score':1.}]),.75)

    def test_position_bot_kicks_only_when_goal_side_aligned(self):
        env=create_eval_env(1,1,1,STADIUM,8);env.reset()
        state=replace(env.prev_state,ball_pos=np.array([[100.,0.]]),player_pos=np.array([[[80.,0.],[80.,0.]]]))
        actions=PositionBot().act(state,[0,1])
        self.assertEqual(actions[0,0,2],1)
        self.assertEqual(actions[0,1,2],0)

if __name__=='__main__':unittest.main()
