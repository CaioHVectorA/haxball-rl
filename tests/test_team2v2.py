import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'headless-bot'))
from environment import STADIUM, TeamPocketPotential, DiscountedBallGoalPotential
from model import Policy
from train import create_eval_env, TeamRandomSetter, build_reward
from live_evaluation import panel
from telemetry import RewardMeter
import export_policy
from replay import dump_trace
from select_checkpoint import h2h


class TeamChecks(unittest.TestCase):
    def setUp(self):
        self.env = create_eval_env(2,2,2,STADIUM,8,table='pbrs')
        self.obs = self.env.reset()
        self.state = self.env.prev_state

    def test_spawn_slots_mirror_and_player_permutation(self):
        self.assertEqual(self.obs.shape, (2,4,24))
        self.assertTrue(np.isfinite(self.obs).all())
        pp = self.state.player_pos
        self.assertTrue((pp[:,:2,0] < 0).all())
        self.assertTrue((pp[:,2:,0] > 0).all())
        distances = np.linalg.norm(pp[:,:,None]-pp[:,None,:],axis=-1)
        self.assertTrue(((distances>32)|np.eye(4,dtype=bool)[None]).all())
        # Teammate slot 16, nearest opponent 12; positions are in policy frame.
        np.testing.assert_allclose(self.obs[0,0,16:18], [0,130*self.env.obs_builder.pos_coef[1]])
        order = [1,0,3,2]
        permuted = replace(self.state,player_pos=pp[:,order],player_vel=self.state.player_vel[:,order])
        np.testing.assert_allclose(self.env.obs_builder.build_obs(permuted),self.obs[:,order])
        mirror = np.array([-1.,1.])
        order = [2,3,0,1]
        mirrored = replace(self.state,player_pos=pp[:,order]*mirror,
            player_vel=self.state.player_vel[:,order]*mirror,ball_pos=self.state.ball_pos*mirror,
            ball_vel=self.state.ball_vel*mirror)
        np.testing.assert_allclose(self.env.obs_builder.build_obs(mirrored),self.obs[:,order],atol=1e-6)

    def test_random_resets_and_step_actions_cover_all_four_players(self):
        setter = TeamRandomSetter(ball_speed=3.,rng=42)
        bp,_,pp,_ = setter.sample(self.env.engine,100)
        distances = np.linalg.norm(pp[:,:,None]-pp[:,None,:],axis=-1)
        self.assertTrue(((distances>=32)|np.eye(4,dtype=bool)[None]).all())
        self.assertTrue((np.linalg.norm(pp-bp[:,None],axis=-1)>=24.8).all())
        actions=np.tile([2,1,1],(2,4,1))
        obs,reward,term,trunc=self.env.step(actions)
        self.assertEqual(obs.shape,(2,4,24))
        self.assertEqual(reward.shape,(2,4))
        np.testing.assert_allclose(reward[:,0],reward[:,1])
        np.testing.assert_allclose(reward[:,2],reward[:,3])
        self.assertTrue(np.isfinite(reward).all())

    def test_shared_reward_terminal_pbrs_and_team_meter(self):
        reward=TeamPocketPotential(21.12,.99)
        state=replace(self.state,player_pos=np.tile([[[-22.,0.],[-200.,100.],[22.,0.],[200.,100.]]],(2,1,1)))
        np.testing.assert_allclose(reward.potential(state),0)
        phi=reward.potential(self.state)
        actual=reward.get_rewards(state,self.state,np.array([True,False]),np.array([False,True]))
        np.testing.assert_allclose(actual,-phi,atol=1e-6)
        ball=DiscountedBallGoalPotential(21.12,.99).potential(self.state)
        np.testing.assert_allclose(ball[:,0],-ball[:,2])
        meter=RewardMeter(self.env.reward_fn,n_red=2)
        result=meter.get_rewards(state,self.state,np.zeros(2,dtype=bool),np.zeros(2,dtype=bool))
        self.assertAlmostEqual(sum(meter.sums.values()),float(result[:,:2].sum()),places=5)

    def test_transfer_preserves_old_logits_and_resets_critic(self):
        old=Policy(16,hidden=8,separate_critic=True)
        target=Policy(24,hidden=8,separate_critic=True)
        critic={k:v.clone() for k,v in target.state_dict().items() if not k.startswith(('trunk.','head_'))}
        target.transfer_1v1_to_2v2(old.state_dict())
        obs=torch.randn(12,24)
        for a,b in zip(old.actor_logits(obs[:,:16]),target.actor_logits(obs)):
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
        for key,value in critic.items():torch.testing.assert_close(target.state_dict()[key],value,rtol=0,atol=0)
        self.assertEqual(torch.count_nonzero(target.trunk[0].weight[:,16:]).item(),0)
        loss=sum(x.square().mean() for x in target.actor_logits(obs));loss.backward()
        self.assertGreater(target.trunk[0].weight.grad[:,16:].abs().sum().item(),0)

    def test_evaluation_controls_both_teams_and_counts_episodes_once(self):
        model=Policy(24,hidden=8)
        rows=panel(model,model,STADIUM,0,lambda row:None,lambda *args:None,
                   n=1,steps=3,eval_seeds=(42,),n_red=2,n_blue=2)
        self.assertEqual(len(rows),20)
        self.assertEqual({row['side'] for row in rows},{0,1})
        self.assertTrue(all(row['red']==2 and row['blue']==2 for row in rows))
        self.assertTrue(all(row['completed_episodes']==row['wins']+row['draws']+row['losses'] for row in rows))

    def test_javascript_observations_and_logits_match_python_both_sides(self):
        model=Policy(24)
        px,goal,hbs=export_policy.stadium_obs_params(STADIUM)
        policy=dict(obs_dim=24,obs_schema='team24_v1',team_size=2,pos_coef=px,goal_x=goal)
        for name,key in [('trunk0','trunk.0'),('trunk2','trunk.2'),('head_x','head_x'),('head_y','head_y'),('head_k','head_k')]:
            policy[name]={'w':model.state_dict()[key+'.weight'].tolist(),'b':model.state_dict()[key+'.bias'].tolist()}
        self.env.state_mutator=TeamRandomSetter(rng=73,ball_speed=3.)
        obs=self.env.reset();state=self.env.prev_state
        discs=lambda e,j:dict(id=j,pos=dict(zip(('x','y'),state.player_pos[e,j])),speed=dict(zip(('x','y'),state.player_vel[e,j])))
        cases=[]
        for e in range(2):
            for j in range(4):
                others=[k for k in range(4) if k!=j]
                cases.append(dict(self=discs(e,j),ball=dict(pos=dict(zip(('x','y'),state.ball_pos[e])),speed=dict(zip(('x','y'),state.ball_vel[e]))),
                    teammates=[discs(e,k) for k in others if state.team[e,k]==state.team[e,j]],
                    opponents=[discs(e,k) for k in others if state.team[e,k]!=state.team[e,j]],team=1 if j<2 else 2))
        script="const fs=require('fs'),r=require('./headless-bot/policy_runtime');const d=JSON.parse(fs.readFileSync(0,'utf8'));r.validatePolicy(d.policy);console.log(JSON.stringify(d.cases.map(c=>{const obs=r.buildObs(d.policy,c.self,c.ball,c.teammates,c.opponents,c.team);return {obs,logits:r.logits(d.policy,obs)};})));"
        result=subprocess.run(['node','-e',script],input=json.dumps(dict(policy=policy,cases=cases)),capture_output=True,text=True,cwd=ROOT,check=True)
        actual=json.loads(result.stdout)
        np.testing.assert_allclose([x['obs'] for x in actual],obs.reshape(8,24),atol=1e-6)
        expected=model.actor_logits(torch.tensor(obs.reshape(8,24)))
        for head in range(3):
            np.testing.assert_allclose([x['logits'][head] for x in actual],expected[head].detach().numpy(),atol=2e-6)


if __name__=='__main__':unittest.main()
