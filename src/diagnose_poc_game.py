"""Reproduce the deterministic POC failure and inspect pre-autoreset trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from bot import ChaseBot
from model import Policy
from train import create_eval_env, create_training_env, build_reward
from haxballgym import RED, BLUE


def trace(checkpoint, *, steps=300, greedy=True, jitter=0., scenario='kickoff', force_kick=False, seed=10, rearm_blue=False):
    torch.manual_seed(seed)
    model=Policy(16)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    env=(create_training_env(1,1,1,'futsal-classic',8,seed=seed,repeat_kick=False) if scenario=='random' else create_eval_env(1,1,1,'futsal-classic',8,repeat_kick=False))
    obs=env.reset()
    if scenario=='offset':
        st=env.prev_state
        pp=st.player_pos.copy();pp[0,0,1]=10.
        env.prev_state=env.engine.set_state(st.ball_pos,st.ball_vel,pp,st.player_vel)
        obs=env.obs_builder.build_obs(env.prev_state)
    chase=ChaseBot(seed=seed,jiggle=jitter)
    captured=[]
    terms=build_reward('simple_progress1').terms
    class Capture:
        def get_rewards(self,st,prev,term,trunc):
            captured.append(st)
            return sum(fn.get_rewards(st,prev,term,trunc)*w for fn,w in terms)
    env.reward_fn=Capture()
    rows=[]; goals=[]; contributions=np.zeros(3)
    for i in range(steps):
        pre=env.prev_state
        with torch.no_grad():
            act=model.act(torch.as_tensor(obs[:,0],dtype=torch.float32),greedy=greedy).numpy()
        distance=float(np.linalg.norm(pre.ball_pos[0]-pre.player_pos[0,0]))
        if force_kick and distance<24.8:act[0,2]=1
        acts=np.stack([act,chase.act(pre,[1]).reshape(1,3)],axis=1)
        if rearm_blue and i >= 60:
            acts[0,1,2] = int(i % 2 == 0)
        obs,_,term,trunc=env.step(acts)
        post=captured.pop()
        parts=[float(fn.get_rewards(post,pre,term,trunc)[0,0]*w) for fn,w in terms]
        contributions+=parts
        row=dict(decision=i,time_s=round((i+1)*8/60,4),ball=post.ball_pos[0].tolist(),
          ball_vel=post.ball_vel[0].tolist(),players=post.player_pos[0].tolist(),
          player_vel=post.player_vel[0].tolist(),actions=acts[0].tolist(),pre_distance=distance,
          scored=int(post.scored[0]),rewards_progress1=parts,terminated=bool(term[0]),truncated=bool(trunc[0]))
        rows.append(row)
        if post.scored[0]!=-1:goals.append(dict(decision=i,conceding=int(post.scored[0]),ball=post.ball_pos[0].tolist()))
    return dict(checkpoint=str(checkpoint),greedy=greedy,jitter=jitter,scenario=scenario,force_kick=force_kick,
                rearm_blue=rearm_blue,gf=sum(g['conceding']==BLUE for g in goals),ga=sum(g['conceding']==RED for g in goals),
                goal_events=goals,reward_sums_progress1=contributions.tolist(),rows=rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--steps',type=int,default=300);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    root=Path('experiments/reward_progress008_seeded_poc_20261013')
    bad=root/'simple_progress1_seed2.pt';low=root/'simple_seed2.pt'
    cases={'failure_greedy':trace(bad,steps=args.steps),'low_greedy':trace(low,steps=args.steps),'failure_sampled':trace(bad,steps=args.steps,greedy=False),
           'failure_noise':trace(bad,steps=args.steps,jitter=.25),'failure_offset':trace(bad,steps=args.steps,scenario='offset'),
           'failure_random':trace(bad,steps=args.steps,scenario='random'),'low_sampled':trace(low,steps=args.steps,greedy=False),'low_noise':trace(low,steps=args.steps,jitter=.25),'low_random':trace(low,steps=args.steps,scenario='random'),'failure_force_kick':trace(bad,steps=args.steps,force_kick=True)}
    (args.out/'traces.json').write_text(json.dumps(cases,indent=2))
    for name,c in cases.items():print(name,c['gf'],c['ga'],'events',[g['decision'] for g in c['goal_events']],'reward sums',c['reward_sums_progress1'])
    for name in ['failure_greedy','low_greedy']:
        print('\n',name)
        for r in cases[name]['rows'][:40]:
            print(r['decision'],'a',r['actions'][0],'d',round(r['pre_distance'],1),'ball',np.round(r['ball'],1).tolist(),'players',np.round(r['players'],1).tolist(),'vx',round(r['ball_vel'][0],2))

if __name__=='__main__':main()
