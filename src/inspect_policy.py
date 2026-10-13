"""Inspect policy motion and action intent, without claiming kick/goal attribution."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from haxballgym import BLUE,RED
from haxballgym.mutator import RandomStateMutator
from bot import ChaseBot,PositionBot
from model import Policy
from train import create_eval_env
from environment import STADIUM


@torch.no_grad()
def inspect(checkpoint,steps=1000,n=16):
    model=Policy(16);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    rows=[];traces={}
    for scenario in ['kickoff','random']:
        for opponent in ['chase_no_noise','striker','keeper']:
            env=create_eval_env(n,1,1,STADIUM,8)
            if scenario=='random':env.state_mutator=RandomStateMutator(ball_speed=3.,rng=2701)
            obs=env.reset();bot=ChaseBot(jiggle=0,seed=2701) if opponent=='chase_no_noise' else PositionBot(defender=opponent=='keeper')
            original=env.reward_fn;captured=[]
            class Capture:
                def get_rewards(self,st,prev,term,trunc):
                    captured.append(st);return original.get_rewards(st,prev,term,trunc)
            env.reward_fn=Capture()
            gf=ga=contact=near_kicks=misaligned=far_kicks=stationary=own_half=covered=0
            distance_sum=0.;streak=np.zeros(n,dtype=int);longest=0;trace=[]
            for t in range(steps):
                pre=env.prev_state
                act=model.act(torch.as_tensor(obs[:,0],dtype=torch.float32),greedy=True).numpy()
                delta=pre.ball_pos-pre.player_pos[:,0];distance=np.linalg.norm(delta,axis=-1)
                toward=pre.goal_center(attacked=True)[:,0]-pre.ball_pos
                cosine=np.sum(delta*toward,axis=-1)/np.maximum(distance*np.linalg.norm(toward,axis=-1),1e-9)
                close=distance<24.8;pressed=act[:,2]==1
                contact+=int(close.sum());near_kicks+=int((close&pressed).sum())
                misaligned+=int((close&pressed&(cosine<0)).sum());far_kicks+=int((~close&pressed).sum())
                defensive=pre.ball_pos[:,0]<0;own_half+=int(defensive.sum())
                covered+=int((defensive&(pre.player_pos[:,0,0]<pre.ball_pos[:,0])).sum())
                distance_sum+=float(distance.sum())
                actions=np.stack([act,bot.act(pre,[1]).reshape(n,3)],axis=1)
                obs,_,term,trunc=env.step(actions);post=captured.pop()
                gf+=int((post.scored==BLUE).sum());ga+=int((post.scored==RED).sum())
                stopped=np.linalg.norm(post.ball_pos-pre.ball_pos,axis=-1)<.01
                stationary+=int(stopped.sum());streak=np.where(stopped,streak+1,0)
                longest=max(longest,int(streak.max()));streak[term|trunc]=0
                if t<300:trace.append(dict(decision=t,ball=post.ball_pos[0].round(3).tolist(),players=post.player_pos[0].round(3).tolist(),actions=actions[0].tolist(),scored=int(post.scored[0]),truncated=bool(trunc[0])))
            rows.append(dict(scenario=scenario,opponent=opponent,gf=gf,ga=ga,mean_distance=distance_sum/(n*steps),
                contact_fraction=contact/(n*steps),kick_contact_fraction=near_kicks/max(1,contact),
                misaligned_near_press_fraction=misaligned/max(1,near_kicks),far_press_fraction=far_kicks/(n*steps),
                defensive_goal_side_fraction=covered/max(1,own_half),stationary_fraction=stationary/(n*steps),
                longest_stationary_s=longest*8/60))
            traces[scenario+'_'+opponent]=trace
    return dict(checkpoint=str(checkpoint),envs=n,decisions=steps,seed=2701,side='red',rows=rows,traces=traces,
        limitation='Action intent metrics do not prove a successful kick or identify who last touched a goal. One side and one scenario seed only.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(2)
    result=inspect(args.checkpoint);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    lines=['# Inspeção de comportamento', '', result['limitation'], '',
      '| Cenário | Adversário | Pró × contra | Chute em proximidade | Pressão desalinhada | Bola parada |',
      '|---|---|---:|---:|---:|---:|']
    for row in result['rows']:
        lines.append(f'| {row["scenario"]} | {row["opponent"]} | {row["gf"]} × {row["ga"]} | {100*row["kick_contact_fraction"]:.1f}% | {100*row["misaligned_near_press_fraction"]:.1f}% | {100*row["stationary_fraction"]:.1f}% |')
    args.out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    if args.out.parent.name.startswith('approach_'):
        root=args.out.parent.parent
        summary=['# Comparação de recompensa — resultados disponíveis', '',
          'Execuções completas até esta atualização. Mesma política inicial e orçamento por trial; não extrapolar para 20 mil iterações.', '',
          '| Aproximação | Seed | Pontuação aleatória inicial | Pontuação aleatória final | Variação |',
          '|---|---:|---:|---:|---:|']
        for directory in sorted(root.glob('approach_*')):
            live_path=directory/'live.json'
            if not live_path.exists():continue
            live=json.loads(live_path.read_text())
            if live.get('status')!='completed':continue
            evaluations=[json.loads(line) for line in (directory/'evaluations.jsonl').read_text().splitlines()]
            scores=[]
            for evaluation in [evaluations[0],evaluations[-1]]:
                values=[case['score'] for case in evaluation['cases'] if case['scenario']=='random' and case['score'] is not None]
                scores.append(sum(values)/len(values)*100)
            summary.append(f'| {live["config"]["approach_weight"]} | {live["config"]["seed"]} | {scores[0]:.1f}% | {scores[1]:.1f}% | {scores[1]-scores[0]:+.1f} pp |')
        (root/'comparison.md').write_text('\n'.join(summary)+'\n')
    for row in result['rows']:print(json.dumps(row),flush=True)

if __name__=='__main__':main()
