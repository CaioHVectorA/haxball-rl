"""Fine-tune with corrected physics/rewards; compare on identical held-out matchups."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from haxballgym import RED, BLUE
from bot import ChaseBot, RandomBot
from environment import STADIUM
from model import Policy
from train import create_training_env, create_eval_env, build_reward
from poc_reward import train, seed_all


@torch.no_grad()
def evaluate_checkpoint(checkpoint, *, envs=16, steps=750, stadium=STADIUM, repeat_kick=True,
                        greedy_modes=(True,False), opponents=("chase","chase_no_noise","random"), eval_seeds=(100,101)):
    model=Policy.from_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    rows=[]
    for greedy in greedy_modes:
        for scenario in ['kickoff','random']:
            for opponent in opponents:
                for seed in eval_seeds:
                    for side in [0,1]:
                        seed_all(seed)
                        env=(create_eval_env(envs,1,1,stadium,8,repeat_kick=repeat_kick) if scenario=='kickoff' else create_training_env(envs,1,1,stadium,8,seed=seed,repeat_kick=repeat_kick))
                        if scenario=='random':
                            from haxballgym.mutator import RandomStateMutator
                            env.state_mutator=RandomStateMutator(ball_speed=3.,rng=seed)
                        obs=env.reset()
                        bot=RandomBot(seed) if opponent=='random' else ChaseBot(seed=seed,jiggle=0. if opponent=='chase_no_noise' else .25)
                        gf=ga=wins=losses=draws=stationary=0
                        accumulated=np.zeros(envs)
                        for _ in range(steps):
                            pre=env.prev_state
                            acts=np.empty((envs,2,3),dtype=np.int64)
                            acts[:,side]=model.act(torch.as_tensor(obs[:,side],dtype=torch.float32),greedy=greedy).numpy()
                            other=1-side
                            acts[:,other]=(bot.act(obs[:,other]) if opponent=='random' else bot.act(pre,[other])).reshape(envs,3)
                            obs,_,term,trunc=env.step(acts)
                            scored=env.prev_state.scored
                            good=scored==(BLUE if side==0 else RED)
                            bad=scored==(RED if side==0 else BLUE)
                            gf+=int(good.sum());ga+=int(bad.sum())
                            accumulated += good.astype(int)-bad.astype(int)
                            done=term|trunc
                            wins+=int(((accumulated>0)&done).sum())
                            losses+=int(((accumulated<0)&done).sum())
                            draws+=int(((accumulated==0)&done).sum())
                            accumulated[done]=0
                            # Skip reset transitions when measuring motion.
                            movement=np.linalg.norm(env.prev_state.ball_pos-pre.ball_pos,axis=-1)
                            stationary+=int(((movement<.01)&~done).sum())
                        rows.append(dict(greedy=greedy,scenario=scenario,opponent=opponent,seed=seed,side=side,
                          gf=gf,ga=ga,net=gf-ga,wins=wins,losses=losses,draws=draws,
                          stationary_fraction=stationary/(envs*steps)))
    return rows


def selection_score(rows):
    """Equal weight per held-out condition, using the deployment (greedy) mode."""
    scores=[]
    for row in rows:
        if row['opponent']=='random' or not row['greedy']:
            continue
        completed=row['wins']+row['draws']+row['losses']
        if completed:
            scores.append((row['wins']+.5*row['draws'])/completed)
    return float(np.mean(scores)) if scores else 0.


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--warm-start',type=Path,default=Path('experiments/reward_progress008_seeded_poc_20261013/initial.pt'))
    ap.add_argument('--keep-critic',action='store_true',help='Keep old value predictions despite changing reward scale')
    ap.add_argument('--baseline-report',type=Path)
    ap.add_argument('--iters',type=int,default=100)
    ap.add_argument('--envs',type=int,default=64)
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    args=ap.parse_args();torch.set_num_threads(2)
    if args.out.exists():raise SystemExit('Use a new output directory.')
    initial=torch.load(args.warm_start,map_location='cpu',weights_only=True)
    args.out.mkdir(parents=True)
    torch.save(initial,args.out/'initial.pt')
    args.reset_value=not args.keep_critic
    args.stadium=STADIUM;args.eval_seeds=[10,11];args.eval_envs=16;args.eval_steps=300
    report=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
       reward_weights={'goal':25,'approach':.6,'progress':.08},
       limitations='Training versus Chase; held-out seeds 100/101 used for selection, not independent final validation.',runs=[])
    if args.baseline_report:
        old = json.loads(args.baseline_report.read_text())
        old_initial = torch.load(args.baseline_report.parent/'initial.pt',weights_only=True)
        if not all(torch.equal(initial[k], old_initial[k]) for k in initial):
            raise ValueError('Baseline checkpoint mismatch')
        if old['config']['stadium'] != STADIUM:
            raise ValueError('Baseline stadium mismatch')
        report['before'] = old['before']
    else:
        report['before']=evaluate_checkpoint(args.out/'initial.pt')
    (args.out/'results.json').write_text(json.dumps(report,indent=2))
    for seed in args.seeds:
        run=train('corrected',seed,args,initial)
        run['evaluation']=evaluate_checkpoint(args.out/f'corrected_seed{seed}.pt')
        report['runs'].append(run)
        (args.out/'results.json').write_text(json.dumps(report,indent=2))
        print(f'Full evaluation completed for seed {seed}',flush=True)
    # Equal-weight episode outcomes avoid letting a repeated kickoff dominate selection.
    best=max(report['runs'],key=lambda r:selection_score(r['evaluation']))
    report['selected_seed']=best['seed']
    report['before_chase_net']=sum(r['net'] for r in report['before'] if r['opponent']!='random')
    report['selected_chase_net']=sum(r['net'] for r in best['evaluation'] if r['opponent']!='random')
    report['before_score']=selection_score(report['before'])
    report['selected_score']=selection_score(best['evaluation'])
    report['selection']='Mean episode score across greedy Chase conditions; each side, seed, scenario and opponent has equal weight.'
    (args.out/'results.json').write_text(json.dumps(report,indent=2))
    print('Selection:',report['selected_seed'],'before',report['before_chase_net'],'after',report['selected_chase_net'],flush=True)

if __name__=='__main__':main()
