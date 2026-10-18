"""Fixed evaluation panel; keep per-case metrics and explicit sample counts."""
import numpy as np
import torch
from haxballgym import BLUE, RED
from haxballgym.mutator import RandomStateMutator
from bot import ChaseBot, PositionBot


@torch.no_grad()
def panel(model, reference, stadium, seed, callback, progress, n=16, steps=400, tick_skip=8, eval_seeds=(1701, 1702), greedy=True, extra_opponents=None, n_red=1, n_blue=1):
    from train import create_eval_env, TeamRandomSetter
    device = next(model.parameters()).device
    rows=[]
    neural_opponents={'initial_policy':reference, **(extra_opponents or {})}
    opponent_names=['chase','chase_no_noise','striker','keeper']+list(neural_opponents)
    total=len(opponent_names)*2*len(eval_seeds)*2
    case=0
    for scenario in ['random','kickoff']:
        for opponent in opponent_names:
            for eval_seed in eval_seeds:
                for side in [0,1]:
                    # Isolate evaluation RNG from the training sampling sequence.
                    with torch.random.fork_rng(devices=[device.index or 0] if device.type=="cuda" else []):
                        torch.manual_seed(eval_seed)
                        env=create_eval_env(n,n_red,n_blue,stadium,tick_skip)
                        if scenario=='random':
                            setter=TeamRandomSetter if (n_red,n_blue)==(2,2) else RandomStateMutator
                            env.state_mutator=setter(ball_speed=3.,rng=eval_seed)
                        obs=env.reset()
                        bot=(ChaseBot(seed=eval_seed,jiggle=0 if opponent=='chase_no_noise' else .25)
                             if opponent.startswith('chase') else PositionBot(defender=opponent=='keeper'))
                        gf=ga=wins=draws=losses=stationary=contacts=kick_contacts=0
                        episode_return=np.zeros(n)
                        last=env.reward_fn
                        captured=[]
                        class Capture:
                            def get_rewards(self,state,prev,terminated,truncated):
                                captured.append(state)
                                return last.get_rewards(state,prev,terminated,truncated)
                        env.reward_fn=Capture()
                        for step in range(steps):
                            pre=env.prev_state
                            players = np.arange(n_red) if side==0 else np.arange(n_red,n_red+n_blue)
                            other = np.arange(n_red,n_red+n_blue) if side==0 else np.arange(n_red)
                            actions=np.empty((n,n_red+n_blue,3),dtype=np.int64)
                            own_obs=obs[:,players].reshape(-1,obs.shape[-1])
                            actions[:,players]=model.act(torch.as_tensor(own_obs,device=device,dtype=torch.float32),greedy=greedy).cpu().numpy().reshape(n,len(players),3)
                            if opponent in neural_opponents:
                                other_obs=obs[:,other].reshape(-1,obs.shape[-1])
                                actions[:,other]=neural_opponents[opponent].act(torch.as_tensor(other_obs,device=device,dtype=torch.float32),greedy=greedy).cpu().numpy().reshape(n,len(other),3)
                            else:actions[:,other]=bot.act(pre,other,n_red=n_red).reshape(n,len(other),3)
                            contact=np.linalg.norm(pre.ball_pos[:,None]-pre.player_pos[:,players],axis=-1)<24.8
                            contacts+=int(contact.sum());kick_contacts+=int((contact&(actions[:,players,2]==1)).sum())
                            obs,_,term,trunc=env.step(actions)
                            state=captured.pop()
                            good=state.scored==(BLUE if side==0 else RED)
                            bad=state.scored==(RED if side==0 else BLUE)
                            gf+=int(good.sum());ga+=int(bad.sum())
                            episode_return+=good.astype(int)-bad.astype(int)
                            done=term|trunc
                            wins+=int(((episode_return>0)&done).sum());losses+=int(((episode_return<0)&done).sum())
                            draws+=int(((episode_return==0)&done).sum());episode_return[done]=0
                            stationary+=int((np.linalg.norm(state.ball_pos-pre.ball_pos,axis=-1)<.01).sum())
                            if step%100==0:progress(case,total,opponent,scenario)
                        completed=wins+draws+losses
                        row=dict(opponent=opponent,scenario=scenario,seed=eval_seed,side=side,
                          policy_mode="greedy" if greedy else "sampled",
                          decisions=steps,envs=n,red=n_red,blue=n_blue,gf=gf,ga=ga,wins=wins,draws=draws,losses=losses,
                          completed_episodes=completed,score=(wins+.5*draws)/completed if completed else None,
                          stationary_fraction=stationary/(n*steps),contacts=contacts,
                          kick_contact_fraction=kick_contacts/contacts if contacts else None)
                        rows.append(row);case+=1;callback(row);progress(case,total,opponent,scenario)
    return rows


def balanced_score(rows):
    values=[r['score'] for r in rows if r['score'] is not None]
    return float(np.mean(values)) if values else None
