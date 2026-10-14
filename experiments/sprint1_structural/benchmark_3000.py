"""Fixed first-episode benchmark: paired initial states, side/mode breakdown and seed-block uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from model import Policy
from train import create_eval_env
from bot import ChaseBot,PositionBot
from haxballgym import BLUE,RED
from haxballgym.mutator import RandomStateMutator
from benchmark_bots import TacticalBot


@torch.no_grad()
def match(candidate,opponent,stadium,seed,side,mode,scenario,n):
    env=create_eval_env(n,1,1,str(stadium),8)
    if scenario=='random':env.state_mutator=RandomStateMutator(ball_speed=3.,rng=seed)
    obs=env.reset()
    initial_hash=hashlib.sha256(np.concatenate([env.prev_state.ball_pos.ravel(),env.prev_state.ball_vel.ravel(),env.prev_state.player_pos.ravel(),env.prev_state.player_vel.ravel()]).tobytes()).hexdigest()
    active=np.ones(n,dtype=bool);outcomes=np.full(n,np.nan);duration=np.zeros(n,dtype=int)
    stationary=contacts=presses=transitions=0
    traces=[]
    torch.manual_seed(seed)
    captured=[];reward=env.reward_fn
    class Capture:
        def get_rewards(self,state,prev,term,trunc):
            captured.append(state);return reward.get_rewards(state,prev,term,trunc)
    env.reward_fn=Capture()
    for step in range(251):
        pre=env.prev_state
        actions=np.empty((n,2,3),dtype=np.int64)
        actions[:,side]=candidate.act(torch.as_tensor(obs[:,side],dtype=torch.float32),greedy=mode=='greedy').numpy()
        if isinstance(opponent,Policy):
            actions[:,1-side]=opponent.act(torch.as_tensor(obs[:,1-side],dtype=torch.float32),greedy=mode=='greedy').numpy()
        else:actions[:,1-side]=opponent.act(pre,[1-side]).reshape(n,3)
        near=np.linalg.norm(pre.ball_pos-pre.player_pos[:,side],axis=-1)<24.8
        contacts+=int((active&near).sum());presses+=int((active&near&(actions[:,side,2]==1)).sum())
        if active[0]:traces.append(dict(decision=step,ball=pre.ball_pos[0].tolist(),players=pre.player_pos[0].tolist(),actions=actions[0].tolist()))
        before=active.copy();obs,_,term,trunc=env.step(actions);state=captured.pop()
        stationary+=int((before&(np.linalg.norm(state.ball_pos-pre.ball_pos,axis=-1)<.01)).sum())
        transitions+=int(before.sum());duration[before]+=1
        done=before&(term|trunc)
        # scored is the conceding team, not the scoring team.
        outcomes[done]=np.where(state.scored[done]==(BLUE if side==0 else RED),1.,np.where(state.scored[done]==(RED if side==0 else BLUE),0.,.5))
        active[done]=False
        if not active.any():break
    if active.any():raise RuntimeError('Censored episodes: benchmark must finish every first episode')
    return dict(seed=seed,side=side,mode=mode,scenario=scenario,n=n,
                initial_sha256=initial_hash,wins=int((outcomes==1).sum()),draws=int((outcomes==.5).sum()),losses=int((outcomes==0).sum()),
                score=float(outcomes.mean()),mean_seconds=float(duration.mean()*8/60),
                stationary_fraction=stationary/transitions,kick_contact_fraction=presses/contacts if contacts else None,
                outcomes=outcomes.tolist()),traces


def summarize(out,rows):
    rng=np.random.default_rng(18001);groups=[]
    for key in sorted({(r['candidate'],r['opponent'],r['mode'],r['scenario']) for r in rows}):
        cases=[r for r in rows if tuple(r[k] for k in ['candidate','opponent','mode','scenario'])==key]
        seeds=sorted({r['seed'] for r in cases})
        values=np.array([np.mean([r['score'] for r in cases if r['seed']==seed]) for seed in seeds])
        deterministic=key[2:] == ('greedy','kickoff') and key[1]!='chase'
        ci=None if deterministic else np.quantile(np.mean(rng.choice(values,(5000,len(values)),replace=True),axis=1),[.025,.975]).tolist()
        groups.append(dict(zip(['candidate','opponent','mode','scenario'],key),score=float(values.mean()),
                           seed_blocks=len(seeds),episodes=sum(r['n'] for r in cases),seed_block_bootstrap_ci95=ci,
                           deterministic_repetitions=deterministic,
                           mean_seconds=float(np.mean([r['mean_seconds'] for r in cases])),
                           stationary_fraction=float(np.mean([r['stationary_fraction'] for r in cases])),
                           wins=sum(r['wins'] for r in cases),draws=sum(r['draws'] for r in cases),losses=sum(r['losses'] for r in cases),
                           side_scores={str(side):float(np.mean([r['score'] for r in cases if r['side']==side])) for side in [0,1]}))
    (out/'summary.json').write_text(json.dumps(groups,indent=2))
    paired=[]
    for candidate,reference in [('best','origin'),('endpoint','origin'),('endpoint','best')]:
        for opponent,mode,scenario in sorted({(r['opponent'],r['mode'],r['scenario']) for r in rows if r['opponent']!='self'}):
            def index(name):
                return {(r['seed'],r['side']):r for r in rows if (r['candidate'],r['opponent'],r['mode'],r['scenario'])==(name,opponent,mode,scenario)}
            a,b=index(candidate),index(reference)
            if not a or not b:continue
            if a.keys()!=b.keys():continue  # Partial progress reports; final case budget is checked below.
            for k in a:
                if a[k]['initial_sha256']!=b[k]['initial_sha256']:raise RuntimeError('Unpaired initial states')
            seeds=sorted({k[0] for k in a})
            delta=np.array([np.mean([a[seed,side]['score']-b[seed,side]['score'] for side in [0,1]]) for seed in seeds])
            ci=None if (mode,scenario)==('greedy','kickoff') and opponent!='chase' else np.quantile(rng.choice(delta,(5000,len(delta)),replace=True).mean(axis=1),[.025,.975]).tolist()
            paired.append(dict(candidate=candidate,reference=reference,opponent=opponent,mode=mode,scenario=scenario,delta=float(delta.mean()),paired_seed_bootstrap_ci95=ci))
    (out/'paired_deltas.json').write_text(json.dumps(paired,indent=2))
    lines=['# Benchmark 3000 — partidas completas','',
           'Score = vitória + meio empate. Cada caso conta somente a primeira partida de cada ambiente;',
           'gol encerra a partida, timeout em 2000 ticks (33,33 s) conta como empate. Isso não é uma partida multigol.',
           'IC95% descritivo por bootstrap de blocos de seeds, não de episódios supostamente independentes.',
           'Kickoff greedy determinístico repete trajetórias: não recebe intervalo nem alegação de precisão por repetição.',
           'Bots táticos são heurísticas explícitas; não representam força humana certificada.','',
           '| Candidato | Adversário | Modo | Cenário | Score | IC95% por seed | V/E/D |','|---|---|---|---|---:|---|---|']
    for g in groups:
        ci=g['seed_block_bootstrap_ci95'];label=f'{ci[0]:.3f}–{ci[1]:.3f}' if ci else 'determinístico'
        lines.append(f"| {g['candidate']} | {g['opponent']} | {g['mode']} | {g['scenario']} | {g['score']:.4f} | {label} | {g['wins']}/{g['draws']}/{g['losses']} |")
    lines+=['','Self-play simétrico não mede força absoluta. Um score próximo de 0,5 é esperado por troca de lados;',
            'taxa de empates, duração e trajetórias ajudam a distinguir atividade de impasse.',
            'Uma única seed de treinamento continua sendo a limitação deste estudo.', '',
            'Diferenças pareadas e intervalos por seed estão em paired_deltas.json; estados iniciais foram conferidos por hash.',
            'Os intervalos são pontuais e descritivos: não corrigem multiplicidade nem certificam generalização humana.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--envs',type=int,default=32);p.add_argument('--seeds',type=int,default=16);args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    run=ROOT/'runs/sprint1_structural_main_20261014'
    launch=json.loads((run/'launch.json').read_text())
    source_hashes={}
    for name in ['model.py','train.py','environment.py','bot.py','ppo_returns.py']:
        source_hashes[name]=hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
        if source_hashes[name]!=launch['source_sha256'][name]:raise RuntimeError(f'Evaluation source differs from frozen training: {name}')
    checkpoint=torch.load(run/'train/resume.pt',map_location='cpu',weights_only=True)
    if checkpoint['iteration']!=3000:raise RuntimeError('Expected completed 3000-iteration endpoint')
    states={'origin':torch.load(run/'sources/start.pt',map_location='cpu',weights_only=True),'best':checkpoint['best_model'],'endpoint':checkpoint['model']}
    models={}
    for name,state in states.items():
        file=out/f'{name}.pt'
        if file.exists():
            saved=torch.load(file,map_location='cpu',weights_only=True)
            if saved.keys()!=state.keys() or not all(torch.equal(v,state[k]) for k,v in saved.items()):raise RuntimeError('Cannot resume benchmark with changed candidate')
        else:torch.save(state,file)
        models[name]=Policy.from_state_dict(state).eval()
    stadium=out/'stadium.hbs';stadium.write_text(checkpoint['stadium_text'])
    config=dict(status='running',endpoint_iteration=3000,best_iteration=checkpoint['best_iteration'],seeds=list(range(9101,9101+args.seeds)),envs=args.envs,
                map_sha256=hashlib.sha256(stadium.read_bytes()).hexdigest(),checkpoint_sha256={name:hashlib.sha256((out/f'{name}.pt').read_bytes()).hexdigest() for name in models},
                fixed_first_episode=True,max_ticks=2000,selection='best checkpoint selected before unseen evaluation; endpoint evaluated separately; no automatic promotion')
    config['expected_cases']=3*(10*(6*args.seeds+2)+8*args.seeds)
    config['evaluation_source_sha256']=source_hashes
    config['benchmark_source_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['benchmark_3000.py','benchmark_bots.py']}
    if (out/'config.json').exists():
        prior=json.loads((out/'config.json').read_text())
        for k in ['seeds','envs','checkpoint_sha256','map_sha256','benchmark_source_sha256','evaluation_source_sha256']:
            if prior[k]!=config[k]:raise RuntimeError(f'Benchmark recipe changed: {k}')
    (out/'config.json').write_text(json.dumps(config,indent=2));rows=[]
    if (out/'cases.jsonl').exists():
        lines=(out/'cases.jsonl').read_text().splitlines()
        for i,line in enumerate(lines):
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i<len(lines)-1:raise
        (out/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def casekey(r):return tuple(r[k] for k in ['candidate','opponent','mode','scenario','seed','side'])
    seen={casekey(r) for r in rows}
    if len(seen)!=len(rows):raise RuntimeError('Duplicate benchmark case')
    for name,candidate in models.items():
        opponents=['self','origin','best','endpoint','chase','striker','keeper',*TacticalBot.STYLES]
        for opponent_name in opponents:
            for mode in ['greedy','sampled']:
                for scenario in ['random','kickoff']:
                    # Deterministic kickoff requires one case, not 16 fake replicates.
                    seeds=config['seeds'][:1] if (mode,scenario)==('greedy','kickoff') and opponent_name!='chase' else config['seeds']
                    for seed in seeds:
                        for side in [0,1]:
                            key=(name,opponent_name,mode,scenario,seed,side)
                            if key in seen:continue
                            opponent=(candidate if opponent_name=='self' else models[opponent_name]) if opponent_name in ['self',*models] else (ChaseBot(seed=seed) if opponent_name=='chase' else PositionBot(defender=opponent_name=='keeper') if opponent_name in ['striker','keeper'] else TacticalBot(opponent_name))
                            row,trace=match(candidate,opponent,stadium,seed,side,mode,scenario,args.envs)
                            row.update(candidate=name,opponent=opponent_name)
                            with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                            rows.append(row)
                            seen.add(key)
                            if seed==config['seeds'][0] and side==0:
                                d=out/'trajectories';d.mkdir(exist_ok=True);(d/f'{name}_{opponent_name}_{mode}_{scenario}.json').write_text(json.dumps(trace))
                    print(name,opponent_name,mode,scenario,len(rows),flush=True)
                    (out/'live.json').write_text(json.dumps(dict(status='running',cases=len(rows),candidate=name,opponent=opponent_name,mode=mode,scenario=scenario)))
        summarize(out,rows)
    if len(rows)!=config['expected_cases']:raise RuntimeError('Incomplete case budget')
    summarize(out,rows);config.update(status='completed',cases=len(rows),finished_at=time.time());(out/'config.json').write_text(json.dumps(config,indent=2));(out/'live.json').write_text(json.dumps(dict(status='completed',cases=len(rows))))


if __name__=='__main__':main()
