"""Audit completed 3000 benchmark and publish summaries/figures, without raw runs or room secrets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'runs/sprint1_benchmark_3000_20261014')
    args=parser.parse_args();run=args.run
    config=json.loads((run/'config.json').read_text())
    if config['status']!='completed':raise RuntimeError('Benchmark not completed')
    rows=[json.loads(s) for s in (run/'cases.jsonl').read_text().splitlines()]
    assert len(rows)==config['expected_cases']==config['cases']
    keys=set();initial={}
    for r in rows:
        key=tuple(r[k] for k in ['candidate','opponent','mode','scenario','seed','side'])
        assert key not in keys;keys.add(key)
        assert r['n']==len(r['outcomes'])==r['wins']+r['draws']+r['losses']==config['envs']
        assert all(v in (0.,.5,1.) for v in r['outcomes'])
        assert math.isclose(r['score'],float(np.mean(r['outcomes'])))
        assert 0<r['mean_seconds']<=2000/60+1e-9
        pairing=(r['scenario'],r['seed'],r['side'])
        if pairing in initial:assert initial[pairing]==r['initial_sha256']
        initial[pairing]=r['initial_sha256']
        for v in r.values():
            if isinstance(v,float):assert math.isfinite(v)
    assert hashlib.sha256((run/'stadium.hbs').read_bytes()).hexdigest()==config['map_sha256']
    for name,sha in config['checkpoint_sha256'].items():
        assert hashlib.sha256((run/f'{name}.pt').read_bytes()).hexdigest()==sha
        state=torch.load(run/f'{name}.pt',weights_only=True,map_location='cpu')
        assert all(torch.isfinite(t).all() for t in state.values())
    groups=json.loads((run/'summary.json').read_text())
    paired=json.loads((run/'paired_deltas.json').read_text())
    assert len(groups)==3*11*4 and len(paired)==3*10*4
    dest=ROOT/'experiments/sprint1_structural/benchmark_3000';dest.mkdir(exist_ok=True)
    for name in ['REPORT.md','summary.json','paired_deltas.json','config.json']:
        shutil.copy2(run/name,dest/name)
    def find(candidate,opponent,mode,scenario):
        return next(g for g in groups if (g['candidate'],g['opponent'],g['mode'],g['scenario'])==(candidate,opponent,mode,scenario))
    opponents=['origin','best','endpoint','chase','striker','keeper','interceptor','counter_defender','dribbler','aimed_striker']
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharey=True)
    for ax,(mode,scenario) in zip(axes.ravel(),[('greedy','random'),('greedy','kickoff'),('sampled','random'),('sampled','kickoff')]):
        x=np.arange(len(opponents))
        for j,name in enumerate(['origin','best','endpoint']):
            gs=[find(name,o,mode,scenario) for o in opponents]
            means=np.array([g['score'] for g in gs]);lo=[];hi=[]
            for g in gs:
                ci=g['seed_block_bootstrap_ci95']
                lo.append(max(0,g['score']-ci[0]) if ci else 0);hi.append(max(0,ci[1]-g['score']) if ci else 0)
            ax.bar(x+(j-1)*.25,means,.25,label=name,yerr=np.array([lo,hi]),capsize=2)
        ax.set(title=f'{mode} / {scenario}',ylim=(0,1.05),ylabel='Score (V + ½E)',xticks=x,xticklabels=opponents)
        ax.tick_params(axis='x',rotation=30);ax.axhline(.5,color='gray',linestyle=':',linewidth=1);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend();fig.suptitle('Benchmark em primeiras partidas completas — IC95% por blocos de seed\nKickoff greedy determinístico sem intervalo; bots heurísticos, sem certificação humana')
    fig.tight_layout();fig.savefig(dest/'performance.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    labels=[];draws=[];duration=[]
    for name in ['origin','best','endpoint']:
        for mode in ['greedy','sampled']:
            for scenario in ['random','kickoff']:
                g=find(name,'self',mode,scenario);labels.append(f'{name}\n{mode}/{scenario}')
                draws.append(g['draws']/g['episodes']);duration.append(g['mean_seconds'])
    axes[0].bar(range(len(labels)),draws);axes[0].set(title='Self-play: fração de empates',ylim=(0,1.05))
    axes[1].bar(range(len(labels)),duration);axes[1].set(title='Self-play: duração média',ylabel='Segundos',ylim=(0,35))
    for ax in axes:ax.set_xticks(range(len(labels)),labels,rotation=60,fontsize=7);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(dest/'selfplay.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ax,opponent in zip(axes.ravel(),['self','origin','interceptor','counter_defender','dribbler','aimed_striker']):
        trace=json.loads((run/'trajectories'/f'best_{opponent}_greedy_random.json').read_text())
        ball=np.array([t['ball'] for t in trace]);players=np.array([t['players'] for t in trace])
        ax.plot(ball[:,0],ball[:,1],color='black',linewidth=1,label='bola')
        for side,color in [(0,'red'),(1,'blue')]:
            ax.plot(players[:,side,0],players[:,side,1],color=color,alpha=.7,label='modelo' if side==0 else 'oponente')
            ax.scatter(*players[0,side],color=color,marker='o',s=20)
            ax.scatter(*players[-1,side],color=color,marker='x',s=30)
        ax.set(title=f'best × {opponent}',aspect='equal',xlabel='x',ylabel='y');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('Exemplos pré-fixados: seed 9101, ambiente 0, lado vermelho, greedy/random\nCírculo = início; × = último estado antes da decisão terminal. Não são melhores partidas escolhidas.')
    fig.tight_layout();fig.savefig(dest/'trajectories.png',dpi=160);plt.close(fig)
    audit=dict(complete=True,cases=len(rows),first_episodes=sum(r['n'] for r in rows),unique_cases=len(keys),
               paired_initial_states_verified=True,finite_weights=True,hashes_verified=True,
               single_training_seed=True,best_iteration=config['best_iteration'],endpoint_iteration=3000)
    (dest/'audit.json').write_text(json.dumps(audit,indent=2))
    report=dest/'REPORT.md';text=report.read_text()
    text+='\n![Desempenho](performance.png)\n\n![Self-play](selfplay.png)\n\n![Trajetórias pré-fixadas](trajectories.png)\n'
    report.write_text(text)
    print(json.dumps(audit))


if __name__=='__main__':main()
