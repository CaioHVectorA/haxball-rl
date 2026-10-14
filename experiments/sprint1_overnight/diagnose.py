"""Preserve raw curves; report paired differences without episode pseudoreplication."""
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/sprint1_timeout_full_20261014'
OUT=ROOT/'experiments/sprint1_overnight'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,3,figsize=(16,11),layout='constrained')
colors={'B0':'#3066BE','E1':'#D1495B'}
all_rows={}
for seed in range(3):
 for arm in ['B0','E1']:
  name=f'{arm}_seed{seed}';p=RUN/name
  es=[json.loads(x) for x in (p/'evaluations.jsonl').read_text().splitlines()]
  ts=[json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()]
  all_rows[name]=(es,ts)
  for col,field in enumerate(['score','eval_random_score','eval_kickoff_score']):
   ax=axs[seed,col]
   ax.plot([x['iteration'] for x in es],[x[field] for x in es],'-o',color=colors[arm],label=arm,markersize=4)
   ax.set(title=f'Seed {seed} — {field}',xlabel='Iteração',ylabel='Score greedy',ylim=(.35,.95));ax.grid(alpha=.2);ax.legend()
fig.suptitle('B0/E1: avaliação por seed e cenário — endpoint não é o pico da curva',fontsize=15)
fig.savefig(OUT/'evaluation_curves.png',dpi=150);fig.savefig(OUT/'evaluation_curves.svg');plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(16,8),layout='constrained')
fields=['timeout_bootstrap_mean','explained_variance','trunk_grad_cosine','grad_norm_actor_pre_clip','grad_norm_critic_pre_clip','approximate_kl']
for ax,field in zip(axs.flat,fields):
 for name,(es,ts) in all_rows.items():
  arm=name[:2];seed=int(name[-1]);x=[];y=[]
  for i in range(0,300,10):
   ys=[r[field] for r in ts[i:i+10] if r[field] is not None]
   if ys:x.append(i+5);y.append(st.mean(ys))
  ax.plot(x,y,color=colors[arm],alpha=.7,linestyle=['-','--',':'][seed],label=name)
 ax.set(title=field,xlabel='Iteração (blocos de 10)');ax.grid(alpha=.2)
axs[0,0].legend(ncol=2)
fig.suptitle('Diagnósticos: normas medem gradientes, não o update efetivo do Adam',fontsize=14)
fig.savefig(OUT/'training_diagnostics.png',dpi=150);fig.savefig(OUT/'training_diagnostics.svg');plt.close(fig)
summary=json.loads((RUN/'summary.json').read_text())
case_deltas=[]
for scenario in ['random','kickoff']:
 for opponent in ['chase','chase_no_noise','striker','keeper','initial_policy']:
  for side in [0,1]:
   deltas=[]
   for seed in range(3):
    a,b=[next(r for r in summary['results'] if r['name']==f'{arm}_seed{seed}') for arm in ['B0','E1']]
    def score(r):return st.mean(c['score'] for c in r['final_cases'] if c['scenario']==scenario and c['opponent']==opponent and c['side']==side)
    deltas.append(score(b)-score(a))
   case_deltas.append(dict(scenario=scenario,opponent=opponent,side=side,deltas=deltas,mean=st.mean(deltas)))
(OUT/'case_deltas.json').write_text(json.dumps(case_deltas,indent=2))
fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
labels=[f"{r['scenario']} / {r['opponent']} / {'R' if r['side']==0 else 'B'}" for r in case_deltas]
ax.barh(labels,[r['mean'] for r in case_deltas],color=['#3066BE' if r['mean']>=0 else '#D1495B' for r in case_deltas])
ax.axvline(0,color='black',linewidth=.8);ax.set(xlabel='Diferença média E1 − B0, score no endpoint',title='Três seeds de treinamento; média sobre duas seeds de avaliação por caso')
fig.savefig(OUT/'case_deltas.png',dpi=150);plt.close(fig)
print(summary['aggregate'])
print('Plots saved:',OUT)
