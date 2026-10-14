"""Plot measured training reward per transition, with its weighted components."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
WINDOW=50


def main():
    rows=[json.loads(s) for s in (ROOT/'runs/sprint1_structural_main_20261014/train/training.jsonl').read_text().splitlines()]
    x=np.array([r['iteration'] for r in rows])
    total=np.array([r['reward_mean'] for r in rows])
    keys=['GoalReward','DiscountedPocketPotential','DiscountedBallGoalPotential']
    values={k:np.array([r['reward_terms_per_sample'][k] for r in rows]) for k in keys}
    assert np.array_equal(x,np.arange(1,3001))
    assert np.isfinite(total).all() and all(np.isfinite(v).all() for v in values.values())
    error=float(np.max(np.abs(total-sum(values.values()))))
    assert error<1e-6
    def smooth(v):return np.convolve(v,np.ones(WINDOW)/WINDOW,mode='valid')
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    axes[0].plot(x,total,color='#9eafbf',alpha=.5,linewidth=.65,label='Média de cada iteração')
    axes[0].plot(x[WINDOW-1:],smooth(total),color='#1667b1',linewidth=2,label='Média móvel: 50 iterações')
    axes[0].set_title('Recompensa total observada no treinamento')
    for key,label,color in [('GoalReward','Gol (peso ±25)','#238743'),
                            ('DiscountedPocketPotential','Posicionamento PBRS (peso 0,15)','#d17b14'),
                            ('DiscountedBallGoalPotential','Progresso PBRS (peso 0,08)','#8c4cba')]:
        axes[1].plot(x[WINDOW-1:],smooth(values[key]),label=label,color=color,linewidth=1.8)
    axes[1].set(title='Componentes já ponderados — média móvel de 50 iterações',xlabel='Iteração')
    axes[1].axhline(0,color='gray',linewidth=.7)
    for ax in axes:
        ax.set_ylabel('Recompensa média por transição')
        ax.grid(alpha=.2)
        ax.axvline(20,color='#555555',linestyle=':',linewidth=1,label='Fim da calibração do crítico (it20)')
        ax.legend(fontsize=9,loc='upper left')
        ax.set_xlim(1,3000)
    fig.suptitle('Sprint 1 — 3.000 iterações, 512 ambientes × 64 decisões por rollout',fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT/'reward_curve.png',dpi=170)
    fig.savefig(OUT/'reward_curve.svg')
    plt.close(fig)
    svg=OUT/'reward_curve.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    audit=dict(iterations=len(rows),smoothing_window=WINDOW,unit='weighted reward per learner transition',
               first_50_mean=float(total[:WINDOW].mean()),last_50_mean=float(total[-WINDOW:].mean()),
               max_component_sum_error=error,
               last_50_components={k:float(v[-WINDOW:].mean()) for k,v in values.items()})
    (OUT/'reward_curve.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit))


if __name__=='__main__':main()
