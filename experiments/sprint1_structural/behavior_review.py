"""Descriptive diagnostics and a replay from pre-fixed benchmark trajectories; no training."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from train import create_eval_env
from haxballgym import RED


def main():
    run=ROOT/'runs/sprint1_benchmark_3000_20261014'
    out=ROOT/'experiments/sprint1_structural/behavior';out.mkdir(exist_ok=True)
    env=create_eval_env(1,1,1,str(run/'stadium.hbs'),8);env.reset();state=env.prev_state
    goal=state.goal_center(True)[0,0]
    opponents=['self','origin','interceptor','counter_defender','dribbler','aimed_striker']
    cases=[json.loads(s) for s in (run/'cases.jsonl').read_text().splitlines()]
    rows=[]
    for candidate in ['origin','best','endpoint']:
        for opponent in opponents:
            for mode in ['greedy','sampled']:
                for scenario in ['random','kickoff']:
                    trace=json.loads((run/'trajectories'/f'{candidate}_{opponent}_{mode}_{scenario}.json').read_text())
                    ball=np.array([t['ball'] for t in trace]);players=np.array([t['players'] for t in trace]);actions=np.array([t['actions'] for t in trace])
                    ego=players[:,0];distance=np.linalg.norm(ball-ego,axis=1)
                    direction=goal-ball;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
                    pocket=ball-22*direction
                    near=distance<24.8;kick=actions[:,0,2]==1
                    cosine=np.sum((ball-ego)*direction,axis=1)/np.maximum(distance,1e-9)
                    case=next(c for c in cases if (c['candidate'],c['opponent'],c['mode'],c['scenario'],c['seed'],c['side'])==(candidate,opponent,mode,scenario,9101,0))
                    row=dict(candidate=candidate,opponent=opponent,mode=mode,scenario=scenario,
                             decisions=len(trace),outcome=case['outcomes'][0],mean_pocket_distance=float(np.linalg.norm(ego-pocket,axis=1).mean()),
                             kick_pressed_fraction=float(kick.mean()),far_kick_pressed_fraction=float((kick&~near).mean()),
                             near_ball_decisions=int(near.sum()),kick_pressed_near_ball=float(kick[near].mean()) if near.any() else None,
                             aligned_near_ball_fraction=float((cosine[near]>.8).mean()) if near.any() else None)
                    rows.append(row)
    (out/'examples.json').write_text(json.dumps(dict(seed=9101,environment=0,side=0,
           independent_training_seeds=1,description='72 pre-fixed example traces; one evaluation seed, not a statistical skill estimate',rows=rows),indent=2))
    fig,axes=plt.subplots(2,2,figsize=(10,7));artists=[];traces=[]
    for ax,opponent in zip(axes.ravel(),['self','origin','interceptor','dribbler']):
        trace=json.loads((run/'trajectories'/f'best_{opponent}_greedy_random.json').read_text());traces.append(trace)
        if state.walls is not None:
            for x0,y0,x1,y1 in state.walls:ax.plot([x0,x1],[y0,y1],color='#dddddd',linewidth=.6)
        for p0,p1,team in zip(state.goal_p0,state.goal_p1,state.goal_team):
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color='red' if team==RED else 'blue',linewidth=3)
        ax.axvline(0,color='#dddddd',linewidth=.6)
        ax.set(xlim=(-490,490),ylim=(-245,245),aspect='equal',title=f'Modelo it2550 × {opponent}')
        ax.set_xticks([]);ax.set_yticks([])
        red,=ax.plot([],[],'o',color='#df3434',markersize=9,label='modelo')
        blue,=ax.plot([],[],'o',color='#327ad6',markersize=9,label='oponente')
        ball,=ax.plot([],[],'o',color='black',markersize=4,label='bola')
        trail,=ax.plot([],[],color='black',alpha=.25,linewidth=.8)
        label=ax.text(.02,.96,'',transform=ax.transAxes,va='top',fontsize=8)
        artists.append((red,blue,ball,trail,label))
    axes[0,0].legend(loc='lower left',fontsize=7)
    fig.suptitle('Exemplos pré-fixados: seed 9101, ambiente 0, greedy/random\nReplay das decisões (8 ticks); reprodução ≈2×. Modelo vermelho, oponente azul.',fontsize=11)
    fig.subplots_adjust(top=.84,bottom=.04,left=.03,right=.99,wspace=.06,hspace=.45)
    def update(frame):
        result=[]
        for trace,objects in zip(traces,artists):
            i=min(frame*2,len(trace)-1);t=trace[i];red,blue,ball,trail,label=objects
            for obj,pos in [(red,t['players'][0]),(blue,t['players'][1]),(ball,t['ball'])]:obj.set_data([pos[0]],[pos[1]])
            recent=np.array([s['ball'] for s in trace[max(0,i-20):i+1]]);trail.set_data(recent[:,0],recent[:,1])
            label.set_text(f'{i*8/60:.1f}s | pressão de chute: {t["actions"][0][2]}'+(' | fim do registro' if i==len(trace)-1 else ''))
            result.extend(objects)
        return result
    animation=FuncAnimation(fig,update,frames=(max(map(len,traces))+1)//2,interval=133,blit=False)
    animation.save(out/'replay_examples.gif',writer=PillowWriter(fps=8))
    update(0);fig.savefig(out/'replay_first_frame.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(example_traces=len(rows),gif=str(out/'replay_examples.gif'))))


if __name__=='__main__':main()
