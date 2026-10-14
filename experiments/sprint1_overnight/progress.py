"""Summarize existing continuation telemetry without changing a live experiment."""
import json
import math
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/sprint1_overnight_model_20261014'
OUT=ROOT/'experiments/sprint1_overnight'

def main():
    def read_rows(path):
        lines=path.read_text().splitlines()
        records=[]
        for i,line in enumerate(lines):
            try:records.append(json.loads(line))
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
        return records
    def check_finite(obj):
        if isinstance(obj,float):assert math.isfinite(obj)
        elif isinstance(obj,dict):
            for v in obj.values():check_finite(v)
        elif isinstance(obj,list):
            for v in obj:check_finite(v)
    train=read_rows(RUN/'train/training.jsonl')
    evals=read_rows(RUN/'train/evaluations.jsonl')
    for row in train+evals:check_finite(row)
    assert all(r['iteration']==i+1 and r['samples']==(i+1)*32768 for i,r in enumerate(train))
    live=json.loads((RUN/'train/live.json').read_text())
    best=json.loads((RUN/'train/meta.json').read_text())
    summaries=[]
    for e in evals:
        summaries.append(dict(iteration=e['iteration'],score=e['score'],
            random=e['eval_random_score'],kickoff=e['eval_kickoff_score'],
            **{f'heuristic_{s}':statistics.mean(r['score'] for r in e['cases']
                 if r['scenario']==s and r['opponent']!='initial_policy') for s in ['random','kickoff']}))
    report=dict(status=live['status'],updated_at=live['updated_at'],last_training=train[-1],
                best_checkpoint_iteration=best['it'],best_selection_score=best['evaluation_score'],
                evaluation_summaries=summaries,finite_metrics=True,continuous_budget=True)
    (OUT/'continuation_progress.json').write_text(json.dumps(report,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for field in ['score','random','kickoff']:
        axs[0].plot([e['iteration'] for e in summaries],[e[field] for e in summaries],label=field)
    axs[0].set(title='Painel interno (referência = origem B0)',xlabel='Iteração',ylabel='Score');axs[0].legend()
    for s in ['random','kickoff']:
        axs[1].plot([e['iteration'] for e in summaries],[e[f'heuristic_{s}'] for e in summaries],label=s)
    axs[1].set(title='Somente quatro adversários heurísticos',xlabel='Iteração',ylabel='Score');axs[1].legend()
    for field in ['explained_variance','stationary_fraction']:
        x=[];y=[]
        for i in range(0,len(train),25):
            vals=[r[field] for r in train[i:i+25] if r[field] is not None]
            if vals:x.append(i+1);y.append(statistics.mean(vals))
        axs[2].plot(x,y,label=field)
    axs[2].set(title='Diagnósticos (blocos de 25)',xlabel='Iteração');axs[2].legend()
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(OUT/'continuation_progress.png',dpi=160)
    fig.savefig(OUT/'continuation_progress.svg');plt.close(fig)
    first,last=summaries[0],summaries[-1]
    lines=['# Continuação da madrugada — acompanhamento', '',f"Estado: {live['status']}; atualizado em {live['updated_at']} (UTC).",'',
      f"Última iteração registrada: {train[-1]['iteration']} — {train[-1]['samples']:,} amostras.",
      f"Melhor checkpoint selecionado: iteração {best['it']}, score {best['evaluation_score']:.5f}.",'',
      '| Medida | Início | Última avaliação |','|---|---:|---:|']
    for f in ['score','random','kickoff','heuristic_random','heuristic_kickoff']:
        lines.append(f'| {f} | {first[f]:.5f} | {last[f]:.5f} |')
    lines += ['',f"Última avaliação: iteração {last['iteration']}.",'',
      '![Curvas da continuação](continuation_progress.png)','',
      'Métricas finitas e contadores de amostras contínuos foram verificados. A referência',
      'initial_policy é o checkpoint B0 de origem. O score balanceado anterior usa outra',
      'referência; a comparação com ele exige avaliação alinhada. O recorte heurístico',
      'mantém adversários comparáveis. Mudanças greedy oscilam entre checkpoints;',
      'mais iterações não garantem melhora monotônica. O melhor checkpoint é selecionado',
      'no próprio painel e precisa de avaliação posterior. Este acompanhamento não altera',
      'o experimento nem promove qualquer modelo.']
    if live['status']=='completed':
        meta=json.loads((RUN/'train/meta_final.json').read_text())
        assert meta['it']==live['completed_iterations']==train[-1]['iteration']
        assert (RUN/'train/model_final.pt').is_file()
        lines += ['',f"Endpoint salvo e metadados conferidos: iteração {meta['it']}."]
    (OUT/'CONTINUATION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['status','best_checkpoint_iteration','best_selection_score']},indent=2))

if __name__=='__main__':main()
