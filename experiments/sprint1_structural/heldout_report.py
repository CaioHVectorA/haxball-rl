"""Report an independent candidate check and preserve the exact evaluated policy."""
import json
from pathlib import Path
import statistics
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from training_checkpoint import atomic_torch_save
from telemetry import atomic_json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/sprint1_structural'
VALID=ROOT/'runs/sprint1_structural_heldout_20261014'
RUN=ROOT/'runs/sprint1_structural_main_20261014'
comparison=json.loads((VALID/'comparison.json').read_text());results=comparison['results']
cases=[json.loads(x) for x in (VALID/'cases.jsonl').read_text().splitlines()]
assert len(cases)==336 and len({(c['candidate'],c['policy_mode'],c['scenario'],c['seed'],c['side'],c['opponent']) for c in cases})==336
assert all(c['completed_episodes']>0 for c in cases)
per_opponent=[]
for name in ['initial_policy','anchor_0','anchor_1','chase','chase_no_noise','striker','keeper']:
 means=[statistics.mean(c['score'] for c in cases if c['candidate']==candidate and c['opponent']==name) for candidate in ['origin','structural']]
 per_opponent.append(dict(opponent=name,origin=means[0],structural=means[1],delta=means[1]-means[0]))
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
labels=['greedy/random','greedy/kickoff','sampled/random','sampled/kickoff']
for offset,candidate,color in [(-.18,'origin','#D1495B'),(.18,'structural','#3066BE')]:
 vals=[next(r[s] for r in results if r['candidate']==candidate and r['mode']==mode) for mode in ['greedy','sampled'] for s in ['random','kickoff']]
 bars=axes[0].bar([i+offset for i in range(4)],vals,width=.35,label=candidate,color=color)
 axes[0].bar_label(bars,fmt='%.3f',fontsize=8)
axes[0].set_xticks(range(4),labels,rotation=20);axes[0].set_ylim(0,1);axes[0].legend();axes[0].set_title('Três seeds inéditas, referência comum')
axes[1].barh([r['opponent'] for r in per_opponent],[r['delta'] for r in per_opponent],color='#3066BE')
axes[1].axvline(0,color='black');axes[1].set_title('Diferença estrutural − origem por adversário')
fig.savefig(OUT/'heldout.png',dpi=160);fig.savefig(OUT/'heldout.svg');plt.close(fig)
lines=['# Validação independente do primeiro candidato estrutural','',
       f"Checkpoint avaliado: iteração {comparison['config']['iteration']} do novo treino.",
       'Seeds 5701–5703, 8 ambientes, 500 decisões por caso, ambos os lados, dois modos,',
       'dois cenários e sete adversários. 336 casos únicos e completos foram verificados.', '',
       '| Candidato | Modo | Média | Random | Kickoff |','|---|---|---:|---:|---:|']
for r in results:lines.append(f"| {r['candidate']} | {r['mode']} | {r['score']:.5f} | {r['random']:.5f} | {r['kickoff']:.5f} |")
lines+=['','![Avaliação inédita](heldout.png)','','| Adversário | Origem | Estrutural | Diferença |','|---|---:|---:|---:|']
for r in per_opponent:lines.append(f"| {r['opponent']} | {r['origin']:.5f} | {r['structural']:.5f} | {r['delta']:+.5f} |")
lines+=['','Todos os quatro grupos modo×cenário melhoraram nessa comparação. O mínimo foi',
        'de 0,69005 para 0,82938. É um avanço observado fora das seeds do painel usado',
        'para selecionar o checkpoint, incluindo referências neurais. Não se infere',
        'robustez em diferentes seeds de treinamento: esta é uma continuação, seed 0.',
        'Nem se atribui o ganho isoladamente ao PBRS, à separação ou ao currículo, pois',
        'o preset combina essas intervenções. O teste não certifica sala real/equipes.',
        '', 'O checkpoint exato fica em `models/sprint1_validated/model.pt`, com metadados',
        'e mapa. Ele é distinto de `models/sprint1_latest`, cujo candidato e resume',
        'podem avançar conforme o treino continua. Não foi substituído o modelo original.']
(OUT/'HELDOUT.md').write_text('\n'.join(lines)+'\n')
(OUT/'heldout_results.json').write_text(json.dumps(dict(comparison=comparison,per_opponent=per_opponent),indent=2))
dest=ROOT/'models/sprint1_validated';dest.mkdir(exist_ok=True)
state=torch.load(VALID/'structural.pt',weights_only=True)
assert all(torch.isfinite(v).all() for v in state.values())
atomic_torch_save(state,dest/'model.pt')
(dest/'stadium.hbs').write_bytes((RUN/'maps/futsal_1v1.hbs').read_bytes())
meta=dict(json.loads((RUN/'train/config.json').read_text()),it=comparison['config']['iteration'],
          stadium='models/sprint1_validated/stadium.hbs',evaluation_score=None,
          heldout_validation=comparison)
atomic_json(dest/'meta.json',meta,durable=True)
print(per_opponent)
