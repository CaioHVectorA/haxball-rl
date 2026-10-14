"""Small-budget paired comparisons; do not turn this into a causal victory claim."""
import json
from pathlib import Path
import statistics
import torch
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/sprint1_structural_ablation_20261014'
OUT=ROOT/'experiments/sprint1_structural'
variants=['C1_corrected_shared','C2_pbrs_shared','C3_pbrs_separate','C4_pbrs_separate_calibrated']
results=[]
for seed in [0,1]:
 initial_states=[]
 initial_cases=[]
 for variant in variants:
  p=RUN/f'{variant}_seed{seed}'
  if not (p/'live.json').exists() or json.loads((p/'live.json').read_text())['status']!='completed':
   raise RuntimeError(f'Not completed: {p}')
  rows=[json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()]
  assert len(rows)==40 and rows[-1]['samples']==163840
  es=[json.loads(x) for x in (p/'evaluations.jsonl').read_text().splitlines()]
  initial_states.append(torch.load(p/'initial_policy.pt',weights_only=True))
  initial_cases.append(es[0]['cases'])
  results.append(dict(seed=seed,variant=variant,baseline=es[0]['score'],endpoint=es[-1]['score'],
                       mean_score=es[-1]['mean_score'],groups=es[-1]['mode_scenario_scores']))
 actor_keys=[k for k in initial_states[0] if k.startswith(('trunk.','head_'))]
 assert all(torch.equal(initial_states[0][k],state[k]) for k in actor_keys for state in initial_states)
 assert all(initial_cases[0]==cases for cases in initial_cases)
means={v:statistics.mean(r['endpoint'] for r in results if r['variant']==v) for v in variants}
lines=['# Ablação curta de componentes','',
       'Dois seeds, 40 iterações ×64 ambientes: **163.840 amostras por braço/seed**.',
       'Ator inicial tensor a tensor, avaliação inicial, currículo e orçamento foram',
       'conferidos iguais. C4 usa o warmup já implementado antes dos resultados; ele',
       'aloca 20 iterações ao crítico e 20 ao ator, enquanto os demais treinam o ator',
       'em 40. Esse é o tratamento comparado, não igualdade de número de passos do ator.', '',
       '| Variante | Seed 0 | Seed 1 | Média do mínimo |','|---|---:|---:|---:|']
for v in variants:
 vals=[next(r['endpoint'] for r in results if r['variant']==v and r['seed']==s) for s in [0,1]]
 lines.append(f'| {v} | {vals[0]:.5f} | {vals[1]:.5f} | {means[v]:.5f} |')
lines+=['','Baseline de todos os braços: 0,69405. Cada endpoint usa mínimo dos quatro',
        'scores modo×cenário. Existem poucos episódios/casos; não confundir este',
        'critério com a média de todos os casos ou com o orçamento do treino principal.', '',
        '## Interpretação', '',
        'C2 supera C1 nos dois seeds deste POC, mas os endpoints permanecem abaixo',
        'do baseline e o orçamento é pequeno. C3 sem calibração tem resultados bem',
        'diferentes entre seeds: separar parâmetros elimina uma interferência direta',
        'demonstrável, porém não é garantia de melhora de score em qualquer seed.',
        'C4 verifica a calibração critic-only que integra o preset principal.',
        'Os números desse teste não justificam escolher hiperparâmetros retrospectivamente',
        'até conseguir um resultado favorável. A evidência de avanço do candidato',
        'principal é a validação em seeds inéditas registrada em HELDOUT.md.', '',
        'A conclusão empírica deve ser ampliada em orçamento maior e mais seeds de',
        'treinamento. O preset principal também contém calibração e está em outro',
        'orçamento. Não se mistura seu resultado com esta ablação como se fosse',
        'efeito causal isolado de uma das peças.']
(OUT/'ABLATION.md').write_text('\n'.join(lines)+'\n')
(OUT/'ablation_results.json').write_text(json.dumps(dict(results=results,means=means,samples_per_arm_seed=163840),indent=2))
print(means)
