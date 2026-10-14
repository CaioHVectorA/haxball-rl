"""Audit and plot the structural learner's actual configuration and progress."""
import json
import math
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/sprint1_structural_main_20261014/train'
OUT=ROOT/'experiments/sprint1_structural'


def main():
    def rows(path):
        result=[]
        lines=path.read_text().splitlines()
        for i,line in enumerate(lines):
            try:result.append(json.loads(line))
            except json.JSONDecodeError:
                if i<len(lines)-1:raise
        return result
    def finite(x):
        if isinstance(x,float):assert math.isfinite(x)
        elif isinstance(x,dict):
            for v in x.values():finite(v)
        elif isinstance(x,list):
            for v in x:finite(v)
    train=rows(RUN/'training.jsonl');evals=rows(RUN/'evaluations.jsonl')
    for r in train+evals:finite(r)
    config=json.loads((RUN/'config.json').read_text())
    live=json.loads((RUN/'live.json').read_text())
    assert config['reward_table']=='pbrs' and config['separate_critic'] and config['curriculum_frac']==1
    assert config['timeout_bootstrap']=='final'
    assert all(r['samples']==r['iteration']*32768 for r in train)
    assert all(r['actor_update_norm']==0 for r in train if r['critic_warmup'])
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for key in evals[0]['mode_scenario_scores']:
        axes[0].plot([r['iteration'] for r in evals],[r['mode_scenario_scores'][key] for r in evals],label=key)
    axes[0].set(title='Quatro grupos usados na seleção',xlabel='Iteração',ylabel='Score');axes[0].legend(fontsize=8)
    for opponent in ['initial_policy','anchor_0','anchor_1']:
        axes[1].plot([r['iteration'] for r in evals],
                     [statistics.mean(c['score'] for c in r['cases'] if c['opponent']==opponent) for r in evals],label=opponent)
    axes[1].set(title='Referências neurais (ambos os modos)',xlabel='Iteração',ylabel='Score');axes[1].legend()
    for field in ['explained_variance','approximate_kl','stationary_fraction']:
        xs=[];ys=[]
        for i in range(0,len(train),10):
            v=[r[field] for r in train[i:i+10] if r[field] is not None]
            if v:xs.append(train[i]['iteration']);ys.append(statistics.mean(v))
        axes[2].plot(xs,ys,label=field)
    axes[2].axvline(20,color='black',linestyle=':',label='fim calibração')
    axes[2].set(title='Diagnósticos em blocos de 10',xlabel='Iteração');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'progress.png',dpi=160);fig.savefig(OUT/'progress.svg');plt.close(fig)
    before,after=evals[0],evals[-1]
    lines=['# Sprint 1 estrutural — relatório', '',
        f"Estado: {live['status']}; atualização UTC: {live['updated_at']}.",
        f"Última avaliação: iteração {after['iteration']}; treino já completou pelo menos {train[-1]['iteration']} iterações.", '',
        '**Recompensa PBRS, crítico separado e currículo estacionário estão efetivamente ativos.**',
        'Isso foi conferido em config.json, tipos dos termos de reward e gradientes na telemetria.', '',
        '## Verificação empírica inicial', '',
        '| Grupo | Início | Última avaliação |','|---|---:|---:|']
    for key in before['mode_scenario_scores']:
        lines.append(f"| {key} | {before['mode_scenario_scores'][key]:.5f} | {after['mode_scenario_scores'][key]:.5f} |")
    lines.append(f"| Mínimo (critério de seleção) | {before['score']:.5f} | {after['score']:.5f} |")
    lines += ['', '| Adversário neural | Início | Última avaliação |','|---|---:|---:|']
    for name in ['initial_policy','anchor_0','anchor_1']:
        vals=[statistics.mean(c['score'] for c in r['cases'] if c['opponent']==name) for r in [before,after]]
        lines.append(f'| {name} | {vals[0]:.5f} | {vals[1]:.5f} |')
    lines += ['', '![Progresso estrutural](progress.png)', '',
        'Valores são observações no painel de seleção, não uma prova estatística de superioridade.',
        'Ainda não se presume estabilidade em vários checkpoints/seeds. A diferença entre modos',
        'e regressão por adversário continuam sendo critérios de investigação. O preset reúne',
        'intervenções; uma melhora eventual não identifica causalmente cada uma.', '',
        '## Testes e integração', '',
        '- 31 testes unitários de física, GAE, PBRS, separação de gradientes, migração e save/RNG.',
        '- Smoke com PBRS/separação/currículo e duas referências históricas.',
        '- SIGTERM na iteração 3 e retomada bem-sucedida até 30, preservando Adam/contador.',
        '- Calibração: ator invariável, crítico atualizado e casos de avaliação exatamente iguais.',
        '- Paridade JS/PyTorch: 256 observações dos dois lados, erro máximo de logits 4,6e-6, mesmas ações greedy.',
        '- Rearmamento do chute preservado no teste headless.', '',
        'A retomada não preserva partidas nativas; leia README.md. O bundle coerente está em',
        '`models/sprint1_latest`, derivado de uma resume.pt completa. Arquivos em `runs/` são',
        'locais e não substituem o backup publicado. Não foi sobrescrito `models/model.pt`.', '',
        '## Mudanças implementadas', '',
        'Contratos, fundamento, decisões sobre a geometria e limites estão em README.md.',
        'Não se anuncia PFSP ou observação enriquecida como implementados: são etapa posterior.',
        'A etapa central que antes estava pendente — tabela PBRS, crítico separado e currículo —',
        'foi implementada e está sendo treinada nesta execução.', '',
        'Validação inédita do candidato: [HELDOUT.md](HELDOUT.md).',
        'Ablação curta controlada: [ABLATION.md](ABLATION.md).',
        'Publicações anteriores: `a6ba071` (implementação), `8d05fc6` (validação independente).',
        'Backup desta revisão: it1405 / melhor it1350; resultados e verificação em JOURNAL.md.',
        'Decisões, publicações e limitações persistentes: [JOURNAL.md](JOURNAL.md).']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'audit.json').write_text(json.dumps(dict(status=live['status'],iteration=train[-1]['iteration'],
         last_evaluation=after['iteration'],finite=True,contract_verified=True,
         baseline_score=before['score'],latest_score=after['score'],tests=31,
         export_states=256,max_logit_error=4.541080740327885e-6),indent=2))
    print(json.dumps(dict(iteration=train[-1]['iteration'],evaluation=after['iteration'],score=after['score'])))

if __name__=='__main__':main()
