"""Consolidate unseen-seed evaluation without changing or promoting checkpoints."""
import hashlib
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
VALID=ROOT/'runs/sprint1_overnight_final_validation_20261014'
OUT=ROOT/'experiments/sprint1_overnight'
RUN=ROOT/'runs/sprint1_overnight_model_20261014'

def main():
    comparison=json.loads((VALID/'comparison.json').read_text())
    results=comparison['results']
    cases=[json.loads(x) for x in (VALID/'cases.jsonl').read_text().splitlines()]
    assert len(results)==6 and len(cases)==600
    keys={(c['candidate'],c['policy_mode'],c['scenario'],c['opponent'],c['seed'],c['side']) for c in cases}
    assert len(keys)==600
    assert all(c['completed_episodes']>0 and 0<=c['score']<=1 for c in cases)
    per_case=[]
    for candidate in ['best_it3650','final_it5236']:
        for mode in ['greedy','sampled']:
            for scenario in ['random','kickoff']:
                for opponent in ['chase','chase_no_noise','striker','keeper','initial_policy']:
                    for side in [0,1]:
                        def score(name):
                            return statistics.mean(c['score'] for c in cases if c['candidate']==name and c['policy_mode']==mode
                                   and c['scenario']==scenario and c['opponent']==opponent and c['side']==side)
                        per_case.append(dict(candidate=candidate,mode=mode,scenario=scenario,opponent=opponent,side=side,
                                             score=score(candidate),delta_vs_origin=score(candidate)-score('origin')))
    (OUT/'final_case_comparison.json').write_text(json.dumps(per_case,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    names=['origin','best_it3650','final_it5236']
    for ax,field in zip(axes,['score','random','kickoff']):
        for offset,mode,color in [(-.18,'greedy','#3066BE'),(.18,'sampled','#D1495B')]:
            vals=[next(r[field] for r in results if r['candidate']==n and r['mode']==mode) for n in names]
            bars=ax.bar([i+offset for i in range(3)],vals,width=.35,color=color,label=mode)
            ax.bar_label(bars,fmt='%.3f',fontsize=8)
        ax.set_xticks(range(3),['Origem','Melhor (3650)','Final (5236)'])
        ax.set(title=field,ylim=(0,1),ylabel='Score');ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('Avaliação final: cinco seeds inéditas, referência comum, sem atualizar pesos')
    fig.savefig(OUT/'final_validation.png',dpi=160);fig.savefig(OUT/'final_validation.svg');plt.close(fig)
    lines=['# Resultado final da madrugada', '',
           'Treino encerrado normalmente às **06h58**, com **5.236 novas iterações** e',
           '**171.573.248 novas amostras**. Nenhum processo de treino permanece ativo.',
           'A automação foi pausada após o horário limite; os modelos principais não foram substituídos.', '',
           '## Avaliação independente do painel de seleção', '',
           'Seeds 3701–3705, 8 ambientes, 600 decisões por caso, referência neural comum.',
           'Ambos os lados, kickoff/random, cinco adversários e modos greedy/amostrado.',
           '600 casos completos e únicos foram verificados. Esta avaliação não atualizou pesos.', '',
           '| Checkpoint | Modo | Balanceado | Random | Kickoff |','|---|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['candidate']} | {r['mode']} | {r['score']:.5f} | {r['random']:.5f} | {r['kickoff']:.5f} |")
    lines += ['', '![Validação final](final_validation.png)', '',
              '## Leitura do resultado', '',
              'O maior score interno (0,87147 na iteração 3.650) veio de um painel usado',
              'repetidamente para selecionar checkpoints. A tabela nova é a evidência mais',
              'apropriada para comparar origem, melhor checkpoint e endpoint nas mesmas condições.',
              'O score final interno foi 0,70907; portanto usar somente a última iteração',
              'descartaria o checkpoint intermediário mais forte no painel de seleção.', '',
              'O aprendizado não foi monotônico: random ficou mais estável, enquanto kickoff',
              'oscilou fortemente. A correção de bootstrap E1 não demonstrou vantagem geral',
              'nas três seeds da ablação; a continuação preservou B0. Não foram implementados',
              'PBRS, crítico separado ou observações enriquecidas nesta etapa.', '',
              'Seeds de avaliação não são réplicas independentes de treinamento. O teste',
              'novo avalia esses checkpoints de uma única continuação; não certifica robustez',
              'entre seeds de treinamento, transferência para sala real ou desempenho em equipes.',
              'O resultado permite inspecionar um candidato; não garante convergência nem',
              'autoriza promoção automática. Diferenças por adversário/lado estão em',
              '`final_case_comparison.json`; curvas e fundamentos em `REPORT.md` e `CONTINUATION.md`.', '',
              '## Checkpoints preservados', '',
              '- Melhor painel interno: `runs/sprint1_overnight_model_20261014/train/model.pt`, iteração 3.650.',
              '- Endpoint: `runs/sprint1_overnight_model_20261014/train/model_final.pt`, iteração 5.236.',
              '- Metadados correspondentes: `meta.json` e `meta_final.json` na mesma pasta.', '',
              'Pesos são finitos e os metadados finais correspondem ao contador real.']
    (OUT/'FINAL.md').write_text('\n'.join(lines)+'\n')
    audit=json.loads((OUT/'audit.json').read_text())
    audit['final_validation']=dict(cases=600,seeds=[3701,3702,3703,3704,3705],modes=['greedy','sampled'],
        unique_cases=True,no_empty_scores=True,results=results)
    audit['automation_status']='PAUSED'
    audit['models_main_unchanged']=hashlib.sha256((ROOT/'models/model.pt').read_bytes()).hexdigest()==json.loads(
        (ROOT/'runs/sprint1_timeout_full_20261014/manifest.json').read_text())['checkpoint_sha256']
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    p=OUT/'REPORT.md';s=p.read_text()
    if '## Avaliação final consolidada' not in s:
        s+='\n## Avaliação final consolidada\n\nResultados em seeds inéditas, gráficos e checkpoints: [FINAL.md](FINAL.md).\nA automação está pausada e o orçamento da madrugada foi encerrado.\n'
        p.write_text(s)

if __name__=='__main__':main()
