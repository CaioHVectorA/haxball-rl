# Sprint 1 estrutural — relatório

Estado: running; atualização UTC: 2026-10-14T12:46:56.197684+00:00.
Última avaliação: iteração 650; treino já completou pelo menos 650 iterações.

**Recompensa PBRS, crítico separado e currículo estacionário estão efetivamente ativos.**
Isso foi conferido em config.json, tipos dos termos de reward e gradientes na telemetria.

## Verificação empírica inicial

| Grupo | Início | Última avaliação |
|---|---:|---:|
| greedy_random | 0.71734 | 0.84495 |
| greedy_kickoff | 0.83606 | 0.70588 |
| sampled_random | 0.72166 | 0.82922 |
| sampled_kickoff | 0.76868 | 0.88831 |
| Mínimo (critério de seleção) | 0.71734 | 0.70588 |

| Adversário neural | Início | Última avaliação |
|---|---:|---:|
| initial_policy | 0.50616 | 0.42422 |
| anchor_0 | 0.67774 | 0.93982 |
| anchor_1 | 0.79257 | 0.62916 |

![Progresso estrutural](progress.png)

Valores são observações no painel de seleção, não uma prova estatística de superioridade.
Ainda não se presume estabilidade em vários checkpoints/seeds. A diferença entre modos
e regressão por adversário continuam sendo critérios de investigação. O preset reúne
intervenções; uma melhora eventual não identifica causalmente cada uma.

## Testes e integração

- 31 testes unitários de física, GAE, PBRS, separação de gradientes, migração e save/RNG.
- Smoke com PBRS/separação/currículo e duas referências históricas.
- SIGTERM na iteração 3 e retomada bem-sucedida até 30, preservando Adam/contador.
- Calibração: ator invariável, crítico atualizado e casos de avaliação exatamente iguais.
- Paridade JS/PyTorch: 256 observações dos dois lados, erro máximo de logits 4,6e-6, mesmas ações greedy.
- Rearmamento do chute preservado no teste headless.

A retomada não preserva partidas nativas; leia README.md. O bundle coerente está em
`models/sprint1_latest`, derivado de uma resume.pt completa. Arquivos em `runs/` são
locais e não substituem o backup publicado. Não foi sobrescrito `models/model.pt`.

## Mudanças implementadas

Contratos, fundamento, decisões sobre a geometria e limites estão em README.md.
Não se anuncia PFSP ou observação enriquecida como implementados: são etapa posterior.
A etapa central que antes estava pendente — tabela PBRS, crítico separado e currículo —
foi implementada e está sendo treinada nesta execução.

## Verificação independente e progresso

[HELDOUT.md](HELDOUT.md) registra comparação em três seeds inéditas.
O checkpoint exato da iteração 450 fica separado em `models/sprint1_validated`.
Não confundir esse arquivo com o bundle de retomada e modelo mais recente.

A trajetória posterior ainda apresentou oscilação de kickoff greedy: na iteração
650 o mínimo caiu a 0,70588, com sampled/kickoff 0,88831; na 700 o mínimo voltou
a 0,83540, com greedy/kickoff 0,99097. Portanto não se declara estabilidade
monotônica de cada checkpoint. O critério mínimo preservou o melhor modelo,
e o candidato validado permanece imutável; isso evita promover automaticamente
o endpoint que perdeu desempenho, sem ocultar a oscilação na análise.
