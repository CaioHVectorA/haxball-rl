# Sprint 1 estrutural — relatório

Estado: running; atualização UTC: 2026-10-14T12:12:01.770990+00:00.
Última avaliação: iteração 250; treino já completou pelo menos 283 iterações.

**Recompensa PBRS, crítico separado e currículo estacionário estão efetivamente ativos.**
Isso foi conferido em config.json, tipos dos termos de reward e gradientes na telemetria.

## Verificação empírica inicial

| Grupo | Início | Última avaliação |
|---|---:|---:|
| greedy_random | 0.71734 | 0.80810 |
| greedy_kickoff | 0.83606 | 0.91589 |
| sampled_random | 0.72166 | 0.79734 |
| sampled_kickoff | 0.76868 | 0.86592 |
| Mínimo (critério de seleção) | 0.71734 | 0.79734 |

| Adversário neural | Início | Última avaliação |
|---|---:|---:|
| initial_policy | 0.50616 | 0.50498 |
| anchor_0 | 0.67774 | 0.92407 |
| anchor_1 | 0.79257 | 0.84184 |

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
