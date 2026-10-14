# Continuação da madrugada — acompanhamento

Estado: completed; atualizado em 2026-10-14T09:58:13.672357+00:00 (UTC).

Última iteração registrada: 5236 — 171,573,248 amostras.
Melhor checkpoint selecionado: iteração 3650, score 0.87147.

| Medida | Início | Última avaliação |
|---|---:|---:|
| score | 0.66739 | 0.70907 |
| random | 0.67575 | 0.74696 |
| kickoff | 0.65902 | 0.67118 |
| heuristic_random | 0.71969 | 0.75517 |
| heuristic_kickoff | 0.69877 | 0.77648 |

Última avaliação: iteração 5236.

![Curvas da continuação](continuation_progress.png)

Métricas finitas e contadores de amostras contínuos foram verificados. A referência
initial_policy é o checkpoint B0 de origem. O score balanceado anterior usa outra
referência; a comparação com ele exige avaliação alinhada. O recorte heurístico
mantém adversários comparáveis. Mudanças greedy oscilam entre checkpoints;
mais iterações não garantem melhora monotônica. O melhor checkpoint é selecionado
no próprio painel e precisa de avaliação posterior. Este acompanhamento não altera
o experimento nem promove qualquer modelo.

Endpoint salvo e metadados conferidos: iteração 5236.
