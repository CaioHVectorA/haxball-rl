# POC de recompensa: resultado

Comparação pareada de fine-tuning: checkpoint `models/model.pt` (hash em results.json), 30 iterações × 64 ambientes × 64 decisões por variante, seeds de treino 0/1/2. Avaliação greedy com seeds 10/11, 16 ambientes × 300 decisões, nos dois lados contra Chase, Chase sem ruído e Random. Não é um teste de convergência em self-play.

| Variante | Saldo seed 0 | Saldo seed 1 | Saldo seed 2 |
|---|---:|---:|---:|
| Anterior | +172 | +146 | +142 |
| Simples: ±25 / 0,6 / 1,0 | +154 | +104 | +145 |

Baseline: +172. Os saldos agregados são dominados pelo adversário Random. Contra Chase sem ruído não houve gols em nenhuma variante; a janela curta e os resets de kickoff limitaram o poder de comparação. A variante simples não demonstrou melhoria neste orçamento.

## Cenários controlados

A função de progresso preserva o sinal: velocidades opostas em posição idêntica cancelam sem desconto. Gol e gol sofrido produzem ±25 quando as velocidades são zero.

No simulador, com atacante a 24 unidades da bola, um chute livre marcou em 13 decisões e acumulou +385,33 de progresso. Com defensor na linha, houve rebote e saldo −6,60 de progresso em 100 decisões. Esses cenários isolam trajetórias; não são partidas entre políticas.

A primeira montagem usou distância 25, fora do alcance de chute neste mapa. Foi corrigida para 24; os cenários foram reexecutados e seus resultados atualizados no JSON após o treino. As execuções de treinamento não dependiam desses cenários.

## Problema de escala encontrado

O ambiente instalado fornece `player_max_speed=0.11865168539325842` no mapa futsal-classic. `VelocityBallToGoal` divide por esse número: velocidade 3 resulta em aproximadamente 25,28 de recompensa por decisão com peso 1. O progresso acumulado pode ultrapassar muito o gol ±25. Verificar o cálculo da velocidade máxima na dependência e a física do mapa, ou definir uma normalização explícita validada, antes de treino longo. O POC preservou a função instalada para medir exatamente a tabela proposta.

## Próximo experimento

Corrigir/verificar normalização; medir contribuições por episódio, prolongar avaliação contra Chase e incluir resets aleatórios pareados. Repetir várias seeds e depois validar em self-play. Nenhum modelo principal foi sobrescrito pelo POC.

## Correção posterior de seeds

Os geradores próprios de MultiSetter e RandomStateMutator não recebiam as seeds nesta execução. Portanto, o pareamento não controlava os resets de treino. A comparação foi corrigida posteriormente; os resultados acima são exploratórios.
