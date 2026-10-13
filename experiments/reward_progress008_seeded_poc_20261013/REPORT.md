# POC: progresso 1,0 versus 0,08

Tabela padrão atual: gol ±25, aproximação 0,6, progresso assinado 0,08.

## Magnitude em cenários controlados

- Chute livre: +30,83 de progresso em 13 decisões, mais +25 pelo gol. Soma dessas duas parcelas: +55,83; aproximação não está incluída.
- Chute bloqueado: −0,53 de progresso em 100 decisões, sem gol.
- Velocidade projetada 3: ±2,02 por decisão.
- Ida/volta simétrica na mesma posição cancela sem desconto.

## Comparação de treino

30 iterações × 64 ambientes × 64 decisões por variante (122.880 transições), três seeds, mesmo checkpoint inicial. Seeds de reset explicitamente controladas. Fine-tuning contra Chase; avaliação greedy nos dois lados, seeds 10/11, 16 ambientes × 300 decisões por condição.

| Seed | Peso 1,0: gols pró/contra | Peso 0,08: gols pró/contra |
|---|---:|---:|
| 0 | 141/0 | 63/0 |
| 1 | 155/0 | 106/0 |
| 2 | 18/448 | 26/1 |

Os resultados agregados são dominados pelo adversário Random. São gols em janelas fixas, não taxa de vitória. A tabela com 0,08 não demonstrou melhora nesse orçamento; reduzir a magnitude não basta para concluir que o comportamento melhorou. Não é avaliação de convergência em self-play.

Os pesos do POC ficam nesta pasta, sem promover nenhum checkpoint para os modelos principais. As execuções anteriores não controlavam os RNGs dos resets e foram marcadas no JSON como exploratórias. Esta repetição usa o mesmo checkpoint salvo antes da correção.

## Resultados separados por adversário

Cada placar agrega os dois lados e as seeds de avaliação 10/11. São gols acumulados em janelas de 300 decisões, não placares de uma partida nem taxa de vitória.

### Chase com ruído

| Seed de treino | Progresso 1,0: pró × contra | Progresso 0,08: pró × contra |
|---|---:|---:|
| 0 | 4 × 0 | 3 × 0 |
| 1 | 3 × 0 | 0 × 0 |
| 2 | 0 × 0 | 0 × 1 |

### Chase sem ruído

| Seed de treino | Progresso 1,0: pró × contra | Progresso 0,08: pró × contra |
|---|---:|---:|
| 0 | 0 × 0 | 0 × 0 |
| 1 | 0 × 0 | 0 × 0 |
| 2 | 0 × 448 | 0 × 0 |

### Random

| Seed de treino | Progresso 1,0: pró × contra | Progresso 0,08: pró × contra |
|---|---:|---:|
| 0 | 137 × 0 | 60 × 0 |
| 1 | 152 × 0 | 106 × 0 |
| 2 | 18 × 0 | 26 × 0 |

O sinal de menor instabilidade com 0,08 vem principalmente da ausência da regressão contra Chase sem ruído observada na seed 2 com peso 1,0. Contra Chase com ruído não houve melhora; contra Random houve menos gols em duas seeds e mais em uma. Três seeds e muitos resultados 0 × 0 não estabelecem superioridade consistente. Seeds diferentes de avaliação não criam diversidade de cenário para Chase sem ruído com kickoff fixo e política greedy; suas repetições são idênticas.
