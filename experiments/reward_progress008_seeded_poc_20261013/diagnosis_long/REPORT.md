# Diagnóstico da falha 0 × 448

## Resultado reproduzido

Um ambiente, política greedy do checkpoint `simple_progress1_seed2.pt`, Chase sem ruído, kickoff fixo. Em 300 decisões foram sete gols sofridos, nas decisões 39/79/119/159/199/239/279. Isso reproduz 7 × 16 ambientes × 2 lados × 2 seeds de avaliação = 448 gols. As repetições determinísticas não são 64 trajetórias independentes.

## Primeira jogada

- Decisões 0–9: os dois bots aproximam-se da bola pelo eixo central.
- Decisão 10: a política escolhe recuar ainda a 43,5 unidades da bola e não pressiona chute; Chase continua avançando.
- Decisão 11: a bola começa a ir para o gol vermelho.
- Decisões 12–27: a política mantém-se perto, recuando/neutra, sem pressionar chute.
- Decisão 28: começa a pressionar chute, com a bola já em x≈−292. O avanço adversário continua.
- Decisão 39: bola em (−409,05; 59,51), evento de gol sofrido. O estado terminal foi capturado antes do autoreset.
- Novo kickoff: mesmas posições e decisões greedy reproduzem a sequência, um gol a cada 40 decisões (5,33 segundos).

Isso não é um gol contra claramente atribuível a um chute do learner: Chase empurra a bola, e o learner recua e não consegue afastá-la. A evidência estabelece a sequência observada, não a causa de aprendizado da escolha de recuar.

## Comparação e intervenções

Em 1.500 decisões por caso:

| Caso | Gols pró × contra |
|---|---:|
| failure_greedy | 0 × 37 |
| low_greedy | 0 × 0 |
| failure_sampled | 0 × 0 |
| failure_noise | 0 × 0 |
| failure_offset | 0 × 31 |
| failure_random | 0 × 3 |
| low_sampled | 0 × 1 |
| low_noise | 0 × 1 |
| low_random | 0 × 0 |
| failure_force_kick | 0 × 16 |

Na mesma política de peso 1,0, ações amostradas ou ruído no Chase quebram a derrota periódica; porém frequentemente levam a bola aos cantos. Forçar chute no alcance reduz as derrotas (37 → 16), sem resolver o problema. Uma perturbação de 10 unidades no Y atrasa o primeiro gol, mas o ciclo aparece depois. São intervenções de uma única seed, não estimativas de robustez.

## O que o 0 × 0 escondia

O checkpoint de peso 0,08 também recua na primeira disputa. Nas decisões 24–26 passa a deslocar-se lateralmente e chuta, desviando a bola para a lateral. Depois a bola atinge o canto externo (474,2; −224,2) e fica parada. Na decisão 200, o learner está a 38,88 unidades da bola e Chase a 20,13. O learner está fora do alcance do chute; Chase está próximo. Uma intervenção que alterna soltar/pressionar o chute azul a partir da decisão 60 também não produz gols em 1.500 decisões e mantém a bola em cantos. Isso enfraquece a hipótese de que o travamento dependa apenas de manter o chute pressionado. O mecanismo exato precisa de auditoria de colisões, geometria e direção dos chutes.

O timeout ocorre nas decisões 249/499/749/... porque `TimeoutCondition(2000)` mede ticks e tick_skip é 8. A política repete a mesma trajetória após cada reset. Os resultados sem gols são, portanto, compatíveis com bola parada no canto, não com defesa consolidada.

## Recompensas e escala

Na primeira derrota do peso 1,0, aproximação soma −149,54, progresso −423,93 e gol −25 (somas sem desconto). Portanto a trajetória perdedora não rende recompensa líquida positiva; não é evidência de reward hacking positivo nesse caso.

No teste longo do peso 0,08, aproximação soma +3331,34 e progresso ponderado +180,34, sem gols. Mesmo com bola parada, o learner recebe cerca de +0,363 por decisão por projetar uma pequena velocidade na direção da bola, embora as posições estejam praticamente fixas. A normalização é importante: o ambiente informa player_max_speed≈0,119, mas foram observadas velocidades de movimento acima de 2. Isso deve ser verificado antes de calibrar pesos com confiança.

## Próximos passos concretos

1. Auditar o canto: distância de chute, latch de chute (soltar/pressionar), paredes e geometria do estádio.
2. Verificar a normalização de velocidade e considerar medir progresso por deslocamento real para não recompensar velocidade reportada sem movimento útil.
3. Avaliar episódios completos com trajetórias variadas, gols e tempo de bola parada, distinguindo greedy de amostrado.
4. Repetir comparação de recompensas após esses controles. Não promover o 0,08 com base apenas no desaparecimento da derrota repetitiva.

Reprodução: `uv run src/diagnose_poc_game.py --out experiments/nova_analise --steps 1500`. Nenhum peso principal foi modificado.
