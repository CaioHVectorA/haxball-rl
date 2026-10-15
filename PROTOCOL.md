# Protocolo de treinamento e avaliação

Valores conferidos em `src/train.py`. Este documento distingue implementação atual de experimentos propostos.

## Correções na branch gpt

O treino usa por padrão `maps/futsal_1v1.hbs`. Removemos o bias das paredes do mapa original após reproduzir escape da bola para cantos externos; testes verificam rebote nas duas laterais. Isso altera o estádio: uma política exportada deve carregar o mesmo mapa. A equivalência com física oficial não foi certificada.

A tabela mantém os pesos desejados (gol ±25, aproximação 0,6, progresso 0,08), mas usa deslocamento efetivo:

- `MotionPlayerToBall`: projeta o deslocamento do jogador na direção anterior da bola. Jogador preso sem mudar de posição recebe zero, mesmo que o engine reporte velocidade.
- `MotionBallToGoal`: redução assinada da distância da bola ao segmento finito do gol adversário. Um percurso de ida/volta fechado cancela sem desconto.
- Ambos dividem pelo deslocamento livre máximo esperado em uma decisão: acceleration × damping / (1 − damping) × tick_skip. No mapa atual, 21,12 unidades por decisão, em vez do player_max_speed ≈0,119 informado pela dependência.

A ação de pressionar chute agora rearma repetidamente no simulador. A exportação inclui `repeat_kick`; bot.js executa a mesma intenção entre decisões quando esse campo é true. Checkpoints antigos sem esse campo preservam seu comportamento anterior na sala.

`src/repair_bot.py` faz fine-tuning contra Chase e avaliação nos dois lados, com kickoff e resets aleatórios, ações greedy/amostradas, múltiplas seeds de avaliação, vitórias/empates/derrotas por episódio e fração de decisões com bola parada. O treino principal continua com PPO self-play; este experimento curto não comprova convergência. As seeds de seleção não substituem validação independente.

As tabelas `simple`, `simple_progress1` e `previous` permanecem para reproduzir recompensas anteriores. São sinais diferentes da tabela corrigida: pesos iguais não implicam magnitude igual.

## PPO atual

| Parâmetro | Valor padrão |
|---|---|
| Ambientes / jogadores | 512 / 1v1 |
| Rollout / épocas / minibatches | 64 / 6 / 4 |
| Amostras por iteração em 1v1 | 32.768 |
| Tick skip | 8 (~7,5 decisões/s a 60 ticks/s) |
| Discount | meia-vida de 12 s; gamma ≈ 0,99233 |
| GAE lambda / clip | 0,95 / 0,20 |
| Learning rate | 3e−4 → 1e−4, linear |
| Entropia | 0,02 → 0,01, linear |
| Coeficiente de valor / norma máxima de gradiente | 0,5 / 0,5 |
| Limite de episódio | 2.000 ticks = 250 decisões com tick skip 8 |
| Resets de treino | 40% kickoff, 60% estado aleatório (`ball_speed=3`) |

A perda de valor já inclui fator 0,5 antes de receber `VF_COEF`. O GAE mantém o controle legado sem bootstrap com `--timeout-bootstrap zero` (padrão). A variante `--timeout-bootstrap final` usa V da observação anterior ao autoreset em timeouts artificiais; gols zeram o bootstrap e ambas as fronteiras interrompem o traço. O protocolo B0/E1 e a instrumentação estão em `experiments/sprint1_timeout/README.md`. `--warm-start` reinicia otimizador, schedules e contador. Por padrão, carrega a política e zera a cabeça de valor para adaptar a escala corrigida de recompensa. `--keep-critic` preserva a cabeça de valor quando apropriado. Essa reinicialização não altera os logits iniciais de ação.

## Tabela atual de recompensas

Antes da branch gpt, a tabela padrão era `build_reward("simple")`: gol ±25, aproximação com peso 0,6 e progresso com peso 0,08. Agora o padrão é `corrected`, descrito acima. Os dois termos de velocidade preservam sinais positivos e negativos. Treino e avaliação interna compartilham essa fábrica.

A tabela anterior permanece disponível como `build_reward("previous")` para comparação controlada:

### Tabela anterior

Treino e avaliação interna usam a mesma composição. O multiplicador interno se combina com o peso de `CombinedReward`:

| Termo | Peso externo | Fator interno | Contribuição efetiva |
|---|---:|---:|---|
| `VelocityPlayerToBall` | 0,7 | — | 0,7 × velocidade projetada, normalizada por `player_max_speed` |
| `VelocityBallToGoal` | 1,0 | — | velocidade projetada para a linha do gol adversário / `player_max_speed` |
| `TangentialGoalShotReward` | 3,0 | 3,0 | 9 × deslocamento da bola por decisão, quando os filtros passam |
| `DeltaBreakawayReward` | 2,0 | 2,0 | 4 × max(phi atual − phi anterior, 0) |
| `PenaltyTackleLostReward` | 1,0 | −0,5 | −0,5 quando o evento definido na função ocorre |
| `GoalReward` | 20,0 | ±1 | +20 ao marcar, −20 ao sofrer |

O chute usa deslocamento, não velocidade normalizada: sua escala depende do tick skip. O filtro aceita trajetória projetada na boca do gol **ou** bola já na faixa |y| ≤ 65, e usa proximidade <36 como contato. Não identifica necessariamente um chute efetivo.

A ultrapassagem define posse como ser o jogador mais próximo da bola, sem distância máxima. Só recompensa aumentos da vantagem: perder e recuperar vantagem pode gerar pagamentos repetidos. Não é shaping baseado em potencial com preservação garantida do objetivo.

Não está demonstrado que o gol domine a soma dos termos densos. Medir contribuições acumuladas por episódio e sua relação com gols antes de ajustar pesos. `ShotVelocityToGoal` está definido no arquivo, mas não participa da composição atual.

### Experimento proposto: simplificar

Comparar a tabela atual com uma base de três termos: gol, aproximação do jogador e progresso da bola. Remover inicialmente chute tangencial, ultrapassagem e penalidade de desarme na variante experimental. Calibrar os dois pesos densos usando suas contribuições acumuladas e comparar com orçamento igual de amostras e várias seeds. A variante de três termos foi aplicada ao treinamento. O POC compara as duas variantes.

## Avaliação implementada

- `eval_vs_chase`: a cada 50 iterações e no fim, 128 ambientes × 350 decisões, learner vermelho com ações amostradas, Chase azul com seed 0 e resets de kickoff. Saldo de gols seleciona `model.pt`.
- `src/eval.py`: padrão de 256 ambientes × 600 decisões, NeuralBot greedy; benchmark contra Chase e Random. O indicador chamado “winrate estimada” é proporção dos gols, não taxa de vitória por partida.
- `src/select_checkpoint.py`: torneio de checkpoints em `models/ckpts/` e modelo final; cada par joga nos dois lados, com ações amostradas. Ranking por proporção de gols, desempate por saldo. Não usa TrueSkill nem matriz de vitórias. Promove o vencedor sobrescrevendo `model.pt`; metadados herdados do modelo final podem não descrever a iteração do vencedor.

Esses caminhos usam configurações de ambiente distintas: treino/avaliação interna criam o ambiente explicitamente; benchmark/torneio usam `make_default_env`. Alinhar condições para comparações reproduzíveis.

## Avaliação planejada para consolidar o 1v1

1. Fixar configuração e registrar seeds de treino, cenário e ações; começar com pelo menos três seeds de treino e cinco de avaliação.
2. Avaliar nos dois lados contra Random, Chase sem ruído, Chase com ruído, um defensor e um bot com interceptação, além de snapshots históricos. Os dois últimos bots ainda precisam ser implementados.
3. Reportar gols pró/contra, saldo, vitórias/empates/derrotas por partida, duração e dispersão entre seeds. Identificar explicitamente resultados greedy e amostrados.
4. Comparar variantes de recompensa com orçamento igual; reservar adversários/cenários para validação final.
5. Inspecionar replays e partidas reais para confirmar finalização, defesa e ausência de exploração da recompensa.

Definir duração e limiar de aprovação antes dos experimentos. Um único saldo contra Chase não estabelece consolidação.

## POC curto reproduzível

```bash
uv run src/poc_reward.py --out experiments/minha_comparacao --iters 30 --envs 64 --seeds 0 1 2 --eval-seeds 10 11
```

O POC usa o mesmo checkpoint inicial, seed e orçamento para cada variante; faz fine-tuning contra Chase (não self-play) e avaliação greedy nos dois lados contra Chase com/sem ruído e Random. Grava configuração, hash do checkpoint, contribuições dos termos, cenários controlados e resultados em pasta nova. Não sobrescreve os modelos principais. Os cenários incluem sinais positivos/negativos, gols e trajetórias no simulador com defensor dentro/fora da linha do chute.

Trinta iterações servem para detectar regressões rápidas; não demonstram convergência nem permitem declarar uma tabela vencedora definitiva.

### Alerta encontrado no POC

No ambiente instalado, `futsal-classic` fornece `player_max_speed ≈ 0,11865`. A função de progresso divide por esse valor: um chute livre controlado acumulou aproximadamente 385 de shaping em 13 decisões, além do gol +25. A tabela simples ainda precisa de verificação/correção da normalização antes de treinos longos. Consulte `experiments/reward_simple_poc_20261013/REPORT.md`; o POC curto não demonstrou melhoria da variante simples.

### Variante com progresso 0,08

A tabela padrão passa a gol ±25, aproximação 0,6 e progresso 0,08. `simple_progress1` preserva a variante de progresso 1,0 para comparação pareada no POC.

```bash
uv run src/poc_reward.py --out experiments/comparacao_008 --tables simple_progress1 simple
```

Na trajetória controlada de chute livre, a contribuição ponderada de progresso cai de +385,33 para +30,83. Com defensor na linha, cai de −6,60 para −0,53. Essas são somas sem desconto só do termo de progresso; aproximação e gol são parcelas separadas. Uma decisão com velocidade projetada 3 gera aproximadamente ±2,02 com peso 0,08.

Os geradores de reset de treino agora recebem `--seed` explicitamente. A comparação anterior não controlava esses geradores; a repetição corrigida está em `experiments/reward_progress008_seeded_poc_20261013/`.

Diagnóstico posterior: o saldo 0 × 448 corresponde à mesma derrota de kickoff repetida deterministicamente. O checkpoint 0,08 evita essa sequência, mas leva a bola ao canto e termina por timeout, sem consolidar ataque ou defesa. Veja `experiments/reward_progress008_seeded_poc_20261013/diagnosis_long/REPORT.md`.

## Telemetria do self-play monitorado

O treinador grava `training.jsonl` (uma linha por iteração), `evaluation_cases.jsonl` (cada caso), `evaluations.jsonl` (painel completo) e `live.json` (heartbeat/estado). Configuração, revisão Git e hash do warm-start ficam em `config.json`. `initial_policy.pt` congela o adversário neural de referência.

Avaliação padrão: 16 ambientes × 400 decisões, seeds 1701/1702, ambos os lados, kickoff e estados aleatórios, cinco adversários, learner greedy. Cada caso recebe peso igual na pontuação de seleção. Na ausência de episódios completos a pontuação é desconhecida, não zero. Avaliação inicial e a cada 50 iterações; snapshots de self-play continuam a cada 20 iterações. A avaliação preserva o estado RNG do PyTorch do treino.

Métricas PPO: perdas de política/valor, entropia somada das três cabeças, KL aproximada, fração de clipping e variância explicada calculada com valores do rollout. Métricas de jogo: gols, episódios completos, bola parada (deslocamento <0,01 por decisão), contato e pressão de chute dentro de 24,8 unidades. A frequência de bola parada inclui kickoffs normais e não mede sozinha travamento. A pontuação não classifica quem deu o último toque de cada gol.

Dashboard: stream de eventos a cada segundo, histórico da execução selecionada, cenário selecionável e linhas por adversário. Detecta perda de conexão, interrupção de processo e heartbeat atrasado. O servidor local é somente leitura e não inicia POCs.

## Comparação controlada após 1.000 iterações

`src/reward_sweep.py` executa seis treinos de self-play: aproximação 0,60 / 0,15 / 0,05, seeds 0 e 1, 300 iterações × 256 ambientes × 64 decisões por treino (4.915.200 amostras). Todos partem do mesmo checkpoint it650, mantêm gol ±25 e progresso 0,08 e reiniciam o critic. A ordem dos pesos é invertida na segunda seed. O runtime de simulação usa quatro threads e PyTorch duas para compartilhar CPU com a sala real.

Cada execução mantém a mesma avaliação por adversário e grava inspeção de comportamento do checkpoint final. Comparar endpoints e cenários aleatórios; registrar melhor checkpoint separadamente. Os seis treinos são uma triagem, não prova de convergência robusta nem extrapolação para 20 mil iterações. O dashboard acompanha a execução ativa; os históricos anteriores permanecem em suas pastas.

A inspeção registra distância da bola, intenção de chute durante proximidade, alinhamento aproximado com o gol, posição defensiva e períodos com bola parada. Essas métricas não identificam com certeza chutes efetivos nem autoria de gols. As parcelas absolutas de recompensa passam a ser gravadas para evitar confundir cancelamento do saldo de gols com ausência de sinal de gol.

## Preset estrutural Sprint 1

`--sprint1` implementa recompensa PBRS, crítico separado, clipping independente,
calibração inicial do crítico, curriculum estacionário com âncoras neurais, avaliação
greedy/amostrada e checkpoints retomáveis. O modo legado permanece disponível para
comparações. A descrição completa, justificativas, limites e comandos estão em
`experiments/sprint1_structural/README.md`; o relatório empírico em `REPORT.md`.
Não interpretar o novo score mínimo modo×cenário como o antigo score médio greedy.
