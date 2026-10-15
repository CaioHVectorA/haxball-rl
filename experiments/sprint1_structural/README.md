# Sprint 1 estrutural — mudanças reais, 14/10/2026

Esta etapa **muda a recompensa, a arquitetura de treinamento e os adversários**.
Não é outra continuação do B0. `--sprint1` é o preset explícito; chamar `train.py`
sem esse preset preserva a configuração legada por compatibilidade.

## Tabela de recompensa

| Termo | Implementação nova | Peso |
|---|---|---:|
| Gol | `GoalReward` | ±25 |
| Posição de chute | `DiscountedPocketPotential` | 0,15 |
| Progresso da bola | `DiscountedBallGoalPotential` | 0,08 |

`F = gamma * (1-terminated) * Phi(next) - Phi(previous)`.
Os pesos numéricos foram preservados, **as funções mudaram**. O gol é terminal
com potencial absorvente zero. Timeout é corte artificial: mantém Phi(next), usa
V da observação antes do reset e interrompe o traço GAE. PBRS com timeout-bootstrap
zero é rejeitado pelo CLI, pois não seria a semântica de continuação escolhida.

Phi de posição é a distância negativa, limitada, a um ponto 22 unidades atrás
da bola na direção do gol, com platô de raio 2. Isso coloca o alvo na faixa
fisicamente alcançável: contato ≈20,8, alcance do chute ≈24,8. O platô de raio 24
do documento deixaria muitos estados sem gradiente de aproximação longe da faixa
de chute. Phi da bola usa `(distância ao gol próprio - distância ao gol atacado)/2`,
dividida pela mesma escala física (21,12 por decisão no mapa atual).

A escolha do potencial dá uma orientação de posicionamento; o objetivo de referência
é o retorno descontado esparso de gols, sob as condições abaixo. A invariância é em
relação a esse objetivo esparso, não à tabela `corrected` antiga, que tinha sinais
densos diferentes e não era PBRS descontado.
Não há penalidade adicional de pressão de chute: ela alteraria esse objetivo,
enquanto a física já reduz aceleração durante o chute. Não foi copiada uma função
`KickDisciplineReward` que, no texto anexado, era anunciada sem implementação ou
contrato de acesso às ações. Esta decisão é explícita, não um termo esquecido.

## Fundamentação

Para uma trajetória, a soma descontada de F telescopa:
`sum(gamma^t * F_t) = -Phi(s0) + gamma^H * Phi(sH)`.
Em gol, Phi(sH)=0. Para horizonte indefinido com gamma<1 e potencial limitado,
o segundo termo tende a zero. O deslocamento restante depende do estado inicial,
não da ação escolhida. Essa é a estrutura de invariância estudada por
[Ng, Harada e Russell (1999)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

Um potencial negativo pode produzir recompensa positiva mesmo com estado parado:
`F=(gamma-1)*Phi`. Isso não é, por si só, um bug ou prova de farming: importam o
retorno descontado completo, os potenciais terminais e o bootstrap. Aproximação do
crítico, observações incompletas e horizontes de coleta podem prejudicar a prática;
a identidade matemática não certifica a convergência de PPO.

A interpretação de timeout artificial segue
[Pardo et al. (2018)](https://proceedings.mlr.press/v80/pardo18a.html).
Uma partida realmente finita, avaliada pelo score aos 2.000 ticks, é outro objetivo;
seria necessário explicitar tempo restante. O benchmark de vitórias/empates no
limite e o objetivo descontado de treinamento continuam distintos.

## Arquitetura e estabilidade

- `Policy(..., separate_critic=True)` mantém as três cabeças do ator e adiciona
  um tronco independente para o valor. Ambos têm duas camadas 256/Tanh.
- Ator e crítico têm parâmetros disjuntos, grupos Adam próprios e clipping 0,5
  separado. A norma combinada pode chegar a sqrt(2)*0,5 após clipping; não é o
  mesmo contrato do clipping global 0,5 da versão legada.
- O ator carregado é preservado exatamente. O tronco inicial do crítico copia o
  ator; a cabeça de valor é zerada ao mudar a recompensa. `--keep-critic` permite
  migração explícita quando a semântica do valor é compatível; não usar para
  presumir que valores de recompensas antigas já estejam calibrados ao PBRS.
- Inicialização ortogonal existe como opção de `Policy`, mas o treino atual
  conserva a inicialização padrão e os pesos do ator. Não se atribui melhoria
  de desempenho a uma troca de inicialização escondida.
- Primeiras 20 iterações do preset: só o crítico aprende. Gradientes do ator
  ficam `None`, sem criar passos/momentos Adam de gradiente zero. Foi verificado
  que o ator e seus casos de avaliação permanecem exatamente iguais nessa fase.
- KL aproximada acima de 0,03 interrompe os minibatches restantes da atualização;
  não é uma garantia formal de trust region. Número real de minibatches e parada
  antecipada ficam na telemetria. Baseline sem preset mantém esse guard desativado.
- Rollout 64, máximo de seis épocas/quatro minibatches, gamma com meia-vida 12 s,
  lambda 0,95, clip PPO 0,2 e VF_COEF 0,5 continuam explícitos.

A separação elimina interferência direta de gradientes de valor no tronco do ator
 e a influência do clipping conjunto. O ator continua dependente das vantagens
estimadas pelo crítico; não elimina toda interação entre aprendizado de política
e valor. Normas de gradientes não substituem a medida do update Adam; também
registramos a norma efetiva da atualização do ator em minibatches instrumentados.

## Currículo estacionário com âncoras neurais

`--curriculum-frac 1` usa somente adversários congelados. A identidade do adversário
é escolhida por ambiente e mantida até gol/timeout. No treino iniciado, a mistura é:

| Adversário | Probabilidade por novo episódio |
|---|---:|
| Chase com ruído | 20% |
| Chase sem ruído | 15% |
| Striker | 20% |
| Keeper | 5% |
| Random | 5% |
| Política inicial desse treino | 11,67% |
| Modelo previamente publicado | 11,67% |
| Origem B0 anterior ao treino noturno | 11,67% |

A inclusão das âncoras é uma resposta à regressão neural observada na madrugada;
65% do currículo está em heurísticas, 35% em políticas neurais fixas. Não é
100% de heurísticas nem um self-play atualizado continuamente. A transição é
estacionária condicionada ao adversário; sua identidade e detalhes internos do
jogo continuam ocultos ao MLP, portanto não se afirma observabilidade Markov plena.

`curriculum-frac < 1` permite comparar com self-play legado: o adversário de
self-play continua escolhido por rollout, podendo trocar dentro da partida.
Esse modo não recebe a alegação de estacionariedade do preset e não foi promovido
como solução de PFSP. O pool preserva snapshots do **ator**, sem critic inútil.

## Avaliação e seleção

O preset avalia greedy e amostrado, dois lados, kickoff/random e as âncoras, além
dos bots existentes. Seleciona pelo **mínimo dos quatro scores modo×cenário**,
para evitar escolher um pico só de kickoff greedy. Os casos individuais continuam
registrados: uma média por cenário ainda pode ocultar regressão em um adversário.
São necessárias novas seeds e inspeção de comportamento antes de promover um modelo.

O preset reúne intervenções para produzir um candidato estável; seu ganho eventual
não identifica qual intervenção causou a melhora. Para ablações, use flags explícitas
 e igual orçamento, por exemplo `--reward-table pbrs --shared-critic` versus
`--reward-table pbrs --separate-critic`, com `--critic-warmup-iters 0` nos dois braços,
o mesmo currículo e target-kl. Não misture os números com os seis braços antigos.

## Execução já iniciada

`experiments/sprint1_structural/launch.py` iniciou `runs/sprint1_structural_main_20261014`:
512 ambientes, alvo de 10.000 iterações, seed 0, PBRS/crítico separado/currículo
fixo, inicialização pelo melhor checkpoint noturno da iteração 3.650. Fonte,
mapa e três checkpoints iniciais foram congelados, com hashes em `launch.json`.
As primeiras 20 iterações calibram apenas o crítico. Há checkpoint retomável a
cada cinco iterações e snapshots do modelo a cada 50. A duração real depende da
máquina; 10:30 era uma possível hora de shutdown do usuário, não uma obrigação
para iniciar um novo modelo ou interromper um treino saudável.

## Shutdown e retomada

`resume.pt` salva pesos, Adam, iteração, horizonte dos schedules, melhor modelo,
referências, pool, RNGs, configuração e texto do mapa. Gravação usa temporário,
fsync e rename: uma falha de gravação não substitui o checkpoint anterior.
SIGTERM/SIGINT concluem ou descartam a coleta parcial, salvam estado coerente e saem.

**Limite real:** o engine não expõe serialização completa das partidas e flags
internas. A retomada começa partidas novas e reatribui os adversários; não é
continuação bit a bit dos estados físicos interrompidos. Pesos/Adam/schedule/pool
são preservados. Um hard shutdown pode perder as últimas cinco iterações ainda
não salvas. Telemetria é atômica, mas o bundle retomável é a fonte de verdade;
pesos/meta avulsos não devem ser usados para reconstruir uma etapa interrompida.

```bash
# Exportar um bundle coerente, sem alterar Git automaticamente:
.venv/bin/python src/backup_training.py \
  --run runs/sprint1_structural_main_20261014/train --dest models/sprint1_latest

# Depois do shutdown, continuar em pasta NOVA (não sobrescrever logs anteriores):
.venv/bin/python src/train.py --resume models/sprint1_latest/resume.pt \
  --out runs/sprint1_resumed_20261014

# Exportar somente o ator sem sobrescrever a política atual da sala:
.venv/bin/python headless-bot/export_policy.py models/sprint1_latest/model.pt \
  --out runs/sprint1_export/policy.json
```

Os scripts de inspeção e NeuralBot reconhecem critic separado; o exportador
preserva DefaultObs16 e inclui somente o ator. Não foi alterado o schema da
observação nem o comportamento de inferência da sala nesta etapa. Observações
enriquecidas/PFSP constituem a Sprint 2: exigem validação própria após esta base,
sem ocultar mudanças de entradas dentro de um experimento de recompensa/crítico.

## Verificações executadas

31 testes: máscaras GAE e física anteriores, telescopagem PBRS com término e
truncamento, simetria de potencial da bola, faixa do alvo, pesos/gamma, migração
sem alterar ator/valor, atualização só do crítico sem mover ator, diagnóstico de
gradientes disjuntos, rejeição de checkpoints parciais e sem mutação parcial,
identidade de adversário durante episódio, restauração de RNG e falha de gravação.

Testes integrados: smoke estrutural com dois anchors; parada SIGTERM e retomada
até a iteração 30; calibração do crítico com ator invariável e avaliação idêntica.
Resultados curtos só verificam integração. Não demonstram superioridade estatística.
O treino longo e as verificações de exportação têm seus resultados em `REPORT.md`.

## Evidência e publicações

A implementação e um bundle retomável foram publicados em `gpt`, commit `a6ba071`.
A verificação em seeds inéditas e o checkpoint exato avaliado estão em
`HELDOUT.md` e `models/sprint1_validated`. É um avanço observado nessa continuação,
não demonstra causalidade de cada intervenção nem robustez entre seeds de treino.

Uma ablação curta controlada está em `ablation.py`: duas seeds, 40 iterações,
64 ambientes, mesma origem/currículo/avaliação, sem warmup em todos os braços,
comparando corrected/shared, PBRS/shared e PBRS/separate (com clipping separado).
Esse POC serve para estudar direção/integridade; não deve determinar um vencedor
definitivo ou invalidar o preset principal por pouca amostragem.

Decisões e histórico de evidências: [JOURNAL.md](JOURNAL.md).

## Atualização do orçamento pelo usuário

O orçamento foi reduzido a **3000 iterações**. O horizonte dos schedules
originais é preservado; o supervisor envia SIGTERM no endpoint e não reinicia
treinamento depois. O benchmark pós-treino e preparação de jogo humano estão
especificados em [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).

## Continuação CUDA autorizada em 2026-10-14

A autorização posterior substitui a parada em 3000: retomar via --resume até o horizonte original de 10000, preservando receita, Adam, schedules, referências e avaliações internas. PyTorch CUDA 12.4; execução em runs/gpu_continuation_20261014_214556. Bundle final em models/sprint1_10000, preservando models/model.pt.

Em 15/10, o usuário autorizou publicar os registros e checkpoints intermediários,
exportações para teste web e instruções para warm start. `src/finalize_training.py`
acompanha a execução existente, aguarda o supervisor original e publica os artefatos
em ZIPs de tamanho limitado com manifesto SHA-256. Não reinicia o treinador.
Após confirmar o commit remoto, agenda shutdown com janela de 20 minutos para aviso.
Há também shutdown obrigatório às 11:44 (America/Sao_Paulo) de 15/10/2026;
se necessário, prepara um bundle parcial 20 minutos antes, preservando a retomada.


Recuperação em 15/10: uma substituição atômica de live.json falhou com WinError 5 após a iteração 5604. Adicionado retry limitado para leitores concorrentes no Windows; retomada via --resume do checkpoint durável 5600 em runs/gpu_continuation_recovery_20261015_035045. Registros anteriores e checkpoints intermediários foram preservados. A execução ativa está em runs/active_gpu_run.txt; o shutdown permanece às 11:48.

