# Adaptação 1v1 para 2v2

Branch `gpt_2v2`. Fonte: `models/sprint1_10000/model.pt`, ator da iteração 9900.
O modelo 1v1 e seus bundles são preservados. Este é um novo orçamento inicial
de 10000 iterações 2v2, sem prolongar o treino 1v1.

## Iniciar

```powershell
python src/run_team_training.py --run runs/team2v2_TIMESTAMP --iters 10000
```

O supervisor executa uma única instância CUDA, sem reinício automático:

```powershell
python src/train.py --team2v2 --transfer-1v1 --warm-start models/sprint1_10000/model.pt --device cuda --envs 256 --iters 10000 --eval-every 100 --out runs/team2v2_TIMESTAMP --checkpoint-every 5 --save-every 50
```

256 partidas x dois aprendizes = 512 amostras por decisão: batch 32768.
MLP 256/Tanh x2, três cabeças, crítico independente novo, Adam novo, warmup do
crítico 20, LR 3e-4 -> 1e-4, entropia .02 -> .01, target KL .03, GAE e clipping
preservados. O mapa futsal e a física são os mesmos; resets posicionam quatro
jogadores sem sobreposição.

## Transferência e equipes

`team24_v1`: base própria/bola/gols (12), adversário mais próximo (4), companheiro
(4), segundo adversário (4). Blocos de outros jogadores contêm posição relativa e
velocidade; escala do mapa e espelhamento de X no azul. Distâncias iguais usam a
ordem dos jogadores. O ator transfere as primeiras 16 colunas, zera as oito novas
e preserva demais camadas/cabeças. Assim, inicialmente reproduz o 1v1 contra o
adversário mais próximo e pode aprender a considerar os demais. Crítico novo.
A origem e o SHA-256 são registrados no checkpoint.

Os dois vermelhos compartilham o ator e recebem retorno de equipe: gol ±25,
progresso PBRS .08 e posicionamento PBRS .15. O potencial de posicionamento usa o
melhor colocado do time, evitando premiar dois perseguidores individuais da bola.
Não são adicionados bônus artificiais por passes.

Azuis usam um ator congelado por rollout: 50% snapshot histórico quando disponível,
caso contrário cópia atual. Pool de 40, adição a cada 20 iterações. A escolha é por
rollout, não por episódio; não é uma liga nem um currículo fixo.

Avaliação a cada 100 iterações controla equipes completas nos dois lados, seeds
1701/1702, random/kickoff e greedy/amostrado, contra quatro heurísticas e o ator
transferido inicial. Seleção pelo menor score dos quatro grupos modo x cenário.
Partidas contam uma vez, não por jogador. Não comparar diretamente scores 1v1 e
2v2 como a mesma tarefa nem inferir força humana/cooperação sem validação.

## Acompanhar e retomar

`runs/active_team2v2_run.txt` aponta para a execução. O dashboard usa
`runs/active.json`: `python live_stats_server.py --port 8085`.
Estado e registros: `live.json`, `training.jsonl`, `evaluations.jsonl`, `train.log`.
Criar `STOP` dentro da execução encerra com save no próximo limite de iteração.

```powershell
python src/train.py --resume runs/team2v2_TIMESTAMP/resume.pt --device cuda --out runs/team2v2_resumed_TIMESTAMP --checkpoint-every 5 --save-every 50
```

Retoma modelo, Adam, schedules, referências, pool e RNGs; partidas recomeçam.
A saída deve ser nova. O supervisor atualiza `models/team2v2_latest` localmente
a cada ~100 iterações e publica o bundle e registros/checkpoints em ZIPs com
manifesto no encerramento. Não agenda shutdown nem reutiliza a automação 1v1.

## Web

`policy.json`: melhor ator; `policy_latest.json`: endpoint. Após existir um bundle:

```powershell
npm --prefix headless-bot ci
$env:POLICY_PATH = (Resolve-Path models/team2v2_latest/policy.json).Path
node headless-bot/bot.js
```

Forneça HEADLESS_TOKEN só no ambiente local. A sala cria dois bots e aguarda dois
humanos no outro time; imprime o link web. `!swap` troca ambos os times. Não faz
inferência com equipes incompletas. Alternar entre 1v1/2v2 requer reiniciar a sala.
Observações e logits Python/JavaScript foram comparados nos quatro jogadores;
uma sala humana real ainda não foi validada.

## Validação

Seis testes específicos: simetria e permutação, resets/ações, recompensa PBRS de
equipe, transferência/logits/novos gradientes, avaliação e paridade JavaScript.
Smoke CUDA incluiu atualização real do ator/crítico, save e retomada. Nenhum
benchmark ou ablação foi refeito.

Suíte completa: 38/39 passando no Windows. O teste legado
`test_iteration_watcher_stops_identified_process_and_saves_endpoint` depende de
`/proc` e SIGTERM do Linux; esse supervisor não é usado pelo treino 2v2.
