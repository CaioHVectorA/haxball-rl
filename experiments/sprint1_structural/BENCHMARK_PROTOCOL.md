# Encerramento em 3000 e benchmark pré-especificado

O usuário pediu parar em 3000, comparar self-play/bots de comportamentos distintos
e jogar contra o modelo. Esse pedido substitui o orçamento anterior de 10000.

## Parada

`stop_at_iteration.py` verifica comando e identidade de início do processo,
observa a iteração concluída e envia SIGTERM em 3000. O treinador já possui
salvamento completo nesse sinal. Os schedules conservam o horizonte original
10000: encerrar a coleta não os comprime retrospectivamente para 3000.
Estado em `runs/sprint1_structural_main_20261014/iteration_stop.json`.
Não reiniciar treinamento depois de atingir o endpoint autorizado.

`after_3000.py` aguarda a parada, verifica a resume.pt final, congela o bundle
em `models/sprint1_3000`, executa o benchmark e prepara exportação/sala humana.
O bundle inclui o melhor checkpoint selecionado no painel e o endpoint separado.
Nada substitui `models/model.pt` original. Estado: `runs/sprint1_after_3000.json`.

## Comparação

- Três candidatos congelados: origem da continuação, melhor selecionado antes
  desse benchmark e endpoint 3000.
- Adversários: a própria política, origem, melhor e endpoint; Chase com ruído,
  Striker, Keeper; novas heurísticas Interceptor, CounterDefender, Dribbler,
  AimedStriker. Self contra self não mede força absoluta.
- Dois lados, random/kickoff, ações greedy/amostradas, mesmo mapa e ticks.
- Seeds inéditas 9101–9116, 32 ambientes por caso. São seeds de avaliação,
  não 16 seeds de treinamento.
- Apenas a primeira partida COMPLETA de cada ambiente: gol encerra a partida;
  timeout em 2000 ticks, aproximadamente 33,33 segundos, conta como empate.
  Não equivale a futebol multigol com placar ao final de vários minutos.
- Kickoff greedy contra oponentes determinísticos usa uma seed: repetições
  idênticas não criam incerteza estatística. Chase tem ruído, portanto esse
  confronto conserva as 16 seeds mesmo com ator greedy.
- Orçamento padrão: 3324 casos, 106368 episódios, incluindo as repetições
  determinísticas explicitamente sinalizadas. Contagens não são tamanho amostral
  de trajetórias independentes.

Ator inicial e mapa são congelados por hash; estados iniciais são pareados e
conferidos por hash entre os candidatos. A seed controla o mutator, ruído do
Chase e ações amostradas. A troca de lados usa os mesmos estados físicos,
sem prometer espelhamento de cada partida aleatória. O bootstrap reamostra
blocos de seed, agregando lados; também há diferenças pareadas contra origem
e entre endpoint/melhor. ICs de 95% são descritivos, pontuais, sem correção de
multiplicidade ou certificação de força humana. Nenhuma seed muda conforme score.

Publicar V/E/D, scores por modo/cenário/adversário/lado, empates, duração,
fração de bola parada e trajetórias. Não substituir essa análise por recompensa
de treino. Um self-score 0,5 pode coexistir com bom jogo ou impasse completo;
os diagnósticos e trajetórias distinguem esses comportamentos.

## Bots táticos: limites explícitos

Interceptor prevê linearmente a posição da bola até 48 ticks e pressiona chute
na faixa; ignora rebotes. CounterDefender guarda a região do gol e contra-ataca
ao se aproximar da bola. Dribbler tenta carregar por uma faixa afastada do defensor,
liberando chute até pressão/proximidade do gol. AimedStriker se posiciona atrás
da bola mirando uma região do gol afastada do adversário.

São scripts que usam o estado físico, não bots com inteligência humana comprovada
nem políticas aprendidas fortes por definição. Têm falhas próprias; seus resultados
medem robustez a esses estilos específicos. Os quatro estilos ficaram fora do
currículo congelado deste treino. Reflexão de time/referencial passou em teste;
o smoke concluiu primeiras partidas de ambos os lados para todos os estilos.

## Jogo humano

O modelo da sala será o **melhor selecionado antes do benchmark**, congelado em
`models/sprint1_3000/model.pt`. Não se escolhe um novo vencedor após olhar esse
benchmark e não se declara que resultados simulados certificam a sala real.

`POLICY_PATH` permite apontar a sala para a exportação desse bundle sem mexer na
política padrão. Paridade JS/PyTorch é conferida em 256 observações dos dois lados
para os novos pesos. O loader tenta usar o token local já configurado, sem publicar
o segredo. Sala só está pronta após aparecer link real de conexão no log.
Se o token tiver expirado, a exportação continua disponível e o erro será informado.

Depois de concluída a avaliação, o comando manual é:

```bash
POLICY_PATH="$PWD/models/sprint1_3000/policy.json" \
  node headless-bot/bot.js
```

Senha padrão `bot`. `!start`, `!restart` e `!swap` controlam a partida.
O acesso depende do computador e do processo da sala permanecerem ligados.

## Auditoria e gráficos finais

Após status completed do benchmark, executar
`.venv/bin/python experiments/sprint1_structural/benchmark_report.py`.
O publicador confere contagem, unicidade, todos os primeiros episódios completos,
estados iniciais pareados, hashes e pesos finitos; copia apenas sumários e figuras
para experiments/sprint1_structural/benchmark_3000. Os gráficos incluem desempenho
por modo/cenário, empates/duração no self-play e seis trajetórias de exemplos
pré-fixados (seed9101, ambiente0, greedy/random). A auditoria não substitui
interpretação crítica de efeitos, empates, incerteza e falhas por adversário.
