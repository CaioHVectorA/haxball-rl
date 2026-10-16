# Warm start local — 16/10/2026

O usuário pediu continuar a partir do modelo atual e esclareceu que espera deixar
rodando por aproximadamente dez horas, podendo parar antes ou deixar mais.
**Não há orçamento de 100 mil iterações nem corte de dez horas.** O modo contínuo
encerra por sinal, deadline opcional ou falha; não relança silenciosamente após erro.
Nenhuma mudança no artigo LaTeX nesta etapa.

## Origem e receita

Melhor checkpoint GPU: `models/sprint1_10000/model.pt`, iteração 9900. Conferido
igual a `resume.pt:best_model`, pesos finitos. Warm start de ator **e crítico**:
Adam, RNG, pool e schedules recomeçam. Vinte iterações iniciais calibram somente
o crítico. Mapa **original exato** em `models/sprint1_10000/stadium.hbs`.

Preservados PBRS 0.15/0.08/25, crítico independente, meia-vida12s,
512 ambientes, rollout64/épocas6/minibatches4, KL .03. LR inicial1e-4 para ambos
os grupos, decrescendo até1e-4/3 em10000 novas iterações e permanecendo nesse piso
se o treino continuar. Entropia .02→.01 no mesmo horizonte, com piso .01.
Esse horizonte controla schedules; **não encerra o processo** com `--continuous`.
Escolha conservadora de fine-tuning, não ablação que demonstrou superioridade.

As duas âncoras antigas foram extraídas do checkpoint retomável. A referência
inicial agora é o modelo9900; não comparar scores brutos com o painel anterior.
Runtime: PyTorch2.14.1+cpu, CUDA indisponível; duas threads PyTorch.

## Incidente inicial e correção

O primeiro lançamento (`launch_manifest.json`, apenas histórico) usou o mapa
`futsal_1v1_contained.hbs`, validado no motor JS da sala. A avaliação inicial
apresentou forte assimetria por lado; ele foi encerrado por SIGTERM ainda durante
a calibração, antes de alterar o ator (identidade dos tensores conferida).
Não houve promoção desse modelo.

O probe reproduzível `map_probe.py`, seed1701,8env,200 decisões,greedy/Chase:

| Mapa | Lado | Gols a favor | Contra |
|---|---|---:|---:|
| Original do checkpoint | vermelho | 20 | 1 |
| Original do checkpoint | azul | 23 | 1 |
| Contenção | vermelho | 0 | 19 |
| Contenção | azul | 18 | 0 |

Números completos em `map_probe.json`. É um diagnóstico de compatibilidade,
não um benchmark de força. A causa interna na física/representação ainda não
foi isolada; não afirmar que determinado bias/plane foi o responsável. Testar
contenção no motor da sala não certifica o motor nativo de treinamento.
A correção desta execução é voltar ao mapa original; o problema de contenção no
motor nativo continua pendente. O mapa da sala web não foi modificado.

## Execução, parada e retomada

Fonte/mapa/origem/âncoras congelados no run; comando e hashes em `launch.json`.
`runs/active_local_warmstart.txt` identifica a execução. O launcher recusa um
outro treinador ativo. O manifesto ativo está em `active_launch_manifest.json`.

Resume atômico a cada5 iterações; snapshots a cada1000; avaliação a cada50, nos
dois modos/cenários/lados e referências. Melhor inclui baseline inicial. Originais
`models/model.pt` e `models/sprint1_10000` são preservados. Supervisor exporta
bundle coerente para `models/local_warmstart_latest` a cada30min e na saída.
É backup local, não push automático. Processo depende da máquina ligada.

Para parar: conferir PID/identidade em `launch.json` e enviar SIGTERM ao treinador,
aguardando saída normal e salvamento. Desligamento abrupto pode perder as últimas
cinco iterações ainda não salvas. Para retomar, UMA pasta nova:

```bash
.venv/bin/python src/train.py --resume CAMINHO/train/resume.pt --device cpu --out NOVO
```

A flag contínua é restaurada do checkpoint, junto de Adam/schedules; as partidas
físicas reiniciam. Não executar `launch.py` de novo para uma retomada: ele cria um
warm start novo. Manter o código que entende `--continuous` ao retomar.

## Validação do controle contínuo

Três testes de orçamento/iterador e floors, quatro de telemetria e14 estruturais
passaram. Teste real com horizonte2 ultrapassou esse horizonte e salvou resume
coerente após SIGTERM (`continuous_check.json`). Não se repetiu benchmark caro.
Esses testes verificam execução, não evolução. Mais treino pode regredir;
comparar melhor versus endpoint em dados reservados antes de promover o resultado.

## Dashboard local

Acesso: http://127.0.0.1:8085 — somente nesta máquina. Servidor somente leitura,
sem interferir no treinador. `runs/active.json` aponta para o subdiretório `train`
do run ativo. Inicialização: `.venv/bin/python live_stats_server.py --port 8085`.
Processo atual registra saída em `runs/dashboard_server.log`.

Painel ajustado para modo contínuo (sem ETA de fim), duas âncoras históricas e
termos PBRS reais. Score mostrado não é taxa de vitória. Gráficos atualizam por
SSE a cada segundo; avaliações novas aparecem a cada50 iterações. JS passou
verificação sintática e endpoints HTML/JS/state/history responderam200 com dados
do run ativo. Não houve alteração na fonte congelada do treinador.
