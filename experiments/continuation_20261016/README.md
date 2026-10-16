# Warm start local — 16/10/2026

Autorização: usuário pediu continuar o treinamento do modelo atual em seu ambiente,
por longo prazo. Uma única execução foi iniciada; nenhuma mudança no artigo LaTeX.

## Origem e diferenças explícitas

- Melhor checkpoint da continuação GPU: `models/sprint1_10000/model.pt`, iteração
  9900. Conferido idêntico a `resume.pt:best_model`, com pesos finitos.
- Warm start de ator **e crítico**, não retomada do endpoint 10000: Adam, RNG,
  pool e schedules reiniciam. O ator permanece fixo nas primeiras 20 iterações
  de calibração do crítico. O estado inicial inteiro foi comparado tensor a tensor.
- Mapa `maps/futsal_1v1_contained.hbs`: correção de fronteira já testada em 2160
  cenários físicos. A física de fronteira mudou; não comparar scores diretamente
  ao mapa antigo nem afirmar ausência de todo bug possível.
- Preservados PBRS 0.15/0.08/25, crítico independente, desconto de meia-vida 12 s,
  512 ambientes, rollout64/épocas6/minibatches4, target KL .03.
- LR inicial de ator e crítico **1e-4**, em vez de reaquecer a 3e-4; schedule
  linear até 1e-4/3 ao final do bloco. Entropia segue preset .02→.01. Isso é uma
  escolha conservadora para esta etapa, não uma ablação demonstrando superioridade.
- Mesmas duas âncoras extraídas do checkpoint retomável anterior; a referência
  inicial agora é a política 9900. Portanto o painel tem uma referência nova:
  seu score bruto não é diretamente comparável ao painel antigo.
- Runtime detectado: PyTorch 2.14.1+cpu, CUDA indisponível. CPU com 2 threads
  PyTorch; não se afirma execução na GPU do outro ambiente.

## Orçamento e durabilidade

Primeiro bloco **100000 novas iterações** (até 3.276.800.000 transições), sem limite
por horário. É um orçamento longo finito, não uma promessa de evolução ilimitada.
O supervisor não lança ciclos sem fim nem reinicia silenciosamente uma falha.
Ao terminar o bloco, comparar melhor versus endpoint e decidir a próxima etapa.

Fonte, mapa, origem e âncoras ficam congelados dentro da pasta do run; hashes e
comando completo em `launch.json`. `runs/active_local_warmstart.txt` identifica o
run ativo. Um novo lançamento recusa outro treinador já ativo.

Resume atômico a cada 5 iterações; snapshots a cada 1000; avaliação a cada 50
inclui modos greedy/amostrado, lados, kickoff/random, heurísticas e referências.
O melhor modelo inclui o baseline antes de treinar e é selecionado pelo mínimo
modo×cenário. Não substitui `models/model.pt` nem o checkpoint GPU publicado.
O supervisor exporta bundle coerente em `models/local_warmstart_latest` a cada
30 minutos e após a saída. Isso é backup local; não significa push automático.

Para encerrar, conferir a identidade do PID em `launch.json` e enviar SIGTERM ao
**treinador**, permitindo salvamento normal. Para retomar após shutdown, usar UMA
pasta nova e `src/train.py --resume CAMINHO/train/resume.pt --device cpu --out NOVO`;
as partidas físicas recomeçam, mas Adam e schedules desse bloco são preservados.
A disponibilidade do processo depende de a máquina permanecer ligada.

## Interpretação

Mais treino pode melhorar e também causar regressão. A avaliação mede progresso
no painel conhecido; só um benchmark reservado posterior pode avaliar generalização.
Nenhum ganho novo é reivindicado no lançamento. Seeds de avaliação não representam
novas seeds independentes de treinamento. Não trocar hipóteses/recompensa em resposta
a um score isolado. Revisar referências individuais, curvas e comportamento antes
 de promover um novo modelo para jogo humano.
