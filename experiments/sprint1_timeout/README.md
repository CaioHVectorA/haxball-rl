# Sprint 1 — etapa B0/E1

Implementação inicial da ablação sequencial, em 1v1. A sprint completa ainda inclui
PBRS e avaliação de crítico separado; currículo estacionário é outra intervenção e
precisa de comparação própria. A observação enriquecida pertence à etapa posterior.

## Contrato

- `--timeout-bootstrap zero` (padrão): controle B0 com a semântica anterior.
- `--timeout-bootstrap final`: E1. Interpreta o limite de 2.000 ticks como corte
  artificial de coleta de uma tarefa que poderia continuar. Gol é terminal.
- E1 usa `delta = r + gamma * (1-terminated) * V(final) - V(current)` nos
  timeouts; ambos os modos cortam o traço em `terminated | truncated`.
- `RewardMeter.last_state` captura o estado antes do autoreset. Usa-se o
  `DefaultObs` atual, sem histórico; futuros builders com memória exigem revisão.
- Terminação simultânea a timeout prevalece: bootstrap zero.
- Permanecem T=64, seis épocas, quatro minibatches, rede compartilhada e
  inicialização padrão. Checkpoints e exportação mantêm o formato.
- Aproximação permanece **0,15**, progresso 0,08, gol 25. O valor 0,05 do texto
  anexado não era o padrão do código. É configurável, mas deve ser igual nos braços.

O bootstrap corrige o alvo **sob a interpretação de timeout artificial**. Se o
objetivo for uma partida realmente finita de 2.000 ticks, essa semântica deve ser
reconsiderada, incluindo a representação do tempo restante. A avaliação continua
reportando empate no limite de tempo; sua pontuação não é o retorno descontado.

## Diagnósticos

`training.jsonl` registra normas separadas por objetivo, antes do clipping:
ator = surrogate PPO menos bônus de entropia; crítico = `VF_COEF * value_loss`.
Obtidas com `autograd.grad`, sem alterar `.grad` usado pelo otimizador, no primeiro
minibatch de cada época (seis medições por iteração). Cosseno usa somente o tronco;
é `null` quando alguma norma é zero, e a média exclui esses casos.
A norma combinada é medida em todos os minibatches. Não se pode deduzir domínio
do update Adam apenas dessas normas.

`timeout_bootstrap_mean` é V(final) medido nos dois braços, acompanhado de contagem
e indicador de aplicação. Ausência de timeout gera `null`, não zero.
`evaluations.jsonl` separa random/kickoff; `evaluation_cases.jsonl` preserva
adversário, lado e seed. Consulte as linhas da iteração registrada em `meta.json`
(melhor checkpoint) e `meta_final.json` (endpoint). Pontuações com poucos episódios
não constituem evidência de melhora.

## Execução pareada

```bash
.venv/bin/python experiments/sprint1_timeout/run.py \
  --checkpoint models/model.pt --out runs/sprint1_timeout_full \
  --seeds 0 1 2 --envs 512 --iters 300
```

O runner exige pasta nova, congela uma cópia do checkpoint e usa a mesma origem,
seed, reset do crítico e orçamento em B0/E1. Executa a cópia congelada do código e do mapa, salva diff, dependências,
hashes e logs; não promove modelos. O orçamento é 9.830.400 amostras por braço/seed,
58.982.400 no total. É uma comparação sequencial, não um desenho fatorial completo.
Uma execução interrompida fica identificada como falha e não oferece retomada exata.

Teste curto executado separadamente: `runs/sprint1_timeout_smoke_20261014`,
seed 0, 16 ambientes, quatro iterações (4.096 amostras por braço), painel de dois
ambientes e 250 decisões por caso. Serve para verificar a integração, não para
selecionar uma política ou afirmar convergência.

## Critério para prosseguir

Testes devem cobrir timeout no fim do rollout, gol+timeout simultâneos, ausência de
vazamento entre episódios, estado final real e diagnóstico sem alterar gradientes.
O teste integrado deve completar B0/E1, registrar timeouts e manter o pareamento
inicial. Depois, executar o orçamento principal antes de julgar desempenho e decidir
sobre E2 (PBRS), E3 (crítico separado) e currículo em comparações adicionais.

## Validação principal iniciada

Execução: `runs/sprint1_timeout_full_20261014`, seeds 0/1/2, 512 ambientes e
300 iterações, mantendo o checkpoint congelado do smoke test. São seis treinos
sequenciais; `manifest.json` registra o braço corrente e os concluídos.

Após cada braço, o runner gera `REPORT.md` e `summary.json` automaticamente,
verificando orçamento, hash da origem, painel completo e, ao concluir cada par,
igualdade dos pesos e avaliações iniciais. O relatório separa diferenças E1−B0
por seed no endpoint, com random/kickoff e casos por adversário/lado.
A avaliação no mesmo painel é exploratória; múltiplos ambientes de kickoff
determinístico não são replicações independentes.

Para reconstruir a análise a qualquer momento:

```bash
.venv/bin/python experiments/sprint1_timeout/summarize.py runs/sprint1_timeout_full_20261014
```

O runner identifica falhas e não promove modelos. A conclusão do processo deve
ser conferida em `manifest.json`; iniciar o experimento não significa validá-lo.
