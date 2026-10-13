# Haxball RL

Agentes de reinforcement learning para Haxball, treinados com PPO e self-play sobre HaxballGym. A prioridade atual é consolidar o 1v1 em `futsal-classic`, com avaliação reproduzível, antes de experimentar equipes e fine-tuning.

## Uso

Python >=3.12, uv e PyTorch CPU conforme `pyproject.toml` e `uv.lock`:

```bash
uv sync
uv run src/train.py --stadium maps/futsal_1v1.hbs --red 1 --blue 1 --envs 512 --iters 1500 --out models/run_1v1_seed0 --seed 0
```

Use uma pasta por execução. O treinamento grava pesos e metadados, mas ainda não salva a configuração completa, estado do otimizador ou pool de adversários. `--warm-start` carrega pesos, zerando a cabeça de valor por padrão; `--keep-critic` preserva essa cabeça.

Os comandos abaixo leem os modelos da pasta padrão `models/`:

```bash
uv run src/eval.py --benchmark --stadium futsal-classic
uv run src/select.py --stadium futsal-classic
```

`select.py` promove o vencedor para `models/model.pt` e atualiza `meta.json`. Os scripts antigos usam gols agregados. `src/repair_bot.py` oferece uma avaliação mais completa com seeds, cenários variados e resultados por episódio.

## Experimento de correção 1v1

```bash
uv run src/repair_bot.py --out experiments/nova_correcao --iters 100 --seeds 0 1 2
uv run python -m unittest discover -s tests -v
```

O experimento salva os pesos em pasta separada. Usa mapa com paredes corrigidas, recompensas por deslocamento e chute rearmado.

## Sala real

Execute a partir da raiz do repositório:

```bash
uv run headless-bot/export_policy.py models/model.pt
cd headless-bot
npm install
HEADLESS_TOKEN="<token>" node bot.js
```

O exportador atual aceita observações de 16 dimensões e uma rede de duas camadas, para 1v1. A transferência para a sala precisa ser validada em partidas reais; equivalência de física, observações e latência não é certificada por este repositório. Veja `headless-bot/README.md`.

## Organização

- `src/model.py`, `src/bot.py`: política e bots de referência.
- `src/train.py`: PPO, recompensas e self-play.
- `src/eval.py`, `src/select.py`: benchmark e torneio de checkpoints.
- `src/replay.py`: utilitários de replay.
- `maps/`: estádios; `models/`: pesos e metadados.
- Scripts `inspect_*` e `test_*` na raiz: diagnósticos e experimentos, sem uma suíte unificada.

`AGENTS.md` descreve a arquitetura implementada; `PROTOCOL.md`, os parâmetros e recompensas; `TODO.md`, as próximas etapas; `CONTEXT.md`, o histórico e as limitações dos resultados disponíveis.
