# Sala de Haxball com política 1v1

`export_policy.py` converte pesos PyTorch para `policy.json`; `bot.js` usa node-haxball para executar o bot na sala. A integração deve ser validada em partidas reais, incluindo observações, controles, mapa e latência.

## Execução

A partir da raiz do repositório:

```bash
uv run headless-bot/export_policy.py models/model.pt --stadium futsal-classic
cd headless-bot
npm install
HEADLESS_TOKEN="<token>" node bot.js
```

Forneça explicitamente o checkpoint: o caminho padrão antigo do exportador aponta para `rl/checkpoints/`, enquanto o treinamento atual grava em `models/` ou no diretório de `--out`.

O exportador exige observações de 16 dimensões e duas camadas na política. Não está preparado para equipes. Reexporte depois de escolher um checkpoint e confira os metadados e o estádio usados.

O bot imprime o link da sala. Comandos de chat: `!start`, `!stop`, `!restart`, `!swap`, `!reload`, `!who`, `!help`. Consulte `bot.js` para comportamento e variáveis disponíveis, incluindo `ROOM_NAME`, `ROOM_PASS`, `TICK_SKIP` e `FLIP_Y`.

O desempenho depende do checkpoint. A existência de `policy.json` não comprova que o agente esteja consolidado nem que o arquivo corresponda aos pesos mais recentes.
