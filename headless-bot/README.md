# Sala de Haxball com política 1v1

`export_policy.py` converte pesos PyTorch para `policy.json`; `bot.js` usa node-haxball para executar o bot na sala. A integração deve ser validada em partidas reais, incluindo observações, controles, mapa e latência.

## Execução

A partir da raiz do repositório:

```bash
uv run headless-bot/export_policy.py models/model.pt
cd headless-bot
npm install
HEADLESS_TOKEN="<token>" node bot.js
```

O caminho padrão do exportador é `models/`. O estádio vem de `meta.json`; forneça `--stadium` apenas para sobrescrever essa escolha. Políticas corrigidas incluem `repeat_kick`, rearmando o chute entre decisões na sala, assim como no simulador.

O exportador aceita `default16` (1v1) e `team24_v1` (2v2), com duas camadas na política. A sala cria um ou dois bots conforme `team_size` e aguarda times completos. Para 2v2, veja [a receita](../experiments/team2v2/README.md). Reexporte depois de escolher um checkpoint e confira os metadados e o estádio usados.

O bot imprime o link da sala. Comandos de chat: `!start`, `!stop`, `!restart`, `!swap`, `!reload`, `!who`, `!help`. Consulte `bot.js` para comportamento e variáveis disponíveis, incluindo `ROOM_NAME`, `ROOM_PASS`, `TICK_SKIP` e `FLIP_Y`.

O desempenho depende do checkpoint. A existência de `policy.json` não comprova que o agente esteja consolidado nem que o arquivo corresponda aos pesos mais recentes.

Para carregar uma política específica sem trocar policy.json padrão, use
`POLICY_PATH=/caminho/absoluto/policy.json node headless-bot/bot.js` na raiz.
O bundle de 3000 será preparado e avaliado antes da sala humana; veja
`experiments/sprint1_structural/BENCHMARK_PROTOCOL.md`.
