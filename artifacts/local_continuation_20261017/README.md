# Dados locais complementares — 17/10/2026

Delta em relação a `artifacts/full_runs_20261014/manifest.json`: arquivos novos
ou cujo SHA mudou em runs, incluindo fonte congelada, telemetria e checkpoints
da continuação encerrada em8018. Partes de até4MiB, manifesto por arquivo/parte,
checksums verificados contra o conteúdo real do TAR. Arquivos antigos sem mudança
continuam no pacote histórico, sem duplicação. Nenhuma execução foi reiniciada.

Tokens, arquivos.env, logs/diretórios de salas privadas, dependências e caches não
fazem parte do pacote. Uma busca por padrões de credenciais nos textos arquivados
não encontrou correspondências. Não representa prova geral de ausência de segredos.
As exclusões estão no manifesto; arquivos privados de sala não são publicáveis.

Para restaurar toda a história em uma pasta separada, primeiro rode o restaurador
de `full_runs_20261014`; depois aplique este delta:

```
python artifacts/full_runs_20261014/restore.py --dest /caminho/recuperacao
python artifacts/local_continuation_20261017/restore.py --dest /caminho/recuperacao
```

O delta sobrepõe versões antigas, não apaga histórico ausente. PIDs, caminhos locais
ou status running de logs históricos não demonstram um processo atual. O treino
principal foi encerrado por pedido do usuário. O modelo preferido está publicado
separadamente em `models/local_warmstart_latest/model.pt` (melhor7300).
