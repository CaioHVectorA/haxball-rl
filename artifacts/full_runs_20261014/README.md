# Snapshot integral dos experimentos — 14/10/2026

Este pacote foi preparado por pedido explícito de publicar todo o projeto.
Os códigos, documentos, modelos escolhidos, gráficos e replays ficam nas pastas
normais do repositório. `codigo_completo.txt` é uma cópia histórica do usuário;
seu conteúdo não substitui a documentação e os arquivos atuais.

Os dados anteriormente ignorados pelo Git estão preservados aqui:

- 951 arquivos de `runs/`: dados brutos, métricas, seeds, mapas, fontes congeladas,
  checkpoints intermediários, Adam/RNGs, avaliações e trajetórias.
- 25 checkpoints legados de `models/ckpts/` em `legacy_model_checkpoints.tar.gz`.
- O smoke legado de `experiments/gpt_smoke/` em `legacy_smoke.tar.gz`.

O arquivo dos runs tem 169573730 bytes, dividido em 41 partes de até4MiB.
O manifesto registra caminhos, tamanhos e SHA256 dos arquivos e das partes.
Todas as951 entradas foram verificadas dentro do arquivo gerado. Hardlinks
evitam armazenar novamente arquivos byte a byte idênticos.

Segredos (`.env`, tokens e credenciais), logs privados da sala, dependências
instaladas (`.venv`, `node_modules`) e caches/bytecode não são conteúdo publicado.
As dependências podem ser reinstaladas com os lockfiles do projeto.

Para restaurar os dados numa cópia do projeto:

```bash
python3 artifacts/full_runs_20261014/restore.py --dest "$PWD"
```

O restaurador confere as partes e os arquivos antes/depois da extração. Use uma
cópia separada se já houver experiências locais com os mesmos nomes.

## Estado da publicação

Este snapshot foi preparado e auditado localmente. A tentativa de publicação
neste ambiente foi bloqueada: Git não resolveugithub.com; a conexão do app
exigiu aprovação nas operações de escrita, com política de aprovação `never`.
Nada aqui afirma que o snapshot novo tenha sido enviado. O último commit remoto
confirmado antes desse pedido foi `e1d31fc7ecb29e46a0c16464f7b17614bcacb837`.

Com acesso normal de escrita/rede, na raiz:

```bash
bash scripts/publish_full_snapshot.sh
```
