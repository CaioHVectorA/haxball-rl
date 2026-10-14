# Validação inicial — 2026-10-14

Escopo concluído: primeira etapa B0/E1 da Sprint 1. PBRS, crítico separado e
currículo ainda não foram implementados nesta etapa.

## Evidência

- `python -m unittest discover -s tests -v`: 16 testes aprovados.
- `git diff --check`: sem erros.
- Runner pareado completou B0 e E1, seed 0, 16 ambientes × 64 decisões × quatro
  iterações = 4.096 amostras por braço.
- Checkpoints iniciais comparados tensor a tensor: idênticos.
- Mesmos resultados de avaliação inicial nos dois braços.
- Nas primeiras três iterações não houve timeout; os registros de desempenho
  e os checkpoints finais não são usados para declarar um vencedor.
- Na quarta iteração, ambos registraram dez timeouts e V(final) médio
  2,7255010605. A correção acrescenta gamma × V(final) ao residual desses
  timeouts em E1. B0 mantém zero; o traço não atravessa reset em nenhum braço.

| Variante | Random final | Kickoff final | Norma ator (it. 4) | Norma crítico (it. 4) |
|---|---:|---:|---:|---:|
| B0 | 0,50417 | 0,475 | 0,90801 | 3,92279 |
| E1 | 0,53958 | 0,450 | 0,91126 | 6,36580 |

Normas por objetivo, médias de seis minibatches instrumentados. O crítico inclui
seu coeficiente de perda; ator inclui entropia. Painel greedy com dois ambientes,
250 decisões por caso; escores são médias dos casos com episódios completos.
Esses números apenas demonstram execução e geração de telemetria. Não há poder
estatístico para afirmar melhoria, convergência, domínio do crítico no Adam ou
aprovar a passagem à Sprint 2.

Artefatos locais: `runs/sprint1_timeout_smoke_20261014/`, com manifesto, fonte,
dependências, logs, checkpoints e métricas. O checkpoint de origem tinha SHA-256
`42eb2b35d0a5591b195da38ce26d95464ce280cb29313eec4819486c238608d9`.
Modelos principais não foram promovidos ou sobrescritos.

O experimento principal de três seeds e 300 iterações está preparado no runner,
mas não foi executado nesta validação inicial. Veja `README.md` desta pasta.
