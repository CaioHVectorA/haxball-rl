# Ablação curta de componentes

Dois seeds, 40 iterações ×64 ambientes: **163.840 amostras por braço/seed**.
Ator inicial tensor a tensor, avaliação inicial, currículo e orçamento foram
conferidos iguais. C4 usa o warmup já implementado antes dos resultados; ele
aloca 20 iterações ao crítico e 20 ao ator, enquanto os demais treinam o ator
em 40. Esse é o tratamento comparado, não igualdade de número de passos do ator.

| Variante | Seed 0 | Seed 1 | Média do mínimo |
|---|---:|---:|---:|
| C1_corrected_shared | 0.65646 | 0.59226 | 0.62436 |
| C2_pbrs_shared | 0.68869 | 0.66084 | 0.67477 |
| C3_pbrs_separate | 0.59762 | 0.73750 | 0.66756 |
| C4_pbrs_separate_calibrated | 0.67560 | 0.72083 | 0.69821 |

Baseline de todos os braços: 0,69405. Cada endpoint usa mínimo dos quatro
scores modo×cenário. Existem poucos episódios/casos; não confundir este
critério com a média de todos os casos ou com o orçamento do treino principal.

## Interpretação

C2 supera C1 nos dois seeds deste POC, mas os endpoints permanecem abaixo
do baseline e o orçamento é pequeno. C3 sem calibração tem resultados bem
diferentes entre seeds: separar parâmetros elimina uma interferência direta
demonstrável, porém não é garantia de melhora de score em qualquer seed.
C4 verifica a calibração critic-only que integra o preset principal.
Os números desse teste não justificam escolher hiperparâmetros retrospectivamente
até conseguir um resultado favorável. A evidência de avanço do candidato
principal é a validação em seeds inéditas registrada em HELDOUT.md.

A conclusão empírica deve ser ampliada em orçamento maior e mais seeds de
treinamento. O preset principal também contém calibração e está em outro
orçamento. Não se mistura seu resultado com esta ablação como se fosse
efeito causal isolado de uma das peças.
