# Decisões e evidências — Sprint 1 estrutural

## 14/10/2026 — implementação

O trabalho anterior mediu B0/E1 mas não executou as alterações centrais. Isso foi
corrigido: PBRS, crítico separado/clipping, currículo estacionário com âncoras,
calibração inicial critic-only, seleção mínimo modo×cenário e retomada durável
foram implementados. O preset está efetivamente em execução, com fonte congelada.
Detalhes, fundamentos e limites em README.md.

Publicações na branch gpt:

- `24e7bb6`: trabalho anterior e análise noturna, publicado antes das novas alterações.
- `a6ba071`: código estrutural, testes, documentos e bundle retomável (it280/best250).
- `8d05fc6`: validação independente e preservação do checkpoint testado (it450).

## Verificação

31 testes unitários e os testes integrados de calibração/retomada passaram. O sinal
SIGTERM interrompeu após iteração 3; a retomada chegou à 30 com estado Adam e
contador preservados. O ator não mudou durante calibração; o crítico mudou e a
avaliação foi exatamente igual. O exportador e JS concordaram em 256 observações,
os dois lados, com erro máximo de logits 4,54e-6 e mesmas ações greedy. Rearmamento
no headless continuou correto.

A retomada preserva RNG PyTorch, NumPy global, gerador do treinador e RNGs do
currículo/bots. As partidas nativas não são serializadas; os mutators do ambiente
são recriados e começam partidas novas a partir de sua receita/seeds. Não afirmar
continuidade física nem repetição bit a bit da trajetória anterior. O bundle único
é a fonte de verdade para modelo/Adam/iteração; pesos e meta avulsos são derivados.

## Progresso e interpretação

No painel de seleção, o mínimo começou em 0,71734 e chegou a 0,78260 na it100,
0,79734 na 250, 0,82701 na 450 e 0,83560 na 600. Os critérios e adversários são
iguais entre essas avaliações, mas novos em relação aos seis braços B0/E1; não
comparar diretamente os números como se fossem o mesmo benchmark.

A validação inédita do candidato it450 usou seeds 5701–5703, 336 casos únicos,
8 ambientes, 500 decisões, dois lados e ambos os modos, sete adversários. O
mínimo modo×cenário foi de 0,69005 na origem a 0,82938 no novo candidato. Cada
adversário apresentou diferença agregada positiva nesse teste; incluindo as
referências neurais. Consulte HELDOUT.md. Isso é evidência observada de avanço
nessa continuação, não prova estatística de robustez em várias seeds de treino
nem atribuição causal a cada intervenção do preset.

A instabilidade greedy não desapareceu: na it650 o grupo greedy/kickoff caiu a
0,70588, enquanto sampled/kickoff foi 0,88831. Na 700 os valores voltaram a
0,99097 e 0,88357. O primeiro comando de kickoff foi idêntico nos checkpoints
450 e 650 ([2,2,0]), portanto esses dados não sustentam culpar exclusivamente uma
troca da primeira ação. Mudanças de outras ações/probabilidades durante a partida
precisam de inspeção localizada para uma conclusão causal. Não se ajustaram
hipóteses/hiperparâmetros apenas para obter um score maior após um ponto ruim.

A seleção pelo mínimo preserva o melhor candidato em vez de promover esse
endpoint ruim. `models/sprint1_validated` conserva exatamente o checkpoint it450
testado fora do painel; `models/sprint1_latest` e o processo continuam evoluindo.
O processo é mantido enquanto íntegro e sem regressão sustentada, com investigação
se houver falha concreta. Não se promete melhoria monotônica de todo checkpoint.

## Estudo controlado curto

`ablation.py` compara C1 corrected/shared, C2 PBRS/shared e C3 PBRS/separate com
clipping separado, duas seeds, 40 iterações ×64 ambientes, mesma origem/currículo,
guard KL e avaliação; warmup zero em todos. Essa ablação não inclui todo o preset
principal e seu orçamento é pequeno. É destinada a estudar direção e componentes,
sem transformar uma média de poucas partidas em um vencedor definitivo.

O treino principal continua separado da ablação, com ator/crítico calibrados e
mais dados. Os resultados completos estão em ABLATION.md. C2 superou C1 nos dois seeds;
C3 foi heterogêneo. C4 calibrado teve mínimos 0,67560/0,72083 (média 0,69821),
contra 0,59762/0,73750 no C3. Não há benefício uniforme entre seeds nem
base para atribuição causal de toda a melhoria principal ao crítico separado.

## Revisão das 10:48 — continuidade e publicação

PID 305794 confirmado com comando da fonte congelada, treino vivo, sem NaNs
na auditoria das curvas. A it1300 teve queda pontual para 0,77453; it1350
recuperou para 0,85336 e it1400 para 0,85105. A política continua oscilando,
sem regressão sustentada neste intervalo. Não se mudou fonte ou receita.

Backup coerente da it1405, melhor it1350, score de seleção 0,8533582242.
Modelo melhor/latest foram comparados tensor a tensor com resume.pt;
contadores, hashes, mapa e crítico separado conferidos. Fonte congelada
conferida e models/model.pt original preservado. Esta publicação inclui
a ablação completa, curvas e bundle; o checkpoint it450 validado continua
preservado separadamente. O orçamento principal ainda não terminou; a
comparação independente de melhor versus endpoint será feita ao finalizar.

Publicação confirmada no GitHub: `337dd69`, branch `gpt`: ablação completa,
curvas auditadas e bundle retomável it1405 / melhor it1350.

## Revisão das 11:15

Processo 305794 vivo, 1750 iterações e métricas finitas. A queda no kickoff
greedy it1600 (mínimo 0,77626) recuperou nas avaliações 1650/1700/1750
(0,84695/0,85367/0,85614). Sem regressão sustentada nesse intervalo.
Receita e fonte congelada mantidas. Backup it1750/best1750 verificado:
meta, pesos finitos, igualdade tensor a tensor, hashes e mapa; original preservado.
A seleção usa esse painel; não é validação independente do novo checkpoint.

## Revisão das 11:45

Processo principal confirmado vivo, >2120 iterações; métricas finitas.
Mínimo it1900: 0,81484; it1950: 0,85327; it2050: 0,86322;
it2100: 0,85302. Oscilações persistem, sem regressão sustentada nesses pontos.
Fonte e receita mantidas. Backup verificado it2125, melhor
it2050, score 0.86322404. Hash, metadados, pesos
e mapa conferidos; modelo original preservado. Esse melhor score é
do painel de seleção, não de validação independente do novo checkpoint.
Publicação anterior confirmada: 745ca46 (backup it1750).

## Revisão das 12:15

PID confirmado, aproximadamente 2500 iterações; auditoria das curvas finita.
Mínimos it2200–2450 entre 0,84931 e 0,86083. Não há regressão
sustentada do critério; as referências continuam oscilando. anchor_1 marcou
0,76718 na 2350, recuperou 0,89558 na 2400 e marcou 0,79188 na 2450
(baseline 0,79257). Não confundir um mínimo agregado saudável com melhora
uniforme por adversário. Fonte/receita mantidas e modelo selecionado preservado.
Backup verificado it2495/melhor it2050, score
0.86322404; hashes, pesos/metas, crítico e mapa coerentes.
Publicação anterior: 2dcd86c (backup it2125). Modelo original preservado.

## Novo orçamento autorizado: 3000

O usuário pediu encerrar em 3000, benchmarks de self-play e bots táticos,
e jogar contra o modelo. Supervisor identificado de parada e worker pós-treino
foram iniciados; schedules originais preservados. Dois novos testes passaram
(parada em endpoint com salvamento e simetria de referencial dos quatro estilos),
mais smoke com partidas completas dos quatro bots nos dois lados. Rearmamento
headless e sintaxe JS passaram após opção POLICY_PATH. Protocolo pré-especificado
em BENCHMARK_PROTOCOL.md; benchmark ainda pendente do endpoint.

## Parada confirmada às 12:54–12:55

Supervisor sinalizou ao observar exatamente it3000; o treinador concluiu a
avaliação em curso e salvou antes de sair. Processo real não está mais vivo,
live status completed, completed_iterations=3000, stop_reason=signal.
98.304.000 amostras na continuação. Checkpoint retomável e pesos/metas/mapa
foram verificados; original preservado. Melhor pré-benchmark it2550
(score 0.86825065); endpoint painel 0.86171749.
Nenhum treino novo foi iniciado. Benchmark pré-especificado está em execução,
sem dados finais ainda; modelos congelados em models/sprint1_3000.
Preparação de gráficos/publicador publicada em c3e5b86.
