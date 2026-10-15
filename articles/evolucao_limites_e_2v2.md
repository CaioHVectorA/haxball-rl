# De perseguir a bola a começar a jogar: evolução, limites e caminho para o 2v2

**Nota técnica acessível — 15/10/2026.** Base: `origin/main` em `9ea2497`, treino GPU até 10.000 iterações, checkpoint selecionado na 9.900 e código local. Esta é uma análise rápida de evidências e hipóteses, não um novo treinamento.

## 1. O que começou a dar certo

Um agente aprende tentando ações e recebendo sinais sobre suas consequências. Ele pode aprender a marcar gols, mas também pode ficar bom em satisfazer sinais auxiliares sem jogar como esperamos. Por isso precisamos cuidar tanto das recompensas quanto dos adversários e da avaliação.

| Aspecto | `main` inspecionada | Sprint 1 usada no treino atual |
|---|---|---|
| Recompensa | Padrão da dependência, com aproximação/progresso/gol | PBRS de posição atrás da bola e progresso antissimétrico, mais gol ±25 |
| Ator e crítico | Compartilham as mesmas camadas | Camadas separadas, clipping separado e calibração inicial do crítico |
| Adversário | Atualizado por rollout; política atual ou snapshot histórico | Escolhido por episódio entre heurísticas e referências neurais congeladas |
| Timeout | Tratado junto com término para zerar o bootstrap | Corte artificial usa valor da observação antes do reset |
| Horizonte | Meia-vida de desconto de 5 segundos | Meia-vida de 12 segundos |
| Atualização | Até 4 épocas | Até 6 épocas, com guarda de KL |
| Escolha do modelo | Avaliação contra Chase | Dois modos, dois cenários, referências neurais e critério do pior grupo |
| Continuidade | Pesos e snapshots | Checkpoint com Adam, schedules, referências, pool e RNGs |

**Tradução:** melhoramos os sinais, separamos duas tarefas que disputavam parâmetros e passamos a medir mais formas de fracassar. O crítico estima quanto uma situação promete; o ator escolhe ações. Separá-los impede interferência direta no mesmo tronco, embora o ator ainda dependa das estimativas do crítico.

No PBRS, o incentivo auxiliar é `gamma × potencial seguinte − potencial atual`. Ele orienta posições úteis sem acrescentar um objetivo independente de gols, sob as condições do método. A invariância é relativa ao objetivo esparso de gols, não à tabela antiga. Aproximação neural e treinamento finito continuam sujeitos a falhas. [Ng, Harada e Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf). A distinção entre término e limite artificial de tempo segue [Pardo et al.](https://proceedings.mlr.press/v80/pardo18a.html).

**Limite da comparação:** inspecionamos diferenças de código com a `main`; não executamos agora um confronto causal com ambas retreinadas do zero sob orçamento igual. O checkpoint chamado “origem” nos benchmarks é o início da nossa continuação, não um novo treino limpo da `main`. Não sabemos atribuir o ganho a uma única mudança. A ablação curta anterior também teve resultados mistos.

## 2. Há evolução, mas ela não é linear

No painel conhecido, a média do score mínimo dos quatro grupos foi:

| Iterações | Média |
|---|---:|
| 3.000–4.000 | 0,86591 |
| 5.000–6.000 | 0,86609 |
| 7.000–8.000 | 0,87095 |
| 9.000–10.000 | 0,88334 |

O melhor checkpoint foi o 9.900, com 0,89644; o endpoint 10.000 marcou 0,88621. Score significa vitória mais metade do empate, não taxa de vitória. São resultados do painel de seleção; ainda precisamos validar o modelo GPU contra adversários/seeds inéditos antes de afirmar generalização.

Já houve evidência independente de melhora do modelo anterior: o checkpoint 2.550 superou a origem e quatro estilos táticos fora do currículo. Isso sustenta que houve aprendizado útil, mas não certifica planejamento de dribles, força humana ou robustez entre várias seeds de treinamento.

## 3. O que pode estar desacelerando o aprendizado

| Possível limite | Evidência atual | Interpretação e teste barato |
|---|---|---|
| **Desafios pouco renovados** | Currículo mantém 65% de heurísticas e 35% de referências fixas | Pode faltar pressão para aprender novas respostas. Comparar a receita atual com uma mistura pequena de referências recentes, preservando âncoras antigas. Hipótese, não diagnóstico provado. |
| **Passos menores no fim** | LR média: 0,000230 → 0,000110; KL: 0,00318 → 0,00190 | A política muda menos por atualização. Isso favorece estabilidade, mas pode reduzir progresso. Testar um novo schedule em controle pareado, sem aumentar agressividade às cegas. |
| **Menor exploração** | Entropia média: 1,350 → 1,209 | Há menos variedade de ações; não demonstra colapso, pois a entropia não está próxima de zero. Testar exploração isoladamente se as falhas persistirem. |
| **Observação e memória limitadas** | MLP sem histórico; 16 entradas no 1v1 | Não recebe explicitamente histórico de ações ou estado interno de rearme do chute. Pode haver situações semelhantes que exigem respostas distintas. Primeiro construir casos concretos de falha; depois comparar observações ou memória. |
| **Crédito de ações demoradas** | Rollout de 64 decisões ≈8,53 s; GAE com lambda 0,95 | O crítico precisa ajudar a atribuir mérito a ações cujo efeito demora. Isso não é um limite rígido de 8,53 s: existe bootstrap. Relevante para passes e construção de jogadas. |
| **Seleção em painel conhecido** | Muitos checkpoints avaliados no mesmo painel | Podemos melhorar para o teste conhecido. Reservar adversários e seeds para avaliação final; repetições de kickoff greedy não são novas evidências independentes. |

A guarda de KL **não parece ser o freio atual**: não houve parada por esse limite nas janelas de mil iterações iniciais e finais examinadas. A variância explicada do crítico aumentou de 0,844 para 0,871, mas isso não prova que ele acerte as situações difíceis.

Também há um limite de **velocidade**, diferente do limite de aprendizado. Nas últimas mil iterações GPU, coleta levou em média 1,10 s e atualização 0,34 s. A coleta inclui simulação, inferência e transferências CPU/GPU; uma GPU ainda maior não acelera tudo. A comparação com a máquina anterior não isola hardware, pois também mudaram CPU e sistema.

### O mapa corrigido exige uma nova referência

A variante de contenção conserva dimensões e parâmetros, mas altera colisões nas fronteiras. Os testes reproduziram fugas no mapa antigo e não observaram fugas nos 2.160 cenários da variante. Isso é correção funcional, não comprovação de desempenho do bot. Todas as próximas comparações devem usar o mesmo mapa corrigido; não misturar seus scores com os antigos.

**Prioridade sugerida:** medir o checkpoint 9.900 no mapa corrigido; então comparar continuação da receita atual com apenas uma mudança de desafio, exploração ou potencial. Em ablações por warm start, reiniciar Adam e calibrar o crítico igualmente no controle. Recompensa acumulada de tabelas diferentes não serve, sozinha, para escolher a melhor.

## 4. Primeiro estudo para 2v2: passar e oferecer opção de passe

Há espaço para aproveitar o ator 1v1, mas o 2v2 exige enxergar o parceiro, distribuir responsabilidades e evitar que os dois ataquem a mesma bola.

**Bloqueios verificados no código instalado:**

- A observação passa de **16 para 24 entradas**: 12 próprias/bola/gols e quatro por outro jogador.
- O reset atual em 2v2 produz vermelhos em `(-200,0)` e `(200,0)` e os dois azuis sobrepostos em `(0,0)`. Deve ser corrigido antes de qualquer treino.
- Os outros jogadores aparecem por índice, sem campo explícito de time. O slot de aliado muda entre vermelho e azul: precisamos organizar aliado/adversários consistentemente e testar troca de lados/IDs.
- O preset e a exportação da sala atual são validados para 1v1 e bloqueiam uso incompatível. Ainda não existe deployment 2v2 pronto.

Transferir o ator exige mapear o significado das colunas da primeira camada. **Copiar 16 colunas e adicionar zeros pode colocar pesos de “adversário” sobre o slot de “aliado”.** Parte das camadas e cabeças pode ser reaproveitada; o carregador atual rejeita a dimensão diferente corretamente. Comparar transferência explícita com início do zero será necessário.

### Experimento inicial proposto

1. **Passe sem adversário ativo:** um jogador tem a bola, o parceiro está livre em posição sorteada. Primeiro usar um receptor roteirizado; depois permitir que ambos aprendam.
2. **Passe sob pressão em 2v1:** variar posição do defensor e incluir casos em que conduzir ou finalizar é melhor que passar. Medir escolha adequada, recepção e manutenção da posse.
3. **Passe e deslocamento:** após soltar a bola, o primeiro jogador oferece uma nova linha de passe. Variar lados e papéis para evitar decorar uma jogada.
4. **2v2 completo:** duplas congeladas com estilos distintos e parceiros variados; comparar gols, perdas de posse, cobertura e decisões em situações inéditas.

Um passe completo deve ser identificado por **toque real de A seguido de controle por B**, sem toque adversário intermediário, com janela temporal e deslocamento mínimo definidos. Distância pequena à bola não comprova toque; o `GameState` atual não fornece esse evento diretamente, portanto será preciso instrumentá-lo. Inicialmente essa definição serve como **métrica**, evitando premiar trocas inúteis apenas para acumular bônus.

Uma base possível é ator compartilhado entre companheiros, recompensa coletiva de gols e crítico que vê o estado global durante o treino. Cada jogador continua decidindo pela sua observação na execução. É a ideia de treinamento centralizado e execução descentralizada estudada em [MAPPO](https://arxiv.org/abs/2103.01955); não é garantia de funcionar aqui.

Também precisamos de apoio sem bola: repetir para ambos o incentivo individual de ficar atrás da bola pode orientar a exploração para aglomeração. Se houver shaping de equipe, estudar um potencial conjunto que favoreça um pressionador e um jogador de apoio/cobertura. Não afirmar que o potencial individual torna cooperação teoricamente impossível; o risco prático deve ser medido.

## 5. Conclusão de engenharia

Temos uma base 1v1 útil e ainda em evolução. Não há evidência de teto definitivo. Os candidatos mais relevantes para investigação são desafio do currículo, schedules e exploração, observações e confiabilidade da avaliação. Para 2v2, o próximo avanço é corrigir os contratos e demonstrar recepção/apoio em cenários simples antes de gastar um treino longo.

**Nenhum novo treino foi iniciado neste estudo.** Evidências locais: `experiments/sprint1_structural/gpu_hourly`, `benchmark_3000`, `ABLATION.md`, `experiments/map_containment` e inspeção da dependência instalada. A origem da `main` e as métricas ficam registradas para evitar atribuir causalidade que os testes ainda não demonstraram.

Dados numéricos desta análise: [evolucao_limites_e_2v2_evidence.json](evolucao_limites_e_2v2_evidence.json).
