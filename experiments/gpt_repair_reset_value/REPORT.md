# Correção do bot na branch gpt

## Alterações verificadas

- `maps/futsal_1v1.hbs`: cópia local do estádio com bias das paredes removido. No simulador instalado, a versão original deixava a bola atravessar para cantos externos; o teste da versão corrigida verifica rebote nas duas laterais. Isso altera a geometria de colisão do mapa usado; exportação acompanha o mesmo mapa.
- Recompensas por deslocamento efetivo do jogador e da bola; os pesos continuam gol ±25 / aproximação 0,6 / progresso 0,08. Jogador bloqueado sem mudar de posição recebe zero de aproximação.
- Escala física explícita: 21,12 unidades de movimento livre por decisão com tick skip 8. Evita depender da velocidade máxima incorreta fornecida pelo engine (≈0,119).
- Pressionar chute rearma dentro da janela de decisão. Exportação e bot.js incluem a mesma configuração. Ações da política continuam [3,3,2].
- Ao mudar de escala de recompensa, a cabeça de valor é zerada e a política inicial é preservada. O critic antigo previa aproximadamente 49 a 850 nos resets corrigidos.
- Avaliação separa kickoff/random, greedy/amostrado, adversários e seeds. Mede gols, resultados de episódios completos e bola parada. Timeout correto: 2.000 ticks = 250 decisões.

## Experimentos

Cada seed fez 100 iterações × 64 ambientes × 64 decisões = 409.600 transições, com fine-tuning contra Chase. O treino principal continua suportando self-play. Os checkpoints iniciais são idênticos; condições de avaliação são iguais.

Pontuação abaixo: média por condição da fração de vitórias + metade da fração de empates, em episódios completos, nos dois lados contra Chase com/sem ruído, modo greedy. Cada cenário e seed de avaliação recebe o mesmo peso. Não é uma estimativa com intervalos de confiança: kickoffs determinísticos geram trajetórias correlacionadas.

| Seed de treino | Mantendo critic antigo | Reiniciando critic |
|---|---:|---:|
| 0 | 63.5% | 66.0% |
| 1 | 39.3% | 51.1% |
| 2 | 41.7% | 41.1% |

Baseline no ambiente corrigido: 58.2%.
A seed 0 foi selecionada. A variação entre seeds continua alta; não está demonstrada convergência consistente.

## Validação independente do candidato

Seeds 200/201, 32 ambientes, 750 decisões por condição, dois lados, kickoff/random, Chase com/sem ruído, ações greedy. Essas seeds não foram usadas para selecionar entre candidatos.

| Indicador | Antes do fine-tuning | Candidato seed 0 |
|---|---:|---:|
| Pontuação média por condição | 56.9% | 65.5% |
| Gols pró × contra: kickoff | 245 × 110 | 1901 × 144 |
| Gols pró × contra: random | 577 × 363 | 527 × 471 |

O ganho se concentra no kickoff. Em posições aleatórias o saldo piorou (+214 → +56), embora permaneça positivo. Não se deve afirmar que o candidato melhorou em todas as situações.

## Entrega

O candidato validado foi promovido para `models/model.pt`; `models/meta.json` registra mapa, pesos, rearmamento e validação. `headless-bot/policy.json` foi reexportado. O checkpoint anterior permanece no commit da main e em initial.pt.

Verificações: seis testes Python, smoke test de uma iteração de PPO self-play, compatibilidade do chute em JavaScript, paridade de observações/ações Python/JavaScript em 64 casos nos dois lados, compilação de Python e sintaxe de JavaScript. Não foi hospedada nem validada uma sala real.

## Próximo passo

Antes de aumentar para equipes, melhorar os cenários aleatórios e testar em sala real. Uma nova etapa pode misturar o adversário Chase sem ruído no treino e usar a pontuação balanceada entre cenários para selecionar checkpoints intermediários. Não continuar escolhendo exclusivamente por gols no kickoff.

## Ressalva sobre os gols do kickoff

Uma inspeção posterior do candidato greedy, um ambiente, Chase sem ruído, revelou uma trajetória repetida de 54 decisões: disputa no centro, desvio/rebote lateral, bola avançando ao gol azul e chute do próprio Chase completando o gol contra. Na decisão 53, a bola começa em (375,6; 4,5), Chase em (352,2; −6,6) pressiona chute, e a política vermelha está distante em (290,8; −116,4), sem pressionar chute. O evento final é gol vermelho. A mesma sequência repete após o reset.

Portanto, os resultados de kickoff não medem qualidade geral de finalização: o baseline pode marcar contra por perseguir a bola sem posicionar-se entre bola e gol. Não foi feita uma classificação de todos os gols; este caso prova a limitação, não sua frequência. “Kickoff” descreve a condição inicial, não a localização ou mecanismo de cada gol. A melhora de pontuação está restrita a esses adversários/condições. Adicionar defensor/goalkeeper que não se limite a Chase e classificar gol contra versus finalização antes de chamar o ganho de consolidação.
