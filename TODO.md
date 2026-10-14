# Roadmap

## Prioridade: consolidar 1v1 em futsal-classic

- [ ] Separar execuções por pasta e salvar configuração completa, seeds, versão do código e dependências.
- [ ] Unificar condições de benchmark, torneio e avaliação interna.
- [ ] Avaliar nos dois lados e com várias seeds; distinguir proporção de gols de taxa de vitória.
- [ ] Expandir adversários: Chase determinístico, Chase com ruído, defensor, interceptador e snapshots.
- [ ] Medir contribuições de recompensa por episódio.
- [x] Implementar tabela de três termos (gol ±25, aproximação 0,6, progresso assinado 0,08) e POC pareado.
- [ ] Ampliar comparação com orçamento maior e self-play após o POC curto.
- [ ] Comparar execução greedy e amostrada e inspecionar replays.
- [ ] Definir critérios de aprovação e validar o checkpoint selecionado em sala real.
- [ ] Corrigir rastreabilidade dos metadados após seleção de checkpoint.

## Depois: equipes e fine-tuning

- [ ] Corrigir resets para múltiplos jogadores e validar observações/recompensas.
- [ ] Comparar treino do zero com transferência parcial/fine-tuning do agente 1v1.
- [ ] Começar com 2v2 e avaliar coordenação antes de avançar para 3v3.
- [ ] Adaptar exportador e observações da sala, hoje restritos ao 1v1.

## Manutenção

- [ ] Organizar scripts exploratórios e selecionar verificações reutilizáveis.
- [ ] Salvar estado de otimização e pool se retomada exata de treino for necessária.

A duração de uma execução ou o número de iterações, isoladamente, não é critério de conclusão.

## Correção em gpt

- [x] Reproduzir falha de kickoff e estagnação no canto.
- [x] Corrigir paredes num estádio local e testar colisões.
- [x] Substituir sinais de velocidade por deslocamento real, mantendo pesos ±25 / 0,6 / 0,08.
- [x] Alinhar rearmamento de chute no simulador e na exportação.
- [x] Implementar avaliação com cenários variados, modos greedy/amostrado e bola parada.
- [x] Validar candidato em seeds separadas no simulador e exportar.
- [ ] Melhorar consistência entre seeds e desempenho em posições aleatórias.
- [ ] Validar transferência para sala real.

## Sprint 1 — ablação controlada (2026-10-14)

- [x] Implementar B0/E1 selecionáveis: bootstrap zero ou estado final anterior ao autoreset.
- [x] Instrumentar gradientes por objetivo e avaliação random/kickoff.
- [x] Criar testes de máscaras GAE e captura no ambiente real.
- [x] Disponibilizar runner pareado com checkpoint congelado e pastas novas.
- [x] Executar comparação principal B0/E1 em três seeds e 9.830.400 amostras por braço/seed.
- [x] Auditar execução e avaliar candidatos em cinco seeds novas, greedy/amostrado.
- [x] Concluir continuação B0 até 07h: parada normal às 06:58, 5.236 iterações; conferir endpoint e melhor checkpoint.
- [ ] Implementar e avaliar E2 (PBRS descontado) e E3 (crítico separado) após validar a base.
- [ ] Avaliar currículo estacionário como intervenção adicional.

Contrato e comandos: `experiments/sprint1_timeout/README.md`. Começar esta etapa
não significa concluir a sprint nem comprovar convergência.

## Sprint 1 estrutural — 2026-10-14

- [x] Trocar sinais legados por PBRS descontado com condição absorvente e gamma do GAE.
- [x] Separar crítico/ator, parâmetros e clipping, com migração estrita de checkpoints.
- [x] Calibrar crítico inicialmente sem mover o ator.
- [x] Implementar adversários fixos por episódio e âncoras neurais históricas.
- [x] Avaliar ambos os modos e selecionar pelo mínimo modo×cenário.
- [x] Implementar save durável, SIGTERM/SIGINT e retomada de pesos/Adam/schedules/pool/RNG.
- [x] Iniciar treino principal com o preset estrutural e fontes congeladas.
- [ ] Verificar ganho em orçamento comparável, múltiplas seeds e avaliação independente.
- [ ] Após validar esta base, estudar PFSP e observações enriquecidas (Sprint 2).

Contrato: `experiments/sprint1_structural/README.md`. A implementação pode estar
correta sem garantir melhora empírica; as duas conclusões devem ser documentadas separadamente.
