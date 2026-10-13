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
