# Validação e continuação limitada à madrugada

Esta pasta registra análise, visualizações e ferramentas da revisão de 14/10/2026.
`REPORT.md` distingue integridade dos seis braços de melhoria de desempenho.

- `diagnose.py`: curvas e diferenças por adversário/lado a partir da comparação congelada.
- `heldout.py`: avaliação de três candidatos em novas seeds; critérios fixados no config.
- `launch_overnight.py`: uma continuação a partir de candidato aprovado, em pasta nova,
  com orçamento estimado pelo tempo por iteração medido no braço correspondente.
- `deadline_supervisor.py`: limita o grupo de processos ao prazo absoluto.

As alterações no painel permitem seeds e modo de ação explícitos, mantendo defaults
anteriores. O treinador recebe `--stop-at` com timezone e reserva padrão de 120 s para
avaliação/salvamento. A opção de reserva curta existe para verificar a parada; a
continuação usa a reserva padrão. Ambos os testes do prazo foram executados:
`runs/sprint1_deadline_test_20261014` e `runs/sprint1_supervisor_kill_test_20261014`.

A continuação conserva arquitetura, recompensa e modo de timeout do candidato.
`--keep-critic` conserva pesos de ator/crítico; warm-start não recupera Adam,
rollout, pool ou schedules antigos. Estender o treinamento é exploração adicional,
sem garantia de que mais iterações melhorem o score.

Artefatos de avaliação: `runs/sprint1_overnight_validation_20261014`.
Artefatos de continuação, se o filtro aprovar:
`runs/sprint1_overnight_model_20261014`. `launch.json` registra configuração e origem;
`supervision/supervisor.json` registra processo e limite;
`train/live.json` registra progresso. Checkpoints não são promovidos automaticamente.
