# Contexto do projeto

## Histórico registrado

A documentação anterior descreve uma versão em Python com throughput limitado, tentativas de behavioral cloning e recompensas densas exploradas sem produzir gols. Esses relatos motivaram a migração para HaxballGym e PPO self-play. Não foram revalidados nesta revisão.

## Estado atual

O código usa o simulador de HaxballGym, PyTorch e uma política pequena em CPU por padrão. Os registros disponíveis mostram aproximadamente 13–14 mil amostras de jogador por segundo numa execução; esse valor depende da máquina, configuração e custo de avaliação. Não há aqui uma certificação de equivalência bit a bit com o jogo oficial.

`training_stats.json`, `models/meta.json`, `models/meta_final.json` e `poc_results.json` contêm resultados de origens/configurações que não estão identificadas de forma consistente. Não devem ser combinados numa única curva ou tomados como prova de convergência. Metadados também podem mudar durante execuções e seleção.

O experimento registrado em `poc_results.json` apresenta forte queda de gols e de atividade ofensiva ao longo de 75 iterações. Sem configuração completa e replicações, isso é um alerta para investigar a recompensa, não uma conclusão causal.

Não se presume que haja treinamento ativo a partir desses arquivos. O status de execução deve ser consultado no processo ou log correspondente.

## Direção acordada

Consolidar o 1v1 primeiro: várias seeds, adversários diversos, avaliação nos dois lados, inspeção de comportamento e comparação controlada de recompensas. Depois, investigar equipes e transferência/fine-tuning, validando mudanças de observação e resets. Implementação atual e propostas estão separadas em `PROTOCOL.md` e `TODO.md`.

## Trabalho na branch gpt

As mudanças anteriores foram salvas na main em 063a4e5. A branch gpt corrige mapa, sinais de deslocamento, rearmamento do chute e avaliação, e promove um candidato após validação independente no simulador. O relatório atual é `experiments/gpt_repair_reset_value/REPORT.md`. O desempenho melhorou no kickoff, mas a variação entre seeds e a defesa em posições aleatórias ainda exigem trabalho. Sala real não foi testada.
