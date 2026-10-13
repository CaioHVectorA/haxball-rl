# Benchmark CPU: 128, 256 ou 512 ambientes

CPU local: AMD Ryzen 7 5700U, 8 núcleos / 16 threads lógicas. RAM: cerca de 30 GiB total e 19 GiB disponíveis durante a leitura.

Foram medidos rollouts de self-play neural mais GAE e atualização PPO completa: 64 decisões, 6 épocas, 4 minibatches. Cada processo teve duas iterações de aquecimento e seis medidas. Teste em CPU, mapa corrigido, PyTorch CPU. Aplicações do desktop permaneceram abertas. Checkpoints e avaliação periódica estão excluídos.

| Ambientes | Threads PyTorch | Amostras/s | Segundos/iteração | Pico RAM do processo | Repetições |
|---|---:|---:|---:|---:|---:|
| 128 | 2 | 10028 | 0.817 | 337 MiB | 2 |
| 256 | 2 | 13065 | 1.254 | 366 MiB | 2 |
| 512 | 2 | 13260 | 2.471 | 416 MiB | 2 |
| 512 | 4 | 12908 | 2.539 | 455 MiB | 1 |
| 512 | 8 | 14067 | 2.329 | 409 MiB | 1 |

512 com duas threads processou 32.2% mais amostras/s que 128. Ficou próximo de 256. A maior medição isolada foi 512 com oito threads; a diferença para as repetições com duas threads é pequena frente à variabilidade do desktop.

512 ambientes não criam 512 threads nem 512 processos: são estados de partidas vetorizados, distribuídos pelo runtime de simulação. Cada iteração coleta 512 × 64 = 32.768 amostras do learner. Mil iterações dão 32.768.000 amostras, quatro vezes o orçamento de mil iterações com 128.
Extrapolando o teste com duas threads, mil iterações de 512 levariam cerca de 41.2 minutos só de coleta/atualização. Avaliação, gravação, aquecimento prolongado e outras tarefas podem aumentar esse tempo. Não é garantia de duração sustentada nem de convergência.

Conclusão: 512 cabe com folga na RAM e é uma configuração razoável nesta CPU. Não é quatro vezes mais rápido que 128; a vantagem medida é de throughput, e o batch maior também muda a dinâmica de otimização. Pode-se manter duas threads como configuração conservadora ou experimentar oito, que teve a maior medição curta.

Reprodução: `uv run src/benchmark_cpu.py --out experiments/outro_benchmark`.
