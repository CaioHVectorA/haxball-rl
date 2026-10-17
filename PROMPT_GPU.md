# Histórico: continuação de 3000 até 10000, já concluída

Este prompt foi usado na etapa GPU anterior. Não o execute como instrução atual: o melhor modelo mais recente está em models/local_warmstart_latest/model.pt; o treino foi encerrado. Para preparar 2v2, consulte experiments/2v2/ROADMAP.md. O texto original é preservado abaixo para rastreabilidade.

---

Continue o projeto https://github.com/CaioHVectorA/haxball-rl, branch gpt, com foco em executar treino e economizar tokens. Leia AGENTS.md; não refaça estudos, benchmarks ou análises já concluídos.

Configure Python e PyTorch com CUDA (o pyproject atual aponta para torch CPU). Confirme que torch.cuda.is_available() é True. Retome models/sprint1_3000/resume.pt, preservando modelo, Adam, RNGs, referências, currículo e schedules. É a iteração 3000; continue até o orçamento original de 10000. Esta instrução autoriza essa continuação e substitui a antiga parada em 3000. Use --resume, não --warm-start.

Execute em pasta NOVA:
python src/train.py --sprint1 --resume models/sprint1_3000/resume.pt --device cuda --out runs/gpu_continuation_TIMESTAMP --checkpoint-every 5 --save-every 50

Substitua TIMESTAMP por data/hora. Preserve 512 ambientes, T64/épocas6/minibatches4, PBRS, crítico separado e todas as avaliações internas existentes. Não altere receita nem arquitetura. Confira rapidamente os primeiros passos e corrija somente falhas impeditivas. Depois mantenha UMA execução até concluir, sem validações extras, consultas históricas ou novas ablações. Salve o bundle com src/backup_training.py ao terminar e publique código/documentação/bundle na branch gpt, sem credenciais, sem force-push e sem substituir models/model.pt original. Comunique apenas início confirmado, falha ou conclusão, em poucas linhas.
