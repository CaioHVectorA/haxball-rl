# Arquitetura dos agentes

Este documento descreve o código atual. O 1v1 consolidado é a base da adaptação experimental 2v2; comportamento cooperativo ainda precisa de validação em partidas.

## Política e observações

`src/model.py` implementa `Policy`: ator MLP de duas camadas 256/Tanh e três cabeças categóricas (X, Y, chute). O modo legado compartilha tronco com o valor. `--sprint1` usa crítico independente 256/Tanh, clipping separado e 20 iterações iniciais só do crítico. A inicialização padrão é preservada; `Policy` também possui uma opção ortogonal explícita, não habilitada pelo treino atual. Checkpoints legados mantêm nomes e formato do ator; loaders validam shapes e migram o crítico explicitamente.

O ambiente usa `DefaultObs(pos_coef="auto")`, de HaxballGym. O mapa padrão é `maps/futsal_1v1.hbs`, uma cópia local de futsal-classic com paredes sem bias e damping explícito. Os checkpoints 1v1 disponíveis têm 16 entradas. A dimensão e o formato das observações dependem do builder e do número de jogadores; a definição exata está na dependência instalada.

As ações são bins `[0, 1, 2]` para movimento (−1, 0, +1) e `[0, 1]` para chute (soltar, pressionar). `RepeatKickAction` converte pressão em input 2 do engine, rearmando dentro da janela de ticks; as cabeças da política continuam com duas opções. O exportador grava `repeat_kick`, e a sala rearma entre decisões quando esse campo está ativo. O ambiente converte o eixo X do time azul para o referencial da política.

`NeuralBot` carrega `model.pt` e `meta.json`, com ações greedy por padrão. Treinamento, avaliação interna e torneio amostram ações; essa diferença deve ser considerada na validação.

## Baselines

- `ChaseBot`: persegue a posição atual da bola, adiciona ruído angular (`jiggle=0.25`) e pressiona chute quando a distância é menor que 34. Não prevê interceptação. Uma seed fixa torna a sequência de ruído reproduzível.
- `RandomBot`: ações uniformes, útil como controle básico.

## Sprint 1 estrutural

`--sprint1` troca a recompensa por PBRS descontado: potencial de posicionamento atrás da bola (peso 0,15), potencial antissimétrico de progresso da bola (0,08) e gol ±25. Gamma é igual ao GAE; gols têm potencial absorvente zero e timeouts artificiais usam valor da observação antes do autoreset. Sem o preset, a tabela padrão permanece `corrected`, para reprodução.

O currículo padrão do preset é 100% de adversários congelados, escolhidos por ambiente/episódio: heurísticas e referências neurais históricas. Essas âncoras tornam o esquecimento neural mensurável e treinável. Avaliação inclui greedy/amostrado e seleção pelo mínimo de modo×cenário, com dados por adversário/lado.

`resume.pt` preserva modelo, Adam, schedules, melhor checkpoint, referências, pool e RNGs. As partidas do engine recomeçam; não existe retomada física bit a bit. Saves atômicos são periódicos e feitos ao encerrar por SIGTERM/SIGINT. Consulte `experiments/sprint1_structural/README.md` e `REPORT.md` para contrato e comandos.

## Self-play legado / modo misto

Fora do currículo estacionário, a política controla todos os jogadores vermelhos com compartilhamento de parâmetros. O azul usa uma política congelada. A cada rollout de 64 decisões, escolhe-se um único adversário para todos os ambientes: snapshot aleatório com probabilidade 0.5 quando o pool existe; caso contrário, cópia atual do learner.

Um snapshot é adicionado a cada 20 iterações; o pool contém no máximo 40. A escolha não ocorre por episódio e pode mudar durante uma partida. O pool reduz a exposição exclusiva ao adversário mais recente, mas não garante progresso monotônico nem elimina ciclos.

## Adaptação 2v2

`--team2v2` usa duas equipes de dois jogadores, `team24_v1`, crítico separado, PBRS e snapshot self-play sem currículo 1v1. Observações: base própria/bola/gols (12), adversário mais próximo (4), companheiro (4), segundo adversário (4). X é espelhado no azul. Os quatro jogadores têm kickoff separado e resets aleatórios sem sobreposição.

`--transfer-1v1 --warm-start models/sprint1_10000/model.pt` preserva as primeiras 16 colunas do ator e zera as oito novas, mantendo as camadas aprendidas. Crítico e Adam são novos; há 20 iterações iniciais só do crítico. Retomadas posteriores usam `--resume`, sem repetir a transferência. A recompensa de posicionamento é compartilhada pelo time e usa o melhor colocado; progresso e gols também são compartilhados. Não há garantia de passes ou papéis. Avaliação controla equipes completas nos dois lados. Exportador e sala suportam dois bots contra dois humanos, com paridade Python/JS testada; jogo humano real ainda precisa de validação. Veja `experiments/team2v2/README.md`. 3v3 não está habilitado.
