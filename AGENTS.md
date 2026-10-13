# Arquitetura dos agentes

Este documento descreve o código atual. A prioridade é o 1v1; comportamento cooperativo em equipes ainda precisa de validação.

## Política e observações

`src/model.py` implementa `Policy`: MLP com duas camadas de 256 unidades e Tanh, três cabeças categóricas independentes (X, Y, chute) e uma cabeça de valor. Usa a inicialização padrão de `torch.nn.Linear`; não há inicialização ortogonal explícita.

O ambiente usa `DefaultObs(pos_coef="auto")`, de HaxballGym. O mapa padrão é `maps/futsal_1v1.hbs`, uma cópia local de futsal-classic com paredes sem bias e damping explícito. Os checkpoints 1v1 disponíveis têm 16 entradas. A dimensão e o formato das observações dependem do builder e do número de jogadores; a definição exata está na dependência instalada.

As ações são bins `[0, 1, 2]` para movimento (−1, 0, +1) e `[0, 1]` para chute (soltar, pressionar). `RepeatKickAction` converte pressão em input 2 do engine, rearmando dentro da janela de ticks; as cabeças da política continuam com duas opções. O exportador grava `repeat_kick`, e a sala rearma entre decisões quando esse campo está ativo. O ambiente converte o eixo X do time azul para o referencial da política.

`NeuralBot` carrega `model.pt` e `meta.json`, com ações greedy por padrão. Treinamento, avaliação interna e torneio amostram ações; essa diferença deve ser considerada na validação.

## Baselines

- `ChaseBot`: persegue a posição atual da bola, adiciona ruído angular (`jiggle=0.25`) e pressiona chute quando a distância é menor que 34. Não prevê interceptação. Uma seed fixa torna a sequência de ruído reproduzível.
- `RandomBot`: ações uniformes, útil como controle básico.

## Self-play

A política controla todos os jogadores vermelhos com compartilhamento de parâmetros. O azul usa uma política congelada. A cada rollout de 64 decisões, escolhe-se um único adversário para todos os ambientes: snapshot aleatório com probabilidade 0.5 quando o pool existe; caso contrário, cópia atual do learner.

Um snapshot é adicionado a cada 20 iterações; o pool contém no máximo 40. A escolha não ocorre por episódio e pode mudar durante uma partida. O pool reduz a exposição exclusiva ao adversário mais recente, mas não garante progresso monotônico nem elimina ciclos.

## Equipes: etapa futura

A coleta já compartilha parâmetros entre jogadores vermelhos, mas o `KickoffSetter` atual só posiciona explicitamente dois jogadores. Observações, resets, recompensas e exportação devem ser validados antes de treinar 2v2/3v3. O gol recebe peso 25 no treinamento atual; não há garantia de que passes, cobertura ou papéis surjam.

Após consolidar o 1v1, comparar treinamento do zero com transferência parcial/fine-tuning. Mudanças na dimensão da observação podem impedir o carregamento direto dos pesos.
