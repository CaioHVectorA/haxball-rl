# Do melhor 1v1 ao primeiro 2v2

Ponto de partida: `models/local_warmstart_latest/model.pt`, melhor checkpoint da
continuação local, it7300. O endpoint8018 e o resume continuam separados.
O usuário encerrou o treino 1v1; **nenhum treino novo foi iniciado** neste estudo.

## O que podemos aproveitar

O ator já contém controle de movimento, chute e reação à bola. Isso fornece uma
inicialização útil, não garante passes/cobertura. A equipe exige representar um
aliado e dois adversários e distribuir responsabilidades. Copiar dois bots 1v1
pode produzir dois jogadores perseguindo a mesma bola.

Não usar `--sprint1 --red 2 --blue 2`: o CLI rejeita corretamente esse caso.
Remover a rejeição não resolve os contratos abaixo.

## 1. Preparar ambiente e observações antes de treinar

- Usar inicialmente a geometria/física do mapa original do checkpoint; não trocar
  simultaneamente para um campo grande. A variante contained não é validada no
  motor nativo: o probe 1v1 encontrou assimetria grave, documentada em
  `experiments/continuation_20261016`. Sua validação JS não se transfere ao nativo.
- Corrigir `KickoffSetter`: hoje posiciona só índices0/1. Com dois vermelhos e
  dois azuis, deixa um vermelho no outro campo e os dois azuis sobrepostos.
  Posicionar por time, com separação física e condições espelhadas.
- Criar esquema versionado `team24_v1`: as12 entradas básicas atuais, um bloco
  de4 do aliado e dois blocos de4 dos adversários. Referencial de ataque +X
  para ambos os times, mesmas escalas físicas do modelo original.
- O DefaultObs instalado produz24 entradas para4 jogadores, mas ordena os
  outros por índice, sem marcar time. Portanto **não basta aceitar shape24**.
  Organizar semanticamente aliados/adversários, com regra explícita e determinística
  para ordenar os dois adversários. Testar permutações e espelhamento; empates na
  ordenação precisam de contrato. Ordenar por proximidade pode criar descontinuidades.
- Validar parsing das ações, reset/autoreset, terminais, timeouts e observação final
  para os quatro jogadores. Garantir que só os dois aprendizes vermelhos recebem
  gradiente e que os azuis executam o adversário escolhido até o fim do episódio.

Gate: testes de separação/spawn, simetria, slots por time, shapes, máscaras e
métricas corretas por equipe. Sem teste longo nem ajuste de recompensa por score.

## 2. Transferir o ator sem atribuir significado errado às entradas

Escolher um adversário de referência por regra fixa documentada. Com W de dimensão
256x16, construir projeção P (16x24): selecionar as12 entradas básicas e o bloco
correspondente desse adversário. Inicializar W_novo = W P e manter bias, segunda
camada e cabeças. As colunas novas iniciam zeradas. Assim, antes de fine-tuning,
o novo ator produz exatamente os logits do antigo sobre a observação projetada.

Isso **não** resolve a decisão sobre qual oponente atender: é apenas um contrato
verificável de inicialização. A política passa a aprender a usar os novos slots.
Não colocar o aliado no antigo slot do oponente; não simplesmente concatenar zeros
sem conferir semântica. O loader1v1 deve continuar rejeitando shapes incompatíveis.

Gate: logits/ações equivalentes sob a projeção em muitas observações dos dois
lados, tensores finitos, modelo1v1 intacto e nova meta com obs_schema team24_v1.
Comparar depois com um controle treinado do zero, em orçamento igual.

## 3. Primeiro pipeline cooperativo simples

Um ator compartilhado pelos dois companheiros, execução por observação individual.
Começar com crítico independente recebendo a nova observação, para reduzir mudanças
simultâneas. Um crítico centralizado com informação global (linha MAPPO) é a hipótese
seguinte, com comparação controlada; crítico separado não significa centralizado.
Nova dimensão/objetivo exigem novo crítico/Adam e calibração antes de mover o ator.

Adversários e companheiros precisam de generalização própria: adaptar a fábrica de
oponentes, hoje explicitamente1v1 e com saída(N,3), para ações(N,2,3) da equipe azul.
Congelar a identidade da dupla por episódio. Os modelos1v1 podem servir como scripts
iniciais por projeção16, mas não como adversários2v2 plenamente competentes.

Recompensa inicial: gol coletivo±25, compartilhado igualmente como sinal por agente.
Somar estatísticas por equipe apenas uma vez; dois agentes recebendo gol não são
"dois gols". Começar sem bônus ad hoc por passe. Usar um controle esparso e, se
necessário, comparar PBRS de equipe limitado com terminal zero e mesmo gamma.
Copiar o potencial individual atrás da bola para ambos pode incentivar aglomeração.

Um potencial conjunto pode associar um companheiro ao alvo de pressão e outro a
um alvo de apoio/cobertura, escolhendo a menor distância total entre as duas
atribuições. Alvos/escala precisam ser definidos e testados, não escolhidos após
ver score. O teorema de shaping não garante que essa pista acelere PPO.

## 4. Ensinar cooperação em exercícios pequenos

1. **Passe e recepção:** receptor livre e posições diversas, incluindo situações
   em que conduzir é melhor. Registrar toques reais, bola recebida e controle.
2. **2v1:** portador pode conduzir ou passar sob pressão, parceiro varia apoio.
3. **Passe e deslocamento:** procurar nova linha após soltar a bola, com cobertura.
4. **2v2 completo:** duplas de estilos diferentes, troca de lados e parceiros.

São tarefas auxiliares diferentes, não medições automaticamente comparáveis a
partidas completas. Fixar critérios de progressão antes dos resultados e manter
alguma prática das etapas anteriores. Não declarar passe pela mera proximidade
ou pressão do botão: o engine deve fornecer/instrumentar evento de contato.

## 5. Como provar que surgiu jogo em equipe

Benchmark reservado, greedy/amostrado, lados e seeds explícitas, primeira partida
completa e intervalos por seed. Não contar os dois agentes como partidas independentes.

Comparar contra: dois bots1v1 independentes; dupla perseguidor/goleiro; dupla com
pressão/cobertura; versões anteriores; parceiros não vistos. Incluir ablação em
que o aliado fica parado ou troca de política: sucesso dependente do parceiro,
com posições variadas, ajuda a distinguir coordenação de exploração de um script.

Medir vitória/empate, gols sofridos, perdas após passe, recepções reais, gols após
passe, espaço entre companheiros e cobertura. Métricas auxiliares descrevem ações;
não substituem a tarefa nem identificam intenção. Avaliar decisões de não passar.
Depois testar com humano como parceiro e como adversário.

## 6. Exportação e implementação incremental

O exportador e `headless-bot/bot.js` atuais aceitam16 entradas/1v1. Criar suporte
explícito ao novo schema, slots e controle dos dois bots antes de teste humano2v2.
Exigir paridade JS/PyTorch em observações reais dos dois times e casos de IDs
reordenados. Não truncar observações para fazer o modelo caber.

Primeira entrega de engenharia: resets+TeamObs24+transferência testada e um smoke
curto sem pretensão de aprender cooperação. Só depois autorizar orçamento de treino.
Mudanças de mapa, recompensa, arquitetura e currículo entram separadamente, com
baseline que recebe o mesmo tratamento de reinicialização/calibração.

## Critério prático

Estamos prontos para desenvolver o ambiente2v2, não para apenas multiplicar
jogadores no comando do treino. O checkpoint forte1v1 reduz o trabalho de aprender
controle básico; coordenação e validação em equipe continuam sendo trabalho novo.
