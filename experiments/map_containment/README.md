# Contenção da bola — variante do mesmo mapa 1v1

O usuário relatou fugas da bola e pediu um mapa o mais próximo possível do atual.
A fuga foi reproduzida no node-haxball usando o estádio embutido na política GPU.
A variante está em maps/futsal_1v1_contained.hbs; o original de treino foi preservado.

Mudanças: bias=-1000 nos12 segmentos de paredes/redis, cuja orientação admite
correção para dentro do campo; dois planos suplementares exclusivos da bola em
y=-200/+200, com o mesmo coeficiente de restituição1 das laterais. A guarda evita
atravessar paredes e escapar pelos cantos quando uma decisão de física termina
além da superfície. O primeiro candidato apenas com bias falhou em16 cenários
extremos de velocidade80; os planos adicionais corrigiram esses casos antes da
variante final ser disponibilizada.

Preservados: dimensões, desenho, gols, postes, raios, massa, amortecimento,
aceleração, força do chute, coeficientes de restituição existentes e spawns.
A colisão nas fronteiras foi alterada intencionalmente: não afirmar que a física
é absolutamente idêntica nem reaproveitar scores antigos como avaliação nova.

Teste: node tests/check_map_containment.cjs experiments/map_containment/audit.json
2160 cenários determinísticos:5 posições×72 direções×6 velocidades,180ticks cada.
Original:22 fugas a velocidade8 e70 a12, por360 casos. Corrigido:zero fugas em
todas as velocidades testadas4/8/12/20/40/80. Os360 casos de velocidade4 tiveram
trajetórias idênticas ponto a ponto. Os dois gols continuam funcionando. Isso não
certifica ausência de qualquer falha fora dessa matriz de testes.

A escala de observação automática permaneceu exatamente igual, assim como os
pesos do ator, repeat_kick e schema16. A política runtime foi criada em
runs/gpu_room_20261015/contained_policy.json, separada do bundle original.
Sala do modelo9900 passou a usar essa variante; host anterior foi encerrado.
Não foi iniciado treino e nenhum checkpoint foi modificado. Para comparar
performance ou treinar novamente, usar explicitamente o mapa novo e manter
referências/controles adequados.
