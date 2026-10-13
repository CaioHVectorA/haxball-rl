import sys
sys.path.insert(0, "src")
import haxballgym
from test_delta_breakaway import DeltaBreakawayReward
import numpy as np

# Vamos testar dois cenários:
# 1. O momento exato em que o jogador corta o zagueiro (drible executado)
# 2. O jogador correndo livre em linha reta já desmarcado (sem drible novo)

db = DeltaBreakawayReward(scale=250.0, mult=3.0)

class FakeState:
    def __init__(self, my_x, opp_x, has_ball=True):
        # Goal at +400
        self.ball_pos = np.array([[my_x, 0.0]])
        self.player_pos = np.array([[[my_x, 0.0], [opp_x, 0.0]]])
        self.team = np.array([[1, 2]])
    def goal_center(self, attacked=True):
        return np.array([[[400.0, 0.0], [-400.0, 0.0]]])

# Passo A -> B: Estava empatado na linha (0px vantagem) e corta abrindo 150px de vantagem!
s_before = FakeState(my_x=100.0, opp_x=100.0) # defensor emparelhado
s_after = FakeState(my_x=200.0, opp_x=50.0)   # ultrapassou o defensor em 150px!

rew_drible = db.get_rewards(s_after, s_before, False, False)
print(f"Cenário 1 (Momento do Drible): Recompensa = +{rew_drible[0, 0]:.2f}")

# Passo B -> C: Já está desmarcado e continua correndo livre (mantém vantagem de 150px)
s_after_more = FakeState(my_x=250.0, opp_x=100.0) # ambos avançaram igual, mantendo 150px
rew_livre = db.get_rewards(s_after_more, s_after, False, False)
print(f"Cenário 2 (Correndo já desmarcado): Recompensa = +{rew_livre[0, 0]:.2f}")
