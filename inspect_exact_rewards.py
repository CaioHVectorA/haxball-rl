import sys
sys.path.insert(0, "src")
import haxballgym
from train import create_training_env

env = create_training_env(1, 1, 1, "futsal-classic", 8)
print("="*60)
print("TABELA OFICIAL DE RECOMPENSAS ATIVA NO TREINAMENTO")
print("="*60)
for fn, weight in env.reward_fn.terms:
    print(f"• Termo: {fn.__class__.__name__:<25} | Peso (Multiplicador): {weight}")
    for attr in dir(fn):
        if not attr.startswith("_") and not callable(getattr(fn, attr)):
            print(f"    - {attr} = {getattr(fn, attr)}")
print("="*60)
