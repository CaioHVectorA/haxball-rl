import haxballgym
import inspect

for cls_name in ["TouchReward", "PowerShotReward"]:
    cls = getattr(haxballgym.reward, cls_name)
    print(f"=== {cls_name} ===")
    print(inspect.getsource(cls))
