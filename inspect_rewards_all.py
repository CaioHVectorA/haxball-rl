import haxballgym
import inspect
print([attr for attr in dir(haxballgym.reward) if not attr.startswith("_")])
