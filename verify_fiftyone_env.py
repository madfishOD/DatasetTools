import sys, importlib
import fiftyone as fo
import fiftyone.brain as fob          # <-- ensures Brain operators register

# list operators (works on recent/older FO)
ops = []
try:
    import fiftyone.operators as fops
    ops = [o.name for o in fops.list_operators()]
except Exception:
    import fiftyone.plugins as fops_old
    ops = [o.name for o in fops_old.list_operators()]

def has(m): return importlib.util.find_spec(m) is not None

print("PYTHON  :", sys.executable)
print("FiftyOne:", fo.__version__)
print("DB DIR  :", fo.config.database_dir)
print("brain   :", has("fiftyone.brain"))
print("torch   :", has("torch"))
print("umap    :", has("umap"))
print("ops     :", sorted(n for n in ops
                         if "brain." in n or "similarity" in n
                         or "visualization" in n or "uniqueness" in n))
