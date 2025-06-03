import importlib.util

spec = importlib.util.find_spec("torch")
if spec is None:
    print("Torch not found")
else:
    print(f"Torch location: {spec.origin}")
