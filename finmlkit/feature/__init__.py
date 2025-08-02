import importlib

# Map public names to where they actually live
_lazy = {
    "MIMOTransform": (".base", "MIMOTransform"),
    "SIMOTransform": (".base", "SIMOTransform"),
    "MISOTransform": (".base", "MISOTransform"),
    "SISOTransform": (".base", "SISOTransform"),
    "FeatureKit":   (".kit",  "FeatureKit"),
    "Feature": (".kit", "Feature"),
    "Compose": (".kit", "Compose"),
    "VolumePro": (".core.volume", "VolumePro"),
}

def __getattr__(name: str):
    try:
        mod_name, attr = _lazy[name]
    except KeyError as e:
        raise AttributeError(name) from e
    mod = importlib.import_module(mod_name, __name__)
    obj = getattr(mod, attr)
    globals()[name] = obj  # cache for next time
    return obj
