# expose subpackages lazily: finmlkit.bar, finmlkit.feature, ...
import importlib, types

__all__ = ["bar", "feature", "label", "sampling"]

def __getattr__(name: str) -> types.ModuleType:
    if name in __all__:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(name)
