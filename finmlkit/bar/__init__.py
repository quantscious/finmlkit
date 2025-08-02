# allow: import finmlkit as fmk; fmk.bar.TimeBarKit(...)
import types

def __getattr__(name: str):
    from . import kit as _kit  # import only on first access
    if hasattr(_kit, name):
        obj = getattr(_kit, name)
        globals()[name] = obj   # cache
        return obj
    raise AttributeError(name)

# (optional) for IDEs/autocomplete
__all__ = []  # keep empty to avoid Sphinx re-export noise