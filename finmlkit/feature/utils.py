from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import importlib
import inspect
import numpy as np
from finmlkit.utils.log import get_logger

from .base import (
    BaseTransform,
    MinMaxOpTransform,
    BinaryOpTransform,
    ConstantOpTransform,
    UnaryOpTransform,
)

logger = get_logger(__name__)

# --- Serialization helpers ----------------------------------------------------

def _serialize_value(val: Any) -> Any:
    """Best-effort JSON-serializable conversion for common types used in transforms."""
    try:
        import pandas as pd  # local import to avoid circulars when packaging
    except Exception:  # pragma: no cover - defensive
        pd = None
    if pd is not None and isinstance(val, pd.Timedelta):
        return {"__timedelta__": True, "seconds": val.total_seconds()}
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    # Fallback to string repr
    return str(val)


def _deserialize_value(val: Any) -> Any:
    try:
        import pandas as pd
    except Exception:  # pragma: no cover
        pd = None
    if isinstance(val, dict) and val.get("__timedelta__") and pd is not None:
        return pd.Timedelta(seconds=val["seconds"])  # type: ignore[arg-type]
    if isinstance(val, list):
        return [_deserialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _deserialize_value(v) for k, v in val.items()}
    return val


def _class_path(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else obj.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _import_class(path: str):
    module_name, cls_name = path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, cls_name)


# Map op_name -> functions for reconstruction
_OP_BINARY = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "rsub": lambda x, y: y - x,
    "rdiv": lambda x, y: y / x,
}

_OP_MINMAX = {
    "min": lambda x, y: np.minimum(x, y),
    "max": lambda x, y: np.maximum(x, y),
}

_OP_UNARY = {
    "abs": lambda x: x.abs(),
    "log": lambda x: x.apply(lambda v: np.log(v) if v > 0 else np.nan),
    "log1p": lambda x: x.apply(lambda v: np.log1p(v) if v >= 0 else np.nan),
    "exp": lambda x: x.apply(np.exp),
    "square": lambda x: x ** 2,
    "sqrt": lambda x: x.apply(lambda v: np.sqrt(v) if v >= 0 else np.nan),
}


def _maybe_unary_from_name(name: str):
    # Dynamic clip pattern: clip_<lower>_<upper>, where bounds may be empty
    if name.startswith("clip_"):
        parts = name.split("_")
        try:
            lower = float(parts[1]) if parts[1] != "" else None
        except Exception:
            lower = None
        try:
            upper = float(parts[2]) if len(parts) > 2 and parts[2] != "" else None
        except Exception:
            upper = None
        return lambda x: x.clip(lower=lower, upper=upper)

    # Rolling/statistical patterns
    try:
        if name.startswith("rmean"):
            n = int(name[len("rmean"):])
            return lambda x: x.rolling(window=n).mean()
        if name.startswith("rstd"):
            n = int(name[len("rstd"):])
            return lambda x: x.rolling(window=n).std()
        if name.startswith("rsum") or name.startswith("tsum"):
            base = "rsum" if name.startswith("rsum") else "tsum"
            n = int(name[len(base):])
            return lambda x: x.rolling(window=n).sum()
        if name.startswith("ema"):
            n = int(name[len("ema"):])
            # Default to adjust=True (matches Feature.ema default)
            return lambda x: x.ewm(span=n, adjust=True).mean()
        if name.startswith("lag"):
            n = int(name[len("lag"):])
            return lambda x: x.shift(n)
    except Exception:
        # Fall through to static map if parsing fails
        pass

    # Static unary map (abs, log, log1p, exp, square, sqrt, ...)
    return _OP_UNARY.get(name)


# --- Transform serialization/deserialization ----------------------------------

def transform_to_config(t: BaseTransform) -> Dict[str, Any]:
    """Serialize a BaseTransform (including operation and Compose transforms) into a dict."""
    cfg: Dict[str, Any] = {
        "class": _class_path(t),
        "requires": list(getattr(t, "requires", [])),
        "produces": list(getattr(t, "produces", [])),
    }
    # Operation transforms
    if isinstance(t, BinaryOpTransform):
        cfg.update({
            "kind": "binary",
        })
        # Prefer explicit attribute when available; fallback to parsing from produces
        op_name = getattr(t, "op_name", None)
        if not op_name:
            name = t.produces[0] if isinstance(t.produces, list) else t.produces
            op_name = name.split("(")[0]
        cfg["op_name"] = op_name
        cfg["left"] = transform_to_config(t.left)
        cfg["right"] = transform_to_config(t.right)
        return cfg
    if isinstance(t, MinMaxOpTransform):
        cfg.update({"kind": "minmax"})
        op_name = getattr(t, "op_name", None)
        if not op_name:
            name = t.produces[0] if isinstance(t.produces, list) else t.produces
            op_name = name.split("(")[0]
        cfg["op_name"] = op_name
        cfg["left"] = transform_to_config(t.left)
        cfg["right"] = transform_to_config(t.right)
        return cfg
    if isinstance(t, ConstantOpTransform):
        cfg.update({"kind": "const"})
        op_name = getattr(t, "op_name", None)
        if not op_name:
            name = t.produces[0] if isinstance(t.produces, list) else t.produces
            op_name = name.split("(")[0]
        cfg["op_name"] = op_name
        cfg["constant"] = t.constant
        cfg["child"] = transform_to_config(t.transform)
        return cfg
    if isinstance(t, UnaryOpTransform):
        cfg.update({"kind": "unary"})
        op_name = getattr(t, "op_name", None)
        if not op_name:
            name = t.produces[0] if isinstance(t.produces, list) else t.produces
            op_name = name.split("(")[0]
        cfg["op_name"] = op_name
        cfg["child"] = transform_to_config(t.transform)
        return cfg
    # External function transform
    if getattr(t, "_is_external_function", False):
        cfg.update({
            "kind": "external",
            "func": getattr(t, "func_path", None),
            "args": _serialize_value(getattr(t, "args", [])),
            "kwargs": _serialize_value(getattr(t, "kwargs", {})),
            "pass_numpy": bool(getattr(t, "pass_numpy", False)),
        })
        return cfg

    # Compose detection: avoid importing Compose to prevent circular imports
    if hasattr(t, "transforms") and isinstance(getattr(t, "transforms", None), (list, tuple)):
        cfg.update({
            "kind": "compose",
            "steps": [transform_to_config(tt) for tt in getattr(t, "transforms")],
        })
        return cfg

    # Generic: capture constructor parameters by signature names found on instance
    try:
        cls = t.__class__
        sig = inspect.signature(cls.__init__)
        params = {}
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            if name == "input_col" and hasattr(t, "requires"):
                params[name] = t.requires[0]
            elif name == "input_cols" and hasattr(t, "requires"):
                params[name] = list(t.requires)
            elif hasattr(t, name):
                params[name] = getattr(t, name)
        cfg["kind"] = "transform"
        cfg["params"] = {k: _serialize_value(v) for k, v in params.items()}
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to introspect params for {t}: {e}")
        cfg["kind"] = "transform"
        cfg["params"] = {}
    return cfg


def transform_from_config(cfg: Dict[str, Any]) -> BaseTransform:
    kind = cfg.get("kind")
    if kind == "binary":
        left = transform_from_config(cfg["left"])  # type: ignore[index]
        right = transform_from_config(cfg["right"])  # type: ignore[index]
        op_name = cfg.get("op_name", "add")
        op = _OP_BINARY.get(op_name)
        if op is None:
            raise ValueError(f"Unsupported binary op: {op_name}")
        return BinaryOpTransform(left, right, op_name, op)
    if kind == "minmax":
        left = transform_from_config(cfg["left"])  # type: ignore[index]
        right = transform_from_config(cfg["right"])  # type: ignore[index]
        op_name = cfg.get("op_name", "min")
        op = _OP_MINMAX.get(op_name)
        if op is None:
            raise ValueError(f"Unsupported minmax op: {op_name}")
        return MinMaxOpTransform(left, right, op_name, op)
    if kind == "const":
        child = transform_from_config(cfg["child"])  # type: ignore[index]
        op_name = cfg.get("op_name", "add")
        const = cfg.get("constant")
        if op_name in ("add", "sub", "mul", "div"):
            op = _OP_BINARY[op_name]
            return ConstantOpTransform(child, const, op_name, lambda x, c: op(x, c))
        if op_name in ("rsub", "rdiv"):
            # right-constant operations
            return ConstantOpTransform(child, const, op_name, (lambda x, c: c - x) if op_name == "rsub" else (lambda x, c: c / x))
        if op_name in ("min", "max"):
            mm = _OP_MINMAX[op_name]
            return ConstantOpTransform(child, const, op_name, lambda x, c: mm(x, c))
        raise ValueError(f"Unsupported const op: {op_name}")
    if kind == "unary":
        child = transform_from_config(cfg["child"])  # type: ignore[index]
        op_name = cfg.get("op_name", "abs")
        op = _maybe_unary_from_name(op_name)
        if op is None:
            raise ValueError(f"Unsupported unary op: {op_name}")
        return UnaryOpTransform(child, op_name, op)
    if kind == "compose":
        # lazy import to avoid circular dependency at import-time
        from finmlkit.feature.kit import Compose  # type: ignore
        steps = [transform_from_config(s) for s in cfg.get("steps", [])]
        return Compose(*steps)  # type: ignore[arg-type]
    if kind == "external":
        # Lazy import to avoid circulars
        from finmlkit.feature.transforms import ExternalFunction  # type: ignore
        func_path = cfg.get("func")
        if not func_path:
            raise ValueError("ExternalFunction config requires 'func' path")
        input_cols = cfg.get("requires", [])
        if not input_cols:
            raise ValueError("ExternalFunction config missing 'requires'")
        input_cols = input_cols[0] if len(input_cols) == 1 else input_cols
        produces = cfg.get("produces", [])
        if isinstance(produces, list):
            output_cols = produces[0] if len(produces) == 1 else produces
        else:
            output_cols = produces
        args = _deserialize_value(cfg.get("args", []))
        kwargs = _deserialize_value(cfg.get("kwargs", {}))
        pass_numpy = bool(cfg.get("pass_numpy", False))
        return ExternalFunction(func_path, input_cols, output_cols, args=args, kwargs=kwargs, pass_numpy=pass_numpy)

    # Generic transform reconstruction via constructor signature
    cls = _import_class(cfg["class"])  # type: ignore[index]
    params = {k: _deserialize_value(v) for k, v in cfg.get("params", {}).items()}
    try:
        obj = cls(**params)
        return obj
    except Exception as e:  # fallback: shallow construct without __init__ (best-effort)
        logger.warning(f"Falling back to shallow reconstruction for {cfg['class']}: {e}")
        obj = cls.__new__(cls)
        # Set minimal fields
        setattr(obj, "requires", cfg.get("requires", []))
        setattr(obj, "produces", cfg.get("produces", []))
        # Restore common attributes
        for k, v in params.items():
            setattr(obj, k, v)
        return obj


# --- Graph utilities ----------------------------------------------------------

class ComputationGraph:
    """Directed acyclic graph (DAG) capturing feature dependencies."""
    def __init__(self):
        self.edges: Dict[str, Set[str]] = {}  # src -> set(dst)
        self.nodes: Set[str] = set()

    def add_edge(self, src: str, dst: str):
        self.nodes.update([src, dst])
        self.edges.setdefault(src, set()).add(dst)
        self.edges.setdefault(dst, set())  # ensure dst in map

    def add_node(self, node: str):
        self.nodes.add(node)
        self.edges.setdefault(node, set())

    def topological_sort(self) -> List[str]:
        # Kahn's algorithm
        indeg = {n: 0 for n in self.nodes}
        for src, dests in self.edges.items():
            for d in dests:
                indeg[d] = indeg.get(d, 0) + 1
        zero = [n for n, d in indeg.items() if d == 0]
        order = []
        while zero:
            n = zero.pop(0)
            order.append(n)
            for d in list(self.edges.get(n, [])):
                indeg[d] -= 1
                if indeg[d] == 0:
                    zero.append(d)
        return order

    def visualize(self) -> str:
        lines = ["ComputationGraph:"]
        for src in sorted(self.edges.keys()):
            dests = ", ".join(sorted(self.edges[src]))
            lines.append(f"  {src} -> [{dests}]")
        return "\n".join(lines)


def _flatten_requires(t: BaseTransform) -> List[str]:
    # For op transforms, union child requires
    if isinstance(t, (BinaryOpTransform, MinMaxOpTransform)):
        return list(set(_flatten_requires(t.left) + _flatten_requires(t.right)))
    if isinstance(t, (UnaryOpTransform, ConstantOpTransform)):
        return _flatten_requires(t.transform)
    if hasattr(t, "transforms") and isinstance(getattr(t, "transforms", None), (list, tuple)):
        return t.transforms[0].requires  # first input drives
    return getattr(t, "requires", [])


def build_feature_graph(features: List["Feature"]) -> ComputationGraph:
    g = ComputationGraph()
    # Map of feature output names for cross-referencing
    outputs = {str(f.name): f for f in features if isinstance(f.name, (str,))}

    def _child_out_names(t: BaseTransform) -> List[str]:
        if isinstance(t, (BinaryOpTransform, MinMaxOpTransform)):
            left_name = t.left.output_name if isinstance(t.left.output_name, str) else str(t.left.output_name)
            right_name = t.right.output_name if isinstance(t.right.output_name, str) else str(t.right.output_name)
            return [left_name, right_name]
        if isinstance(t, (UnaryOpTransform, ConstantOpTransform)):
            child_name = t.transform.output_name
            return [child_name if isinstance(child_name, str) else str(child_name)]
        if hasattr(t, "transforms") and isinstance(getattr(t, "transforms", None), (list, tuple)):
            # Link to first step output if it exists among features
            first_out = t.transforms[0].output_name
            return [first_out if isinstance(first_out, str) else str(first_out)]
        return []

    for f in features:
        out_name = str(f.name)
        g.add_node(out_name)
        # Edges from raw inputs
        reqs = _flatten_requires(f.transform)
        for r in reqs:
            in_node = f"input:{r}"
            g.add_node(in_node)
            g.add_edge(in_node, out_name)
        # Edges from child transforms if those outputs are also features
        for child_out in _child_out_names(f.transform):
            if child_out in outputs and child_out != out_name:
                g.add_edge(child_out, out_name)
        # Back-compat: dependencies on other features if require equals other feature's output
        for other_name in outputs.keys():
            if other_name == out_name:
                continue
            if other_name in reqs:
                g.add_edge(other_name, out_name)
    return g
