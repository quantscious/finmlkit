from .base import BaseTransform, SISOTransform, MISOTransform
import pandas as pd
from typing import Callable


class Feature:
    """
    Wraps a BaseTransform and enables mathematical operations between features.
    """

    def __init__(self, transform: BaseTransform):
        self.transform = transform
        self._name = transform.output_name

    def __call__(self, x: pd.DataFrame, *, backend="nb"):
        """Forward the call to the underlying transform"""
        return self.transform(x, backend=backend)

    @property
    def name(self):
        """Get the output name from the wrapped transform"""
        return self._name

    @name.setter
    def name(self, output_name):
        """Set a custom name from the feature"""
        assert type(output_name) == type(self._name), "Same type"
        if isinstance(output_name, (tuple, list)):
            assert len(output_name) == len(self._name), "same length"
        self._name = output_name

    # Mathematical operations
    def __add__(self, other):
        if isinstance(other, Feature):
            return Feature(BinaryOpTransform(self.transform, other.transform, "add", lambda x, y: x + y))
        elif isinstance(other, (int, float)):
            return Feature(ConstantOpTransform(self.transform, other, "add", lambda x, c: x + c))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Feature):
            return Feature(BinaryOpTransform(self.transform, other.transform, "sub", lambda x, y: x - y))
        elif isinstance(other, (int, float)):
            return Feature(ConstantOpTransform(self.transform, other, "sub", lambda x, c: x - c))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Feature):
            return Feature(BinaryOpTransform(self.transform, other.transform, "mul", lambda x, y: x * y))
        elif isinstance(other, (int, float)):
            return Feature(ConstantOpTransform(self.transform, other, "mul", lambda x, c: x * c))
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Feature):
            return Feature(BinaryOpTransform(self.transform, other.transform, "div", lambda x, y: x / y))
        elif isinstance(other, (int, float)):
            return Feature(ConstantOpTransform(self.transform, other, "div", lambda x, c: x / c))
        return NotImplemented

    def __abs__(self):
        return Feature(UnaryOpTransform(self.transform, "abs", lambda x: x.abs()))

    # Right-side operations (for const op feature cases)
    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Feature(ConstantOpTransform(self.transform, other, "add", lambda x, c: x + c))
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Feature(ConstantOpTransform(self.transform, other, "rsub", lambda x, c: c - x))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Feature(ConstantOpTransform(self.transform, other, "mul", lambda x, c: x * c))
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Feature(ConstantOpTransform(self.transform, other, "rdiv", lambda x, c: c / x))
        return NotImplemented


class FeatureKit:
    def __init__(self, features: list[Feature], retain: list[str] = None):
        self.features = features
        self.retain = retain or []

    def build(self, df, *, backend="nb"):
        out = df[self.retain].copy()
        for feat in self.features:
            res = feat.transform(df, backend=backend)
            if isinstance(res, pd.Series):
                # Single output transform case
                out[feat.name] = res
            elif isinstance(res, tuple):
                # Multi output transform case
                for item in res:
                    out[item.name] = item
            else:
                raise TypeError(f"Transform {feat} returned unexpected type: {type(res)}")

        return out


class BinaryOpTransform(BaseTransform):
    """Transform that applies binary operations between two transforms"""
    def __init__(self, left: BaseTransform, right: BaseTransform, op_name: str, op_func: Callable):
        # Combine all input requirements from both transforms
        combined_inputs = list(set(left.requires + right.requires))
        output_name = f"{op_name}({left.output_name},{right.output_name})"
        super().__init__(combined_inputs, output_name)
        self.left = left
        self.right = right
        self.op_func = op_func

    def _validate_input(self, x):
        # binary operations are valid for SISO and MISO transforms
        if not isinstance(self.left, (SISOTransform, MISOTransform)):
            raise TypeError(f"Left transform must be SISO or MISO for binary OP, got {type(self.left)}")
        if not isinstance(self.right, (SISOTransform, MISOTransform)):
            raise TypeError(f"Right transform must be SISO or MISO for binary OP, got {type(self.right)}")
        return self.left._validate_input(x) and self.right._validate_input(x)

    @property
    def output_name(self) -> str:
        if isinstance(self.produces, list) and len(self.produces) == 1:
            return self.produces[0]
        return self.produces

    def __call__(self, x, *, backend="nb"):
        left_result = self.left(x, backend=backend)
        right_result = self.right(x, backend=backend)
        result = self.op_func(left_result, right_result)
        result.name = self.output_name
        return result


class ConstantOpTransform(BaseTransform):
    """Transform that applies operations between a transform and a constant"""
    def __init__(self, transform: BaseTransform, constant: float, op_name: str, op_func: Callable):
        super().__init__(transform.requires, f"{op_name}({transform.output_name},{constant})")
        self.transform = transform
        self.constant = constant
        self.op_func = op_func

    def _validate_input(self, x):
        return self.transform._validate_input(x)

    @property
    def output_name(self) -> str:
        if isinstance(self.produces, list) and len(self.produces) == 1:
            return self.produces[0]
        return self.produces

    def __call__(self, x, *, backend="nb"):
        result = self.transform(x, backend=backend)
        result = self.op_func(result, self.constant)
        result.name = self.output_name
        return result


class UnaryOpTransform(BaseTransform):
    """Transform that applies unary operations to a transform"""
    def __init__(self, transform: BaseTransform, op_name: str, op_func: Callable):
        super().__init__(transform.requires, f"{op_name}({transform.output_name})")
        self.transform = transform
        self.op_func = op_func

    def _validate_input(self, x):
        return self.transform._validate_input(x)

    @property
    def output_name(self) -> str:
        if isinstance(self.produces, list) and len(self.produces) == 1:
            return self.produces[0]
        return self.produces

    def __call__(self, x, *, backend="nb"):
        result = self.transform(x, backend=backend)
        result = self.op_func(result)
        result.name = self.output_name
        return result