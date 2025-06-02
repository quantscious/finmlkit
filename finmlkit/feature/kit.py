from .base import BaseTransform, BinaryOpTransform, ConstantOpTransform, UnaryOpTransform
import pandas as pd


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


