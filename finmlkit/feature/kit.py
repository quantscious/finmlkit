from .base import BaseTransform, BinaryOpTransform, ConstantOpTransform, UnaryOpTransform, SISOTransform, MISOTransform
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


class Compose(BaseTransform):
    def __init__(self, *transforms: SISOTransform|MISOTransform):
        requires = transforms[0].requires[0]  # First tfs determines the source column
        first_output = transforms[0].output_name
        produces = "_".join([first_output] + [t.produces[0] for t in transforms[1:]])
        super().__init__(requires, produces)
        self.transforms = transforms

    def _validate_input(self, x: pd.DataFrame) -> bool:
        """
        Validate that the input DataFrame contains the required columns for all transforms.
        :param x: DataFrame to validate
        :return: True if the input is valid
        """
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires} not found in DataFrame")
        return True

    @property
    def output_name(self) -> str:
        """
        Get the output name of the composed transform.
        The output name is a combination of the first transform's output and the subsequent transforms' produces.
        :return: Output name
        """
        return self.produces[0]

    def _run_pipeline(self, x: pd.DataFrame, *, backend) -> pd.Series:
        """
        Apply the composed transforms to the input DataFrame.
        :param x: DataFrame to transform
        :param backend: Backend is already specified in the transforms
        :return: Transformed Series
        """
        self._validate_input(x)
        series_out = None
        for i, tfs in enumerate(self.transforms):
            if i == 0:
                # First transform on the input DataFrame
                # Check if the first product is already in the DataFrame (Often the case for the first transform in the chain)
                if tfs.produces[0] in x.columns:
                    series_out = x[tfs.produces[0]]
                else:
                    series_out = tfs(x)
            else:
                # Subsequent transforms on the output of the previous transform
                print(tfs.requires[0])
                series_out = tfs(pd.DataFrame(series_out.values, index=series_out.index, columns=[tfs.requires[0]]), backend=backend)

        # Return the final output Series with the composed name
        series_out.name = self.output_name

        return series_out

    def __call__(self, x: pd.DataFrame, *, backend="nb") -> pd.Series:
        """
        Apply the composed transforms to the input DataFrame.
        :param x: DataFrame to transform
        :param backend: Backend to use for the transform. Can be "pd" or "nb". Default is "nb".
        :return: Transformed Series
        """
        assert backend == "pd" or backend == "nb", "Backend must be either 'pd' or 'nb'."

        return self._run_pipeline(x, backend=backend)


class FeatureKit:
    def __init__(self, features: list[Feature], retain: list[str] = None):
        self.features = features
        self.retain = retain or []

    def build(self, df, *, backend="nb"):
        out = df[self.retain].copy()
        df = df.copy()
        for feat in self.features:
            res = feat.transform(df, backend=backend)
            if isinstance(res, pd.Series):
                # Single output transform case
                out[feat.name] = res
                df[feat.transform.output_name] = res  # cache the result in the DataFrame (for compose transforms)
            elif isinstance(res, tuple):
                # Multi output transform case
                for item in res:
                    out[item.name] = item
                    df[item.name] = item # cache the result in the DataFrame (for compose transforms)
            else:
                raise TypeError(f"Transform {feat} returned unexpected type: {type(res)}")

        return out


