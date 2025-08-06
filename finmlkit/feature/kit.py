from .base import BaseTransform, MinMaxOpTransform, BinaryOpTransform, ConstantOpTransform, UnaryOpTransform, SISOTransform, MISOTransform
import pandas as pd
import numpy as np
from finmlkit.utils.log import get_logger
import time

logger = get_logger(__name__)


class Feature:
    r"""High-level wrapper for data transformations enabling intuitive mathematical operations and fluent feature engineering.

    This class provides a user-friendly interface for financial feature engineering by wrapping :class:`BaseTransform`
    instances and enabling mathematical operations, function composition, and chainable transformations using familiar
    Python operators and methods. It serves as the primary building block for constructing complex feature engineering
    pipelines through an intuitive, expression-based syntax that mirrors mathematical notation.

    **Core Design Philosophy:**

    The Feature class implements a **fluent interface** design pattern that enables natural mathematical expressions
    for feature engineering. Instead of manually composing transform objects, users can write feature engineering
    logic using familiar mathematical operators and method chaining:

    .. code-block:: python

       # Traditional transform composition (verbose)
       ma_transform = SimpleMovingAverageTransform('close', 'sma_20')
       ratio_transform = BinaryOpTransform(price_transform, ma_transform, 'div', lambda x, y: x / y)

       # Feature-based composition (intuitive)
       price = Feature(PriceTransform('close'))
       ma20 = Feature(SimpleMovingAverageTransform('close', 'sma_20'))
       price_to_ma_ratio = price / ma20

    **Mathematical Operations Framework:**

    The class overloads Python's mathematical operators to create new Feature instances with automatically
    composed transformations. Supported operations include:

    - **Binary Operations**: Addition (+), subtraction (-), multiplication (*), division (/)
    - **Unary Operations**: Absolute value (abs()), negation
    - **Comparison Operations**: Element-wise minimum and maximum (static methods)
    - **Constant Operations**: Mathematical operations with scalar values (e.g., `feature * 2`)
    - **Reverse Operations**: Enable natural syntax like `3 + feature`

    Each mathematical operation creates a new Feature instance wrapping an appropriate transform that
    performs the mathematical computation during evaluation.

    **Function Application and Composition:**

    The :meth:`apply` method enables applying arbitrary functions to feature outputs, supporting:

    - **Custom Functions**: User-defined lambda functions or named functions
    - **Built-in Methods**: Pandas methods like rolling operations, transformations
    - **Mathematical Functions**: NumPy mathematical functions (log, exp, sqrt, etc.)
    - **Automatic Naming**: Generates descriptive names based on function names and parameters

    **Performance Optimization Features:**

    - **Caching Support**: Optional caching mechanism to avoid recomputing expensive transformations
    - **Backend Selection**: Supports both pandas ("pd") and Numba ("nb") computational backends
    - **Lazy Evaluation**: Transformations are only computed when :meth:`__call__` is invoked
    - **Transform Reuse**: Wrapped transforms can be shared across multiple Feature instances

    **Built-in Convenience Methods:**

    The class provides pre-implemented methods for common financial operations:

    - **Statistical Functions**: Rolling mean, standard deviation, exponential moving average
    - **Mathematical Transforms**: Logarithms, exponentials, square roots, clipping
    - **Time Series Operations**: Lagging, shifting, rolling aggregations
    - **Risk/Return Metrics**: Log returns, volatility calculations, normalized features

    **Feature Naming and Metadata:**

    Features maintain intelligent naming schemes that:

    - **Preserve Traceability**: Names reflect the sequence of operations applied
    - **Support Customization**: Allow manual name overrides for semantic clarity
    - **Enable Pipeline Integration**: Generate column names suitable for DataFrame integration
    - **Maintain Consistency**: Ensure naming conventions across mathematical operations

    **Integration with Transform Hierarchy:**

    The Feature class seamlessly integrates with the transform ecosystem:

    - **Transform Wrapping**: Can wrap any :class:`BaseTransform` subclass (SISO, MISO, SIMO, MIMO)
    - **Operation Transforms**: Automatically creates appropriate operation transforms for mathematical expressions
    - **Backend Compatibility**: Supports both pandas and Numba backends through wrapped transforms
    - **Validation Inheritance**: Inherits input validation and error handling from underlying transforms

    .. note::
        Feature instances are designed to be **immutable** - mathematical operations create new Feature
        objects rather than modifying existing ones. This design promotes functional programming patterns
        and prevents unintended side effects in complex feature engineering pipelines.

    .. tip::
        The caching mechanism is particularly valuable for expensive transformations that are reused
        across multiple features. Consider enabling caching for computationally intensive operations
        like rolling correlations, technical indicators, or statistical decompositions.

    Args:
        transform (BaseTransform): The underlying transform to wrap. Can be any subclass of
            :class:`BaseTransform` including SISO, MISO, SIMO, or MIMO transforms.

    Raises:
        AttributeError: If the wrapped transform doesn't have the required `output_name` attribute.
        TypeError: If name setter receives incompatible types during custom name assignment.
        AssertionError: If custom names have mismatched lengths for multi-output transforms.

    Examples:
        Basic feature creation and mathematical operations:

           >>> # doctest: +SKIP
           >>> # Create base features
           >>> dates = pd.date_range('2023-01-01', periods=20, freq='D')
           >>> data = pd.DataFrame({
           ...     'close': 100 + np.random.randn(20).cumsum(),
           ...     'volume': np.random.randint(1000, 5000, 20)
           ... }, index=dates)
           >>>
           >>> # Wrap transforms as features
           >>> price = Feature(SimpleMovingAverageTransform('close', 'sma_20'))  # doctest: +SKIP
           >>> volume = Feature(VolumeTransform('volume'))  # doctest: +SKIP
           >>>
           >>> # Mathematical operations
           >>> price_vol_ratio = price / volume  # doctest: +SKIP
           >>> normalized_price = (price - price.rolling_mean(10)) / price.rolling_std(10)  # doctest: +SKIP

        Advanced function application:

           >>> # doctest: +SKIP
           >>> # Custom function application
           >>> log_returns = price.apply(lambda x: x.pct_change().apply(np.log), suffix='log_ret')  # doctest: +SKIP
           >>>
           >>> # Built-in convenience methods
           >>> clipped_returns = log_returns.clip(lower=-0.1, upper=0.1)  # doctest: +SKIP
           >>> volatility = log_returns.rolling_std(30)  # doctest: +SKIP
           >>>
           >>> # Composite feature engineering
           >>> momentum = price / price.lag(20) - 1  # doctest: +SKIP
           >>> momentum_signal = Feature.max(Feature.min(momentum, 0.2), -0.2)  # doctest: +SKIP

        Performance optimization with caching:

           >>> # doctest: +SKIP
           >>> # Create expensive computation
           >>> complex_indicator = Feature(ComplexTechnicalIndicator('close', window=100))  # doctest: +SKIP
           >>>
           >>> # Use caching for repeated calculations
           >>> cache = pd.DataFrame()  # doctest: +SKIP
           >>> result1 = complex_indicator(data, cache=cache)  # doctest: +SKIP
           >>> result2 = complex_indicator(data, cache=cache)  # Uses cached result  # doctest: +SKIP

        Feature pipeline construction:

        .. code-block:: python

           # Build a comprehensive feature set
           base_price = Feature(PriceTransform('close'))

           # Technical indicators
           sma_20 = base_price.rolling_mean(20)
           sma_50 = base_price.rolling_mean(50)
           rsi = Feature(RSITransform('close', 14))

           # Derived features
           price_momentum = base_price / base_price.lag(10) - 1
           sma_ratio = sma_20 / sma_50
           mean_reversion = (base_price - sma_20) / base_price.rolling_std(20)

           # Composite signals
           trend_signal = Feature.max(Feature.min(sma_ratio - 1, 0.1), -0.1)
           momentum_signal = price_momentum.clip(lower=-0.2, upper=0.2)
           combined_signal = (trend_signal + momentum_signal) / 2

    See Also:

        - :class:`BaseTransform`: The underlying transform interface that Feature wraps.
        - :class:`CoreTransform`: Base class for dual-backend transforms used within Features.
        - :class:`BinaryOpTransform`: Transform class created for binary mathematical operations.
        - :class:`UnaryOpTransform`: Transform class created for unary mathematical operations.
        - :class:`ConstantOpTransform`: Transform class created for operations with scalar constants.

    References:
        .. _`Fluent Interface Pattern`: https://martinfowler.com/bliki/FluentInterface.html
        .. _`Feature Engineering for Machine Learning`: https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/
        .. _`Operator Overloading in Python`: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    """

    def __init__(self, transform: BaseTransform):
        self.transform = transform
        self._name = transform.output_name

    def __call__(self, x: pd.DataFrame, *, cache: pd.DataFrame = None, backend="nb"):
        """Forward the call to the underlying transform"""
        if cache is not None:
            if isinstance(self.transform.output_name, str) and self.transform.output_name in cache.columns:
                logger.info(f"Using cached output for {self.transform.output_name}")
                return cache[self.transform.output_name]
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

    def apply(self, func, *args, suffix=None, **kwargs):
        """
        Apply an arbitrary function to the output of this feature.

        :param func: The function to apply to the feature output
        :param args: Additional positional arguments to pass to the function
        :param suffix: Optional suffix to add to the feature name (default is function name)
        :param kwargs: Additional keyword arguments to pass to the function
        :return: A new Feature with the function applied
        """
        func_name = suffix if suffix is not None else func.__name__

        # Use the current feature name as the base name instead of the input column
        base_name = str(self.name)  # Convert to string to be safe

        # Combine the base name with the function name
        new_name = f"{base_name}_{func_name}"

        # Create the transform with the operation
        transform = UnaryOpTransform(self.transform, func_name, lambda x: func(x, *args, **kwargs))

        # Create and initialize the new feature
        feature = Feature(transform)

        # Update the name in both places - crucial for consistency
        transform.produces = [new_name]  # Update the transform's produces list to contain the new name
        feature.name = new_name  # Set the name directly on the feature

        return feature

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

    # Common operations that can be added as convenience methods
    def clip(self, lower=None, upper=None):
        """
        Clip the values of the feature between lower and upper bounds.

        :param lower: Lower boundary (optional)
        :param upper: Upper boundary (optional)
        :return: A new Feature with clipped values
        """
        suffix = f"clip_{lower}_{upper}".replace("None", "")
        return self.apply(lambda x: x.clip(lower=lower, upper=upper), suffix=suffix)

    def abs(self):
        """
        Get the absolute values of the feature.

        :return: A new Feature with absolute values
        """
        return Feature(UnaryOpTransform(self.transform, "abs", lambda x: x.abs()))

    def log(self):
        """
        Get the natural logarithm of the feature.

        :return: A new Feature with log values
        """
        return self.apply(lambda x: x.apply(lambda v: np.log(v) if v > 0 else np.nan), suffix="log")

    def log1p(self):
        """
        Get the natural logarithm of the feature.

        :return: A new Feature with log values
        """
        return self.apply(lambda x: x.apply(lambda v: np.log1p(v) if v >= 0 else np.nan), suffix="log1p")

    def exp(self):
        """
        Get the exponential of the feature.

        :return: A new Feature with exp values
        """
        return self.apply(lambda x: x.apply(np.exp), suffix="exp")

    def square(self):
        """
        Get the square of the feature.

        :return: A new Feature with squared values
        """
        return self.apply(lambda x: x ** 2, suffix="square")

    def sqrt(self):
        """
        Get the square root of the feature.

        :return: A new Feature with square root values
        """
        return self.apply(lambda x: x.apply(lambda v: np.sqrt(v) if v >= 0 else np.nan), suffix="sqrt")

    def rolling_mean(self, window):
        """
        Calculate the rolling mean of the feature.

        :param window: Rolling window size
        :return: A new Feature with rolling mean values
        """
        return self.apply(lambda x: x.rolling(window=window).mean(), suffix=f"rmean{window}")

    def ema(self, span, adjust=True):
        """
        Calculate the Exponential Moving Average (EMA) of the feature.

        :param span: Span for the EMA calculation
        :param adjust: Whether to adjust the EMA calculation (default is True)
        :return: A new Feature with EMA values
        """
        return self.apply(lambda x: x.ewm(span=span, adjust=adjust).mean(), suffix=f"ema{span}")

    def rolling_sum(self, window):
        """
        Calculate the rolling sum of the feature.

        :param window: Rolling window size
        :return: A new Feature with rolling sum values
        """
        return self.apply(lambda x: x.rolling(window=window).sum(), suffix=f"rsum{window}")

    def rolling_std(self, window):
        """
        Calculate the rolling standard deviation of the feature.

        :param window: Rolling window size
        :return: A new Feature with rolling std values
        """
        return self.apply(lambda x: x.rolling(window=window).std(), suffix=f"rstd{window}")

    def lag(self, period):
        """
        Create a lagged version of the feature.

        :param period: Number of periods to lag
        :return: A new Feature with lagged values
        """
        return self.apply(lambda x: x.shift(period), suffix=f"lag{period}")

    @staticmethod
    def min(a, b):
        """
        Calculate the element-wise minimum between two features.

        :param a: First feature or scalar
        :param b: Second feature or scalar
        :return: A new Feature containing the element-wise minimum
        """
        if isinstance(a, Feature) and isinstance(b, Feature):
            return Feature(MinMaxOpTransform(a.transform, b.transform, "min", lambda x, y: np.minimum(x, y)))
        elif isinstance(a, Feature) and isinstance(b, (int, float)):
            return Feature(ConstantOpTransform(a.transform, b, "min", lambda x, c: np.minimum(x, c)))
        elif isinstance(b, Feature) and isinstance(a, (int, float)):
            return Feature(ConstantOpTransform(b.transform, a, "min", lambda x, c: np.minimum(x, c)))
        return NotImplemented

    @staticmethod
    def max(a, b):
        """
        Calculate the element-wise maximum between two features.

        :param a: First feature or scalar
        :param b: Second feature or scalar
        :return: A new Feature containing the element-wise maximum
        """
        if isinstance(a, Feature) and isinstance(b, Feature):
            return Feature(MinMaxOpTransform(a.transform, b.transform, "max", lambda x, y: np.maximum(x, y)))
        elif isinstance(a, Feature) and isinstance(b, (int, float)):
            return Feature(ConstantOpTransform(a.transform, b, "max", lambda x, c: np.maximum(x, c)))
        elif isinstance(b, Feature) and isinstance(a, (int, float)):
            return Feature(ConstantOpTransform(b.transform, a, "max", lambda x, c: np.maximum(x, c)))
        return NotImplemented


class Compose(BaseTransform):
    r"""Composite transform that chains multiple single-output transforms into a sequential processing pipeline.

    This class implements the **Composite Pattern** for data transformations, enabling the creation of complex
    feature engineering pipelines by sequentially chaining :class:`SISOTransform` and :class:`MISOTransform`
    instances. It provides a unified interface for executing multi-step transformations while maintaining
    the same interface as individual transforms, enabling seamless integration with the broader transform ecosystem.

    **Pipeline Composition Framework:**

    The Compose class creates a linear processing pipeline where each transform's output becomes the input
    to the next transform in the sequence. For a composition of transforms :math:`T_1, T_2, \ldots, T_n`,
    the overall transformation is:

    .. math::
        Y = T_n(T_{n-1}(\ldots T_2(T_1(X)) \ldots))

    where :math:`X` is the input DataFrame and :math:`Y` is the final output Series. This composition
    enables building sophisticated feature engineering workflows from simple, reusable transform components.

    **Key Design Features:**

    - **Sequential Processing**: Transforms are applied in the order specified during initialization
    - **Type Safety**: Only accepts SISO and MISO transforms that produce single outputs compatible with subsequent inputs
    - **Automatic Naming**: Generates descriptive output names by concatenating all transform identifiers
    - **Input Validation**: Validates the initial input and ensures compatibility throughout the pipeline
    - **Backend Consistency**: Maintains the same computational backend across all pipeline stages
    - **Caching Optimization**: Supports skipping initial transforms if their outputs already exist in the input DataFrame

    **Pipeline Execution Logic:**

    The composition handles several execution scenarios:

    1. **Fresh Computation**: All transforms are executed sequentially from the input DataFrame
    2. **Partial Caching**: If the first transform's output already exists in the input DataFrame, it uses the cached result
    3. **Intermediate Processing**: Each subsequent transform receives a temporary DataFrame containing only the required input column
    4. **Result Propagation**: Intermediate results are passed through the pipeline until the final output is produced

    **Naming Convention:**

    Output names are constructed by concatenating the first transform's output name with all subsequent transforms'
    `produces` identifiers:

    .. math::
        \text{output_name} = \text{first_output} + \text{"_"} + \text{produces}_2 + \text{"_"} + \ldots + \text{"_"} + \text{produces}_n

    For example, composing a moving average transform (producing `'ma20'`), RSI transform (producing `'rsi14'`),
    and signal transform (producing `'signal'`) results in the output name `'ma20_rsi14_signal'`.

    **Performance Considerations:**

    - **Memory Efficiency**: Only one intermediate Series is maintained at a time, minimizing memory footprint
    - **Backend Optimization**: All transforms use the same computational backend for consistency
    - **Caching Benefits**: Can leverage pre-computed results to avoid redundant calculations
    - **Error Propagation**: Validation errors are caught early before expensive computations begin

    **Use Cases in Financial Engineering:**

    - **Technical Indicator Chains**: Price → Moving Average → RSI → Trading Signal
    - **Risk Metric Pipelines**: Returns → Volatility → Value-at-Risk → Risk Adjusted Return
    - **Factor Construction**: Raw Data → Normalization → Winsorization → Z-Score → Factor Score
    - **Signal Processing**: Price → Log Returns → Smoothing → Momentum → Regime Classification

    .. important::
        All transforms in the composition must produce single outputs (SISO or MISO only). For transforms
        producing multiple outputs (SIMO, MIMO), use explicit pipeline construction with intermediate
        DataFrame management instead of the Compose class.

    Args:
        *transforms (SISOTransform|MISOTransform): Variable number of transforms to compose into a pipeline.
            Must be single-output transforms with compatible input/output column specifications.
            The first transform determines the pipeline's input requirements.

    Raises:
        TypeError: If input is not a pandas DataFrame during validation.
        ValueError: If the required input column is not found in the DataFrame.
        AssertionError: If backend parameter is not "pd" or "nb".
        AttributeError: If transforms don't have required attributes (requires, produces, output_name).

    Examples:
        Creating a technical analysis pipeline:

        .. code-block:: python

           from your_module import Compose, SimpleMovingAverageTransform, RSITransform, SignalTransform

           # Create individual transforms
           ma_transform = SimpleMovingAverageTransform('close', 'ma20')
           rsi_transform = RSITransform('ma20', 'rsi14')  # RSI of moving average
           signal_transform = SignalTransform('rsi14', 'signal')  # Trading signal from RSI

           # Compose into pipeline
           pipeline = Compose(ma_transform, rsi_transform, signal_transform)

           print(f"Pipeline input: {pipeline.requires}")  # ['close']
           print(f"Pipeline output: {pipeline.output_name}")  # 'ma20_rsi14_signal'

        Using the composed pipeline:

           >>> # doctest: +SKIP
           >>> # Sample price data
           >>> dates = pd.date_range('2023-01-01', periods=100, freq='D')
           >>> data = pd.DataFrame({
           ...     'close': 100 + np.random.randn(100).cumsum()
           ... }, index=dates)
           >>>
           >>> # Apply the complete pipeline
           >>> pipeline = Compose(ma_transform, rsi_transform, signal_transform)  # doctest: +SKIP
           >>> result = pipeline(data, backend='pd')  # doctest: +SKIP
           >>> print(f"Pipeline output type: {type(result)}")  # doctest: +SKIP
           Pipeline output type: <class 'pandas.core.series.Series'>
           >>> print(f"Output name: {result.name}")  # doctest: +SKIP
           Output name: ma20_rsi14_signal

        Advanced pipeline with caching optimization:

           >>> # doctest: +SKIP
           >>> # Data with pre-computed moving average
           >>> data_with_ma = data.copy()
           >>> data_with_ma['ma20'] = data['close'].rolling(20).mean()  # doctest: +SKIP
           >>>
           >>> # Pipeline will skip the first transform and use cached MA
           >>> result_cached = pipeline(data_with_ma, backend='nb')  # doctest: +SKIP
           >>> # First transform is skipped, starts with RSI calculation

        Multi-step risk analysis pipeline:

        .. code-block:: python

           # Risk analysis pipeline: Returns → Volatility → VaR → Risk Score
           returns_transform = ReturnsTransform('close', 'returns')
           volatility_transform = VolatilityTransform('returns', 'vol30')
           var_transform = VaRTransform('vol30', 'var95')
           risk_score_transform = RiskScoreTransform('var95', 'risk_score')

           risk_pipeline = Compose(
               returns_transform,
               volatility_transform,
               var_transform,
               risk_score_transform
           )

           # Single call computes entire risk analysis chain
           risk_metrics = risk_pipeline(price_data, backend='nb')

        Integration with Feature class:

        .. code-block:: python

           # Compose can be wrapped in Feature for mathematical operations
           from your_module import Feature

           technical_pipeline = Compose(ma_transform, rsi_transform)
           technical_feature = Feature(technical_pipeline)

           # Mathematical operations on the composed pipeline
           normalized_signal = (technical_feature - 50) / 50  # Normalize RSI
           combined_signal = technical_feature * volume_feature

    See Also:

        -  :class:`BaseTransform`: The base interface that Compose implements and extends.
        - :class:`SISOTransform`: Single-input, single-output transforms that can be composed.
        - :class:`MISOTransform`: Multiple-input, single-output transforms that can be composed.
        - :class:`Feature`: High-level wrapper that can encapsulate Compose instances for mathematical operations.
        - :class:`Pipeline`: Alternative approach for more complex multi-branch transformation workflows.

    References:
        .. _`Composite Pattern`: https://refactoring.guru/design-patterns/composite
        .. _`Pipeline Pattern in Data Processing`: https://martinfowler.com/articles/collection-pipeline/
        .. _`Functional Composition`: https://en.wikipedia.org/wiki/Function_composition_(computer_science)
    """

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
                # print(tfs.requires[0])
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
    r"""High-level orchestration framework for executing collections of Feature objects in financial machine learning pipelines.

    This class serves as the primary interface for building comprehensive feature engineering workflows by coordinating
    multiple :class:`Feature` instances, managing computational resources, and providing performance diagnostics.
    It represents the culmination of the feature engineering framework, enabling practitioners to construct, execute,
    and analyze complex feature sets with minimal boilerplate code and maximum computational efficiency.

    **Pipeline Orchestration Architecture:**

    FeatureKit implements a **batch processing pattern** for feature computation, where multiple Feature objects are
    executed sequentially against a shared DataFrame. This approach enables sophisticated optimization strategies:

    - **Incremental Caching**: Intermediate feature outputs are cached within the working DataFrame, enabling dependent
      features to reuse previously computed results without redundant calculations
    - **Selective Retention**: Original DataFrame columns can be preserved alongside computed features, maintaining
      data lineage and enabling hybrid analytical workflows
    - **Performance Profiling**: Optional timing analysis identifies computational bottlenecks and guides optimization efforts
    - **Backend Consistency**: All features execute with the same computational backend for consistent performance characteristics

    **Mathematical Processing Framework:**

    For a collection of features :math:`F_1, F_2, \ldots, F_n` applied to input DataFrame :math:`D`, the processing
    follows this computational model:

    .. math::
        \begin{align}
        D_0 &= D \cup \text{retain_columns} \\
        D_1 &= D_0 \cup \{F_1(D_0)\} \\
        D_2 &= D_1 \cup \{F_2(D_1)\} \\
        &\vdots \\
        D_n &= D_{n-1} \cup \{F_n(D_{n-1})\}
        \end{align}

    where :math:`\cup` represents column-wise DataFrame concatenation and each :math:`F_i(D_{i-1})` can access
    all previously computed features through the caching mechanism.

    This iterative approach enables:

    - **Memory Efficiency**: Only one working DataFrame is maintained, with features added incrementally
    - **Computational Reuse**: Expensive intermediate calculations are preserved for downstream feature computations

    **Performance Analysis System:**

    When timing analysis is enabled, FeatureKit generates detailed performance metrics:

    .. math::
        \text{Relative Performance} = \frac{t_i}{\max(t_1, t_2, \ldots, t_n)} \times 100\%

    where :math:`t_i` is the execution time for feature :math:`i`. Results are visualized using ASCII bar charts
    that provide immediate visual feedback on computational bottlenecks.

    **Integration with Feature Ecosystem:**

    FeatureKit seamlessly integrates with the complete feature engineering framework:

    - **Feature Objects**: Accepts any Feature instance, regardless of the underlying transform type (SISO, MISO, SIMO, MIMO)
    - **Mathematical Expressions**: Can execute features created through mathematical operations (addition, multiplication, etc.)
    - **Composed Transforms**: Supports features built using the :class:`Compose` class for complex transformation chains
    - **Custom Functions**: Works with features created using the :meth:`Feature.apply` method for arbitrary function application

    **Production Deployment Considerations:**

    The class is designed for both research and production environments:

    - **Scalability**: Efficient memory management enables processing of large datasets without excessive resource consumption
    - **Reproducibility**: Deterministic execution order ensures consistent results across runs
    - **Debugging**: Timing analysis and clear error messages facilitate troubleshooting in complex pipelines
    - **Flexibility**: Support for both pandas and Numba backends enables optimization for different deployment scenarios

    **Caching Strategy and Optimization:**

    The caching mechanism provides significant performance benefits:

    1. **Intermediate Result Reuse**: Features that depend on common sub-computations automatically benefit from cached results
    2. **Memory Efficiency**: Results are stored directly in the working DataFrame, minimizing memory overhead
    3. **Cache Coherence**: The cache is updated incrementally, ensuring all features see consistent intermediate state

    Args:
        features (list[Feature]): Ordered list of Feature instances to execute. Order determines execution sequence
            and affects caching behavior for interdependent features.
        retain (list[str], optional): Column names from the input DataFrame to preserve in the output unchanged.
            If None or empty, only computed features are included in the output DataFrame.

    Raises:
        TypeError: If any feature returns an unexpected type (not Series or tuple of Series).
        KeyError: If retained columns are not present in the input DataFrame.
        AttributeError: If Feature objects lack required attributes or methods.

    Examples:
        Basic feature pipeline construction:

           >>> # doctest: +SKIP
           >>> # Prepare sample financial data
           >>> dates = pd.date_range('2023-01-01', periods=100, freq='D')
           >>> np.random.seed(42)
           >>> prices = 100 + np.random.randn(100).cumsum()
           >>> data = pd.DataFrame({'close': prices, 'volume': np.random.randint(1000, 10000, 100)}, index=dates)
           >>>
           >>> # Create individual features
           >>> price_feature = Feature(SimpleMovingAverageTransform('close', 'sma_20'))
           >>> rsi_feature = Feature(RSITransform('close', 'rsi_14'))
           >>>
           >>> # Build feature pipeline
           >>> feature_kit = FeatureKit([price_feature, rsi_feature], retain=['close', 'volume'])
           >>> result_df = feature_kit.build(data, backend='nb', timeit=False)
           >>> print(f"Output shape: {result_df.shape}")  # doctest: +SKIP
           Output shape: (100, 4)
           >>> print(f"Columns: {list(result_df.columns)}")  # doctest: +SKIP
           Columns: ['close', 'volume', 'close_sma_20', 'close_rsi_14']

        Advanced pipeline with interdependent features:

           >>> # doctest: +SKIP
           >>> # Create features with dependencies
           >>> base_price = Feature(PriceTransform('close'))
           >>> sma_20 = base_price.rolling_mean(20)
           >>> price_to_sma_ratio = base_price / sma_20  # Depends on sma_20
           >>> momentum_signal = price_to_sma_ratio.clip(lower=0.8, upper=1.2)  # Depends on ratio
           >>>
           >>> advanced_kit = FeatureKit([
           ...     base_price,
           ...     sma_20,
           ...     price_to_sma_ratio,
           ...     momentum_signal
           ... ], retain=['close'])
           >>>
           >>> # Execute with performance profiling
           >>> advanced_result = advanced_kit.build(data, backend='nb', timeit=True)  # doctest: +SKIP

        Production-scale feature engineering:

        .. code-block:: python

           # Large-scale feature construction
           import pandas as pd
           from finmlkit.features import Feature, FeatureKit
           from finmlkit.transforms import *

           # Load large dataset
           large_data = pd.read_csv('large_financial_dataset.csv', index_col='timestamp', parse_dates=True)

           # Comprehensive feature set
           features = []

           # Price-based features
           price = Feature(PriceTransform('close'))
           features.extend([
               price.rolling_mean(10),
               price.rolling_mean(20),
               price.rolling_mean(50),
               price.rolling_std(20),
               price.log().diff(),  # Log returns
           ])

           # Volume-based features
           volume = Feature(VolumeTransform('volume'))
           features.extend([
               volume.rolling_mean(20),
               (price * volume).rolling_mean(20),  # Dollar volume
           ])

           # Technical indicators
           features.extend([
               Feature(RSITransform('close', 'rsi_14')),
               Feature(MACDTransform('close', 'macd')),
               Feature(BollingerBandsTransform('close', 'bb')),
           ])

           # Cross-asset features (if multiple assets)
           if 'close_spy' in large_data.columns:
               spy_price = Feature(PriceTransform('close_spy'))
               beta = Feature(BetaTransform(['close', 'close_spy'], 'beta_spy'))
               features.append(beta)

           # Create comprehensive feature kit
           production_kit = FeatureKit(features, retain=['close', 'volume', 'open', 'high', 'low'])

           # Execute with timing for optimization analysis
           feature_matrix = production_kit.build(large_data, backend='nb', timeit=True)

           # Save results for model training
           feature_matrix.to_parquet('feature_matrix.parquet')

    See Also:

        - :class:`Feature`: Core wrapper class for individual transformations with mathematical operations.
        - :class:`BaseTransform`: Abstract base class for all transformation implementations.
        - :class:`Compose`: Pipeline composition class for chaining single-output transforms.
        - :class:`SISOTransform`, :class:`MISOTransform`, :class:`SIMOTransform`, :class:`MIMOTransform`: Concrete transform base classes.

    References:
        .. _`Feature Engineering for Machine Learning`: https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/
        .. _`Pipeline Pattern in Data Processing`: https://martinfowler.com/articles/collection-pipeline/
        .. _`Efficient Feature Computation`: https://papers.nips.cc/paper/2019/hash/496e05e1aea0a9c4655800e8a7b9ea28-Abstract.html
    """
    def __init__(self, features: list[Feature], retain: list[str] = None):
        self.features = features
        self.retain = retain or []

    def build(self, df, *, backend="nb", timeit=False):
        out = df[self.retain].copy()
        df = df.copy()

        timing_info = {}

        for feat in self.features:
            if timeit:
                start_time = time.time()

            res = feat(df, cache=df, backend=backend)

            if timeit:
                elapsed = time.time() - start_time
                key = feat.name if isinstance(feat.name, str) else str(feat.name)
                timing_info[key] = elapsed

            if isinstance(res, pd.Series):
                # Single output transform case
                out[feat.name] = res
                df[feat.transform.output_name] = res  # cache the result in the DataFrame (for compose transforms)
            elif isinstance(res, tuple):
                # Multi output transform case
                for item in res:
                    out[item.name] = item
                    df[item.name] = item  # cache the result in the DataFrame (for compose transforms)
            else:
                raise TypeError(f"Transform {feat} returned unexpected type: {type(res)}")

        if timeit:
            # Create a simple console plot for timing information
            print("\nFeature Timing Analysis:")
            print("=======================")

            # Sort features by execution time
            sorted_times = sorted(timing_info.items(), key=lambda x: x[1], reverse=True)

            # Find the max time for scaling
            max_time = max(t for _, t in sorted_times) if sorted_times else 0
            max_bar_length = 50  # Maximum number of characters for the bar

            # Print bars for each feature
            for feature_name, time_taken in sorted_times:
                bar_length = int((time_taken / max_time) * max_bar_length) if max_time > 0 else 0
                bar = '█' * bar_length
                print(f"{feature_name:<30} | {bar} {time_taken:.4f}s")
        return out
