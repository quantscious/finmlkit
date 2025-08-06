from abc import ABC, abstractmethod
from typing import Union, Optional, Sequence, Callable
import pandas as pd
from finmlkit.utils.log import get_logger
import numpy as np
from numpy.typing import NDArray

logger = get_logger(__name__)


class BaseTransform(ABC):
    r"""Abstract base class for data transformations in financial machine learning pipelines.

    This class provides a standardized interface for implementing feature transformations, technical indicators,
    and data processing operations on financial time series data. It serves as the foundation for a modular
    transformation system that enables composable data preprocessing workflows with consistent input/output
    handling and validation.

    The transform framework is designed around the concept of **declarative data dependencies**, where each
    transform explicitly specifies its required input columns and produced output columns. This approach
    enables automatic dependency resolution, pipeline validation, and efficient computation planning for
    complex feature engineering workflows.

    **Core Design Principles:**

    - **Explicit Dependencies**: Each transform declares required input columns (``requires``) and output
      columns (``produces``), enabling automated pipeline construction and validation.

    - **Backend Flexibility**: Supports multiple computational backends (pandas ``"pd"`` for development/debugging,
      NumPy ``"nb"`` for production performance) with consistent interfaces.

    - **Immutable Operations**: Transforms are designed as pure functions that don't modify input data,
      promoting reproducibility and thread safety in parallel processing environments.

    - **Composability**: Transforms can be chained together to create complex feature engineering pipelines,
      with automatic handling of intermediate column dependencies.

    **Transform Lifecycle:**

    The execution of a transform follows this standardized pattern:

    1. **Input Validation**: :meth:`_validate_input` ensures required columns are present and data types are appropriate
    2. **Computation**: :meth:`__call__` applies the core transformation logic using the specified backend
    3. **Output Formatting**: Results are returned as properly named Series or tuples for integration into DataFrames

    This lifecycle enables robust error handling and consistent behavior across different transform implementations.

    **Backend Architecture:**

    The dual-backend system provides flexibility for different use cases:

    - **Pandas Backend (``"pd"``)**: Uses pandas operations for readable, debuggable code with excellent
      error messages and automatic handling of missing data, timestamps, and mixed data types.

    - **Numba Backend (``"nb"``)**: Leverages Numba for high-performance vectorized
      operations on numeric data, suitable for production environments with large datasets.

    Subclasses typically implement both backends to provide optimal performance characteristics for their
    specific use case while maintaining consistent results across backends.

    .. note::
        Subclasses must implement all abstract methods (``__call__``, ``_validate_input``, ``output_name``)
        to provide complete functionality. The base class handles input/output column management and
        provides the structural framework for consistent transform behavior.

    .. note::
        For transforms producing multiple outputs, ensure that the length of ``produces`` matches the
        number of Series returned by ``__call__``. This enables proper column naming in downstream
        DataFrame construction and pipeline operations.

    Args:
        input_cols (Union[Sequence[str], str]): Column name(s) required as input for the transformation.
            Can be a single string for single-column transforms or a sequence for multi-column operations.
        output_cols (Union[Sequence[str], str]): Column name(s) produced by the transformation.
            Must match the number of outputs returned by the ``__call__`` method.

    Raises:
        AssertionError: If input_cols or output_cols are not strings or sequences of strings.
        NotImplementedError: If abstract methods are not implemented in subclasses.

    See Also:
        :class:`CoreTransform`: Extends this base class to implement specific transformations

    """
    requires: list[str]      # input column names
    produces: list[str]      # output column name
    _output_name: Union[str, list[str]]

    def __init__(self, input_cols: Union[Sequence[str], str], output_cols: Union[Sequence[str], str]):
        assert isinstance(input_cols, (str, tuple, list)), f"Input columns must be a string or a sequence of strings. Got {type(input_cols)}"
        assert isinstance(output_cols, (str, tuple, list)), f"Output columns must be a string or a sequence of strings. Got {type(output_cols)}"
        self.requires = [input_cols] if isinstance(input_cols, str) else list(input_cols)
        self.produces = [output_cols] if isinstance(output_cols, str) else list(output_cols)

    # --- public API ---------------------------------------------------------
    @abstractmethod
    def __call__(self, x: pd.DataFrame, *, backend="nb") -> Union[pd.Series, tuple[pd.Series, ...]]:
        r"""Apply the transformation to input data using the specified computational backend.

        This method implements the core transformation logic and must be overridden by subclasses.
        The implementation should support both pandas and NumPy backends for flexibility and performance.

        :param x: Input DataFrame containing the required columns specified in ``self.requires``.
        :param backend: Computational backend to use. Options:
            - ``"pd"``: Use pandas operations (slower but more robust with mixed data types)
            - ``"nb"``: Use Numba for high-performance operations
        :returns: Transformed data as a Series (single output) or tuple of Series (multiple outputs).
            The number of returned Series must match the length of ``self.produces``.
        :raises ValueError: If required columns are missing or data validation fails.
        :raises NotImplementedError: Must be implemented by subclasses.

        .. note::
            Implementations should call ``self._validate_input(x)`` before processing to ensure
            data integrity and provide clear error messages for invalid inputs.
        """
        pass

    @abstractmethod
    def _validate_input(self, x: pd.DataFrame) -> bool:
        """
        Check if the input columns are present in the input DataFrame.
        This method is called before applying the transform.

        :param x: DataFrame to validate
        :return: True if the input is valid
        """
        pass

    @property
    @abstractmethod
    def output_name(self) -> Union[str, list[str]]:
        """
        Get the output names of the transform.
        This is used to determine the output column names in the DataFrame.
        Used by prepare_output_nb to create the output Series.
        :return: Output name or list of output names
        """
        pass


class CoreTransform(BaseTransform, ABC):
    r"""Concrete implementation framework for data transformations with dual-backend support and temporal data handling.

    This class extends :class:`BaseTransform` by providing a complete implementation skeleton for data transformations
    that require both pandas and Numba computational backends. It serves as the primary base class for
    financial indicators, technical analysis functions, and time-series feature engineering operations that need
    to handle temporal data with high performance requirements.

    **CoreTransform Architecture:**

    The class implements the abstract :meth:`BaseTransform.__call__` method and introduces a structured approach
    to backend-specific computation through four new abstract methods that subclasses must implement:

    - :meth:`_pd`: Pandas-based implementation for development and mixed data types
    - :meth:`_nb`: Numba-based implementation for production performance
    - :meth:`_prepare_input_nb`: Data preparation for NumPy backend operations
    - :meth:`_prepare_output_nb`: Result formatting for consistent DataFrame integration

    This separation enables clean implementation of complex transforms while maintaining performance optimization
    opportunities through specialized NumPy operations and potential Numba compilation.

    **Temporal Data Support:**

    CoreTransform provides specialized utilities for time-series data processing, which is essential for financial
    machine learning applications:

    - **DateTime Index Validation**: Ensures input DataFrames have proper temporal indexing for time-based features
    - **Timestamp Extraction**: Converts pandas datetime indexes to nanosecond timestamps for efficient numerical operations
    - **Temporal Consistency**: Maintains index alignment between input and output data for proper time-series handling

    These features enable transforms to work seamlessly with financial time series while preserving temporal
    relationships and enabling vectorized operations on timestamp data.

    **Backend Implementation Pattern:**

    The dual-backend pattern follows this structure:

    .. code-block:: python

       def _pd(self, x: pd.DataFrame) -> pd.Series:
           # Pandas implementation - readable, handles edge cases
           return x['price'].rolling(window=self.window).mean()

       def _nb(self, x: pd.DataFrame) -> pd.Series:
           # NumPy implementation - optimized for performance
           inputs = self._prepare_input_nb(x)
           result = fast_moving_average_nb(inputs['price'], self.window)
           return self._prepare_output_nb(x.index, result)

    This pattern enables subclasses to provide both readable pandas code for development and optimized
    NumPy/Numba code for production, with automatic backend selection based on performance requirements.

    **Error Handling and Validation:**

    The class enhances the validation framework from :class:`BaseTransform` with temporal-specific checks:
    - Validates datetime indexes for time-based operations
    - Ensures sufficient data history for windowed computations
    - Provides clear error messages for temporal data inconsistencies

    .. note::
        Subclasses implementing time-based features should call :meth:`_check_datetime_index` in their
        :meth:`_validate_input` implementation to ensure proper temporal data handling.

    .. note::
        The NumPy backend (:meth:`_nb`) should leverage vectorized operations and consider Numba compilation
        for transforms that will be applied to large datasets or in real-time processing scenarios.

    Args:
        input_cols (Union[Sequence[str], str]): Column name(s) required as input for the transformation.
            Inherited from :class:`BaseTransform`.
        output_cols (Union[Sequence[str], str]): Column name(s) produced by the transformation.
            Inherited from :class:`BaseTransform`.

    Raises:
        ValueError: If backend is not "pd" or "nb", or if datetime index validation fails.
        TypeError: If input is not a pandas DataFrame for temporal operations.
        NotImplementedError: If required abstract methods are not implemented by subclasses.

    See Also:
        :class:`SISOTransform`
    """
    def __init__(self, input_cols: Union[Sequence[str], str], output_cols: Union[Sequence[str], str]):
        super().__init__(input_cols, output_cols)

    # --- public API ---------------------------------------------------------
    def __call__(self, x: pd.DataFrame, *, backend="nb") -> Union[pd.Series, tuple[pd.Series, ...]]:
        r"""Apply the transformation using the specified computational backend.

        This method implements the abstract :meth:`BaseTransform.__call__` by providing backend
        routing and input validation. It delegates the actual computation to backend-specific
        methods (:meth:`_pd` or :meth:`_nb`) implemented by subclasses.

        :param x: Input DataFrame containing required columns with datetime index for temporal features.
        :param backend: Computational backend selection:

            - ``"pd"``: Use pandas operations via :meth:`_pd` method
            - ``"nb"``: Use Numba operations via :meth:`_nb` method
        :returns: Transformed data as Series or tuple of Series, depending on transform output count.
        :raises ValueError: If backend is not "pd" or "nb".
        :raises ValueError: If input validation fails (datetime index, missing columns, etc.).

        .. note::
            This method calls :meth:`_validate_input` before computation to ensure data integrity
            and provides consistent error handling across all transform implementations.
        """
        self._validate_input(x)

        if backend == "pd":
            return self._pd(x)
        elif backend == "nb":
            return self._nb(x)
        raise ValueError(f"Unknown backend {backend!r}")


    @staticmethod
    def _check_datetime_index(x: pd.DataFrame) -> bool:
        r"""Validate that input DataFrame has a datetime index suitable for time-based operations.

        This static method provides a reusable validation check for transforms that require temporal
        data. It ensures the DataFrame index can support time-based feature calculations and
        windowed operations that depend on temporal ordering.

        :param x: DataFrame to validate for datetime index.
        :returns: True if validation passes.
        :raises ValueError: If DataFrame does not have a datetime index.
        :raises TypeError: If input is not a pandas DataFrame.

        .. note::
            This method should be called in the :meth:`_validate_input` implementation of
            subclasses that perform time-based computations.

        """
        if isinstance(x, pd.DataFrame):
            if not pd.api.types.is_datetime64_any_dtype(x.index):
                raise ValueError("Input DataFrame must have a datetime index for time-based features.")
        else:
            raise TypeError("Input must be a pandas DataFrame")

        return True

    def _get_timestamps(self, x: pd.DataFrame) -> NDArray[np.int64]:
        r"""Extract nanosecond timestamps from DataFrame index for numerical operations.

        Converts pandas datetime index to NumPy array of int64 nanosecond timestamps,
        enabling efficient vectorized operations on temporal data while preserving
        precision for high-frequency financial data.

        :param x: DataFrame with datetime index to extract timestamps from.
        :returns: NumPy array of timestamps as int64 nanoseconds since epoch.
        :raises ValueError: If DataFrame does not have a datetime index.

        .. note::
            Nanosecond precision is maintained to support high-frequency trading data
            where microsecond or nanosecond timing precision may be relevant for analysis.

        """
        self._check_datetime_index(x)
        return x.index.values.astype(np.int64)

    # --- to be implemented by children --------------------------------------
    @abstractmethod
    def _prepare_input_nb(self, x: pd.DataFrame) -> Union[dict[str, NDArray], NDArray]:
        """
        Prepare array inputs for numba functions.

        :param x: DataFrame or Series to transform
        :return: Dict of input data for DataFrame or array for Series
        """
        pass

    @abstractmethod
    def _prepare_output_nb(self, idx: pd.Index, y: Union[NDArray, tuple[NDArray]]) -> Union[pd.Series, tuple[pd.Series, ...]]:
        """
        Prepare the output data for numba functions.
        :param idx: index of the original DataFrame
        :param y: Output data from the transform
        :return: Series or tuple of Series with the same index as the input data
        """
        pass

    @abstractmethod
    def _pd(self, x: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, tuple[pd.Series]]:
        """
        Transform the input data using pandas. For fast prototyping
        :param x: DataFrame or Series to transform
        """
        pass

    @abstractmethod
    def _nb(self, x: Union[pd.DataFrame, pd.Series])  -> Union[pd.Series, tuple[pd.Series]]:
        # Fall back to pandas if not overridden
        pass


class SISOTransform(CoreTransform, ABC):
    r"""Specialized transform for single-input, single-output operations on financial time series data.

    This class extends :class:`CoreTransform` to provide a streamlined interface for transforms that operate
    on exactly one input column and produce exactly one output column. It implements the most common pattern
    in financial feature engineering, where individual price series, volumes, or derived metrics are
    transformed into new features through mathematical operations, statistical calculations, or technical indicators.

    **SISO Transform Pattern:**

    The Single-Input, Single-Output pattern is fundamental in quantitative finance for creating derived features:

    - **Price Transformations**: Converting raw prices to returns, log-returns, or normalized values
    - **Statistical Features**: Computing rolling statistics like moving averages, volatility, or z-scores
    - **Technical Indicators**: Calculating RSI, MACD, Bollinger Bands, or momentum indicators
    - **Mathematical Operations**: Applying log transforms, power functions, or custom mathematical mappings

    This specialization provides several advantages over the general :class:`CoreTransform`:

    - **Simplified Interface**: Single string parameters instead of sequences for input/output specification
    - **Automatic Naming**: Output columns follow a standardized ``{input_col}_{output_col}`` naming convention
    - **Type Safety**: Guarantees single Series input/output for cleaner implementation
    - **Performance Optimization**: Streamlined data preparation methods optimized for single-column operations

    **Naming Convention:**

    The class implements a consistent naming scheme where output columns combine the input column name
    with the transform-specific suffix:

    .. math::
        \text{output\_name} = \text{input\_col} + \text{"\_"} + \text{output\_col}

    For example, transforming the ``'close'`` price with a ``'sma_20'`` transform produces ``'close_sma_20'``.
    This convention enables clear traceability of feature derivation and prevents naming conflicts in
    complex feature engineering pipelines.

    **Implementation Framework:**

    Subclasses need only implement the abstract methods from :class:`CoreTransform`:

    - :meth:`_pd`: Pandas-based computation for development and debugging
    - :meth:`_nb`: NumPy/Numba-based computation for production performance

    The class provides concrete implementations for input/output preparation and validation,
    significantly reducing boilerplate code for single-column transforms.

    .. note::
        The standardized naming convention assumes that transform names (``output_col``) are descriptive
        and unique within a feature set. Consider using prefixes or suffixes that clearly identify
        the transform type and parameters (e.g., ``'sma_20'``, ``'rsi_14'``, ``'vol_30d'``).

    .. note::
        For transforms requiring multiple input columns (e.g., price and volume for VWAP), use the
        more general :class:`CoreTransform` base class instead. SISO transforms are optimized
        specifically for single-column operations.

    Args:
        input_col (str): Name of the input column to transform (e.g., 'close', 'volume', 'high').
        output_col (str): Suffix for the output column name. Combined with input_col to create
            the full output column name following the pattern ``{input_col}_{output_col}``.

    Raises:
        TypeError: If input is not a pandas DataFrame during validation.
        ValueError: If the specified input column is not present in the DataFrame.
        NotImplementedError: If abstract methods from :class:`CoreTransform` are not implemented.

    Examples:
        Implementing a simple moving average transform:

        .. code-block:: python

           class SimpleMovingAverageTransform(SISOTransform):
               def __init__(self, window: int, input_col: str = 'close'):
                   super().__init__(input_col, f'sma_{window}')
                   self.window = window

               def _pd(self, x: pd.DataFrame) -> pd.Series:
                   outp = x[self.requires[0]].rolling(window=self.window).mean()
                   return outp.rename(self.output_name)

               def _nb(self, x: pd.DataFrame) -> pd.Series:
                   import numpy as np
                   from scipy import ndimage

                   data = self._prepare_input_nb(x)
                   # Use uniform filter for moving average
                   result = ndimage.uniform_filter1d(data.astype(float),
                                                    size=self.window,
                                                    mode='constant',
                                                    cval=np.nan)
                   return self._prepare_output_nb(x.index, result)

        Using SISO transforms in a feature pipeline:

           >>> # doctest: +SKIP
           >>> import pandas as pd
           >>> import numpy as np
           >>> # Sample price data
           >>> dates = pd.date_range('2023-01-01', periods=10, freq='D')
           >>> data = pd.DataFrame({
           ...     'close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
           ... }, index=dates)
           >>>
           >>> # Create transform
           >>> sma_transform = SimpleMovingAverageTransform(window=3)  # doctest: +SKIP
           >>> print(f"Input column: {sma_transform.requires[0]}")  # doctest: +SKIP
           Input column: close
           >>> print(f"Output name: {sma_transform.output_name}")  # doctest: +SKIP
           Output name: close_sma_3
           >>>
           >>> # Apply transform
           >>> sma_values = sma_transform(data, backend='pd')  # doctest: +SKIP
           >>> print(f"First valid SMA: {sma_values.dropna().iloc[0]:.2f}")  # doctest: +SKIP
           First valid SMA: 101.00

        Chaining multiple SISO transforms:

        .. code-block:: python

           # Create multiple transforms
           sma_5 = SimpleMovingAverageTransform(5, 'close')     # close_sma_5
           sma_20 = SimpleMovingAverageTransform(20, 'close')   # close_sma_20

           # Apply to same data
           data_with_sma = data.copy()
           data_with_sma['close_sma_5'] = sma_5(data, backend='pd')
           data_with_sma['close_sma_20'] = sma_20(data, backend='pd')

    See Also:
        :class:`CoreTransform`: General transform base class for multi-input/output operations.
        :class:`MISOTransform`: Transform base class for multiple-input, single-output operations.
    """
    def __init__(self, input_col: str, output_col: str):
        super().__init__(input_col, output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires[0]} not found in DataFrame")
        return True

    def _prepare_input_nb(self, x: pd.DataFrame) -> NDArray:
        """
        Prepare the input data for numba functions.
        :param x: DataFrame to transform
        :return: Numpy array of the input column
        """
        return x[self.requires[0]].values

    @property
    def output_name(self) -> str:
        """
        Get the output name of the transform.
        This is used to determine the output column name in the DataFrame.
        :return: Output name
        """
        return f"{self.requires[0]}_{self.produces[0]}"

    def _prepare_output_nb(self, idx: pd.Index, y: NDArray) -> pd.Series:
        """
        Prepare the output data for numba functions.
        :param idx: index of the original DataFrame
        :param y: Output data from the transform
        :return: Series with the same index as the input data
        """
        return pd.Series(y, index=idx, name=self.output_name)


class MISOTransform(CoreTransform, ABC):
    r"""Specialized transform for multiple-input, single-output operations in financial feature engineering.

    This class extends :class:`CoreTransform` to provide an optimized interface for transforms that require
    multiple input columns but produce exactly one output column. This pattern is fundamental in quantitative
    finance for creating composite features, ratios, statistical relationships, and cross-sectional indicators
    that combine information from multiple data sources or time series.

    **MISO Transform Applications:**

    The Multiple-Input, Single-Output pattern captures essential relationships in financial

    - **Price Ratios and Spreads**: Computing price ratios between assets, bid-ask spreads, or relative strength metrics
    - **Cross-Asset Correlations**: Rolling correlations, beta calculations, or cointegration measures between multiple series
    - **Volume-Price Relationships**: VWAP calculations, volume-weighted returns, or price-volume divergence indicators
    - **Multi-Timeframe Indicators**: Combining fast and slow moving averages, momentum crossovers, or trend convergence measures
    - **Statistical Composites**: Principal component features, factor loadings, or custom composite scores

    **Mathematical Framework:**

    For MISO transforms operating on input columns :math:`X_1, X_2, \ldots, X_n`, the transformation produces
    a single output :math:`Y` through a function :math:`f`:

    .. math::
        Y_t = f(X_{1,t}, X_{2,t}, \ldots, X_{n,t}, \theta)

    where :math:`\theta` represents transform-specific parameters (e.g., window sizes, weights, thresholds).

    Common examples include:

    - **Price Ratio**: :math:`Y_t = \frac{P_{1,t}}{P_{2,t}}`
    - **Spread**: :math:`Y_t = P_{1,t} - P_{2,t}`
    - **Correlation**: :math:`Y_t = \text{Corr}(X_{1,t-w:t}, X_{2,t-w:t})`
    - **VWAP**: :math:`Y_t = \frac{\sum_{i=t-w}^{t} P_i \cdot V_i}{\sum_{i=t-w}^{t} V_i}`

    **Performance Optimization:**

    MISO transforms benefit significantly from Numba compilation due to their multi-column computational requirements:

    - **Vectorized Operations**: Multiple input arrays can be processed simultaneously with optimized loops
    - **Memory Efficiency**: Dictionary-based input preparation minimizes data copying and memory allocation
    - **JIT Compilation**: Complex mathematical operations across multiple series compile to efficient machine code
    - **Parallel Processing**: Independent calculations across time periods can leverage parallel execution

    **Input Management:**

    The class provides structured input handling through:

    - **Column Validation**: Ensures all required input columns are present before computation
    - **Type Consistency**: Maintains data type integrity across multiple input series
    - **Missing Data Handling**: Provides framework for consistent NaN propagation across inputs
    - **Index Alignment**: Preserves temporal relationships when combining multiple time series

    .. note::
        Unlike :class:`SISOTransform`, MISO transforms use the output column name directly rather than
        combining input and output names. This prevents unwieldy names when multiple inputs are involved
        (e.g., prefer ``'price_ratio'`` over ``'high_low_close_price_ratio'``).

    .. note::
        When implementing Numba-compiled transforms (`_nb` method), ensure all input arrays have compatible
        dtypes to avoid compilation issues. Consider explicit type conversion in :meth:`_prepare_input_nb`
        for numerical stability across different data sources.

    Args:
        input_cols (Sequence[str]): Names of input columns required for the transformation.
            Order matters for transforms where column sequence affects computation.
        output_col (str): Name of the single output column produced by the transformation.

    Raises:
        TypeError: If input is not a pandas DataFrame during validation.
        ValueError: If any required input columns are missing from the DataFrame.
        NotImplementedError: If abstract methods from :class:`CoreTransform` are not implemented.

    Examples:
        Implementing a simple price ratio transform:

        .. code-block:: python

           class PriceRatioTransform(MISOTransform):
               def __init__(self, numerator_col: str, denominator_col: str, output_name: str = None):
                   if output_name is None:
                       output_name = f'{numerator_col}_{denominator_col}_ratio'
                   super().__init__([numerator_col, denominator_col], output_name)

               def _pd(self, x: pd.DataFrame) -> pd.Series:
                   num_col, den_col = self.requires
                   ratio = x[num_col] / x[den_col]
                   return ratio.rename(self.output_name)

               def _nb(self, x: pd.DataFrame) -> pd.Series:
                   import numba as nb

                   inputs = self._prepare_input_nb(x)
                   numerator = inputs[self.requires[0]]
                   denominator = inputs[self.requires[1]]

                   @nb.jit(nopython=True)
                   def compute_ratio(num_arr, den_arr):
                       return num_arr / den_arr

                   result = compute_ratio(numerator, denominator)
                   return self._prepare_output_nb(x.index, result)

        Implementing a rolling correlation transform:

        .. code-block:: python

           class RollingCorrelationTransform(MISOTransform):
               def __init__(self, col1: str, col2: str, window: int):
                   super().__init__([col1, col2], f'corr_{window}d')
                   self.window = window

               def _pd(self, x: pd.DataFrame) -> pd.Series:
                   col1, col2 = self.requires
                   corr = x[col1].rolling(self.window).corr(x[col2])
                   return corr.rename(self.output_name)

               def _nb(self, x: pd.DataFrame) -> pd.Series:
                   import numba as nb
                   import numpy as np

                   inputs = self._prepare_input_nb(x)
                   arr1 = inputs[self.requires[0]].astype(np.float64)
                   arr2 = inputs[self.requires[1]].astype(np.float64)

                   @nb.jit(nopython=True)
                   def rolling_correlation(x1, x2, window):
                       n = len(x1)
                       result = np.full(n, np.nan)
                       for i in range(window-1, n):
                           start_idx = i - window + 1
                           sub1 = x1[start_idx:i+1]
                           sub2 = x2[start_idx:i+1]
                           result[i] = np.corrcoef(sub1, sub2)[0, 1]
                       return result

                   corr_values = rolling_correlation(arr1, arr2, self.window)
                   return self._prepare_output_nb(x.index, corr_values)

        Using MISO transforms in practice:

           >>> # doctest: +SKIP
           >>> import pandas as pd
           >>> import numpy as np
           >>> # Sample data with multiple price series
           >>> dates = pd.date_range('2023-01-01', periods=20, freq='D')
           >>> data = pd.DataFrame({
           ...     'stock_a': np.random.randn(20).cumsum() + 100,
           ...     'stock_b': np.random.randn(20).cumsum() + 100,
           ...     'volume_a': np.random.randint(1000, 5000, 20),
           ...     'volume_b': np.random.randint(1000, 5000, 20)
           ... }, index=dates)
           >>>
           >>> # Create price ratio transform
           >>> ratio_transform = PriceRatioTransform('stock_a', 'stock_b', 'a_b_ratio')  # doctest: +SKIP
           >>> print(f"Input columns: {ratio_transform.requires}")  # doctest: +SKIP
           Input columns: ['stock_a', 'stock_b']
           >>> print(f"Output name: {ratio_transform.output_name}")  # doctest: +SKIP
           Output name: a_b_ratio
           >>>
           >>> # Apply transform
           >>> ratio_series = ratio_transform(data, backend='pd')  # doctest: +SKIP
           >>> print(f"Ratio range: {ratio_series.min():.3f} - {ratio_series.max():.3f}")  # doctest: +SKIP
           Ratio range: 0.943 - 1.089

    See Also:

        - :class:`CoreTransform`: General transform base class for multi-input/output operations.
        - :class:`SISOTransform`: Transform base class for single-input, single-output operations.
        - :class:`MIMOTransform`: Transform base class for multiple-input, multiple-output operations.
        - :class:`CrossSectionalTransform`: Specialized MISO for cross-asset relationship analysis.

    References:
        .. _`Cross-Sectional Analysis`: https://www.investopedia.com/terms/c/cross_sectional_analysis.asp
        .. _`Numba JIT Compilation`: https://numba.pydata.org/numba-doc/latest/user/jit.html
    """
    def __init__(self, input_cols: Sequence[str], output_col: str):
        super().__init__(input_cols, output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        missing_cols = [col for col in self.requires if col not in x.columns]
        if missing_cols:
            raise ValueError(f"Input columns {missing_cols} not found in DataFrame")
        return True

    def _prepare_input_nb(self, x: pd.DataFrame) -> dict[str, NDArray]:
        """
        Prepare the input data for numba functions.
        :param x: DataFrame to transform
        :return: Dict of input data for each column
        """
        return {col: x[col].values for col in self.requires}

    @property
    def output_name(self) -> str:
        """
        For MISO transforms, the output name is the same as the produces.

        :return: Output name
        """
        return self.produces[0]

    def _prepare_output_nb(self, idx: pd.Index, y: NDArray) -> pd.Series:
        """
        Prepare the output data for numba functions.
        :param idx: index of the original DataFrame
        :param y: Output data from the transform
        :return: Series with the same index as the input data
        """
        return pd.Series(y, index=idx, name=self.output_name)


class SIMOTransform(CoreTransform, ABC):
    r"""Specialized transform for single-input, multiple-output operations in financial feature engineering.

    This class extends :class:`CoreTransform` to provide an optimized interface for transforms that operate
    on exactly one input column but produce multiple related output columns. This pattern is essential in
    quantitative finance for decomposing complex indicators, generating feature sets from single time series,
    and creating comprehensive technical analysis outputs from individual price or volume streams.

    **SIMO Transform Applications:**

    The Single-Input, Multiple-Output pattern captures sophisticated analytical relationships:

    - **Technical Indicator Decomposition**: Bollinger Bands (upper, middle, lower), MACD components (line, signal, histogram)
    - **Statistical Decomposition**: Rolling statistics (mean, std, skew, kurtosis) from single price series
    - **Time Series Analysis**: Trend, seasonal, and residual components from decomposition algorithms
    - **Risk Metrics**: Multiple percentiles (VaR at different confidence levels) from returns distributions
    - **Momentum Indicators**: RSI with associated momentum, rate of change, and divergence signals
    - **Volatility Measures**: Different volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell) from OHLC data

    **Mathematical Framework:**

    For SIMO transforms operating on input column :math:`X`, the transformation produces multiple outputs
    :math:`Y_1, Y_2, \ldots, Y_m` through related functions :math:`f_1, f_2, \ldots, f_m`:

    .. math::
        Y_{1,t} = f_1(X_t, \theta_1), \quad Y_{2,t} = f_2(X_t, \theta_2), \quad \ldots, \quad Y_{m,t} = f_m(X_t, \theta_m)

    where :math:`\theta_i` represents function-specific parameters. Often, these functions are mathematically
    related or derived from common intermediate calculations.

    **Common SIMO Examples:**

    - **Bollinger Bands**:
      .. math::
          \text{Middle} = \text{SMA}(X, n), \quad \text{Upper} = \text{Middle} + k \cdot \text{Std}(X, n), \quad \text{Lower} = \text{Middle} - k \cdot \text{Std}(X, n)

    - **MACD Components**:
      .. math::
          \text{MACD} = \text{EMA}(X, 12) - \text{EMA}(X, 26), \quad \text{Signal} = \text{EMA}(\text{MACD}, 9), \quad \text{Histogram} = \text{MACD} - \text{Signal}

    - **Rolling Statistics**:
      .. math::
          \mu_t = \text{Mean}(X_{t-w:t}), \quad \sigma_t = \text{Std}(X_{t-w:t}), \quad S_t = \text{Skew}(X_{t-w:t})

    **Performance Optimization:**

    SIMO transforms are particularly well-suited for Numba compilation because:

    - **Shared Computation**: Multiple outputs often share intermediate calculations, reducing redundant operations
    - **Vectorized Processing**: Single input array can be processed once to generate multiple output arrays
    - **Memory Efficiency**: Intermediate results can be reused across output calculations
    - **Batch Operations**: All related outputs computed in single pass through input data

    **Naming Convention:**

    Following the established pattern from :class:`SISOTransform`, output columns combine the input column name
    with each transform-specific suffix:

    .. math::
        \text{output\_names} = [\text{input\_col} + \text{"\_"} + \text{output\_col}_i \text{ for } i \text{ in produces}]

    For example, Bollinger Bands on ``'close'`` prices with outputs ``['bb_upper', 'bb_middle', 'bb_lower']``
    produces ``['close_bb_upper', 'close_bb_middle', 'close_bb_lower']``.

    .. note::
        SIMO transforms excel when multiple related features are derived from the same input, sharing
        computational overhead. For unrelated outputs from the same input, consider separate SISO transforms
        for better modularity and debugging capabilities.

    .. note::
        When implementing Numba-compiled transforms, ensure all output arrays have consistent lengths and
        appropriate dtypes. The :meth:`_prepare_output_nb` method validates output count automatically.

    Args:
        input_col (str): Name of the single input column to transform.
        output_cols (Sequence[str]): Sequence of output column suffixes that will be combined with
            the input column name to create full output column names.

    Raises:
        TypeError: If input is not a pandas DataFrame during validation.
        ValueError: If the specified input column is not present in the DataFrame.
        ValueError: If the number of output arrays doesn't match the expected number of outputs.
        NotImplementedError: If abstract methods from :class:`CoreTransform` are not implemented.

    Examples:
        Implementing Bollinger Bands as a SIMO transform:

        .. code-block:: python

           class BollingerBandsTransform(SIMOTransform):
               def __init__(self, window: int = 20, std_dev: float = 2.0, input_col: str = 'close'):
                   super().__init__(input_col, ['bb_upper', 'bb_middle', 'bb_lower'])
                   self.window = window
                   self.std_dev = std_dev

               def _pd(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
                   price_series = x[self.requires[0]]

                   # Shared calculations
                   rolling_mean = price_series.rolling(self.window).mean()
                   rolling_std = price_series.rolling(self.window).std()

                   # Multiple outputs
                   bb_upper = rolling_mean + (self.std_dev * rolling_std)
                   bb_middle = rolling_mean
                   bb_lower = rolling_mean - (self.std_dev * rolling_std)

                   # Rename outputs according to SIMO convention
                   return (
                       bb_upper.rename(self.output_name[0]),
                       bb_middle.rename(self.output_name[1]),
                       bb_lower.rename(self.output_name[2])
                   )

               def _nb(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
                   import numba as nb
                   import numpy as np

                   prices = self._prepare_input_nb(x)

                   @nb.jit(nopython=True)
                   def compute_bollinger_bands(prices, window, std_dev):
                       n = len(prices)
                       upper = np.full(n, np.nan)
                       middle = np.full(n, np.nan)
                       lower = np.full(n, np.nan)

                       for i in range(window-1, n):
                           window_data = prices[i-window+1:i+1]
                           mean_val = np.mean(window_data)
                           std_val = np.std(window_data)

                           middle[i] = mean_val
                           upper[i] = mean_val + std_dev * std_val
                           lower[i] = mean_val - std_dev * std_val

                       return upper, middle, lower

                   results = compute_bollinger_bands(prices, self.window, self.std_dev)
                   return self._prepare_output_nb(x.index, results)

        Implementing rolling statistics as a SIMO transform:

        .. code-block:: python

           class RollingStatsTransform(SIMOTransform):
               def __init__(self, window: int, input_col: str = 'returns'):
                   super().__init__(input_col, ['mean', 'std', 'skew', 'kurt'])
                   self.window = window

               def _pd(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
                   series = x[self.requires[0]]
                   rolling = series.rolling(self.window)

                   mean_vals = rolling.mean()
                   std_vals = rolling.std()
                   skew_vals = rolling.skew()
                   kurt_vals = rolling.kurt()

                   return (
                       mean_vals.rename(self.output_name[0]),
                       std_vals.rename(self.output_name[1]),
                       skew_vals.rename(self.output_name[2]),
                       kurt_vals.rename(self.output_name[3])
                   )

               def _nb(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
                   import numba as nb
                   import numpy as np
                   from scipy import stats

                   data = self._prepare_input_nb(x)

                   @nb.jit(nopython=True)
                   def compute_rolling_stats(data, window):
                       n = len(data)
                       means = np.full(n, np.nan)
                       stds = np.full(n, np.nan)
                       skews = np.full(n, np.nan)
                       kurts = np.full(n, np.nan)

                       for i in range(window-1, n):
                           window_data = data[i-window+1:i+1]
                           means[i] = np.mean(window_data)
                           stds[i] = np.std(window_data)
                           # Note: Numba-compatible skew/kurtosis implementations needed

                       return means, stds, skews, kurts

                   results = compute_rolling_stats(data, self.window)
                   return self._prepare_output_nb(x.index, results)

        Using SIMO transforms in feature pipelines:

           >>> # doctest: +SKIP
           >>> import pandas as pd
           >>> import numpy as np
           >>> # Sample price data
           >>> dates = pd.date_range('2023-01-01', periods=50, freq='D')
           >>> data = pd.DataFrame({
           ...     'close': 100 + np.random.randn(50).cumsum()
           ... }, index=dates)
           >>>
           >>> # Create Bollinger Bands transform
           >>> bb_transform = BollingerBandsTransform(window=20, input_col='close')  # doctest: +SKIP
           >>> print(f"Input column: {bb_transform.requires[0]}")  # doctest: +SKIP
           Input column: close
           >>> print(f"Output names: {bb_transform.output_name}")  # doctest: +SKIP
           Output names: ['close_bb_upper', 'close_bb_middle', 'close_bb_lower']
           >>>
           >>> # Apply transform
           >>> bb_results = bb_transform(data, backend='pd')  # doctest: +SKIP
           >>> print(f"Generated {len(bb_results)} output series")  # doctest: +SKIP
           Generated 3 output series
           >>>
           >>> # Integrate into DataFrame
           >>> enhanced_data = data.copy()  # doctest: +SKIP
           >>> for i, series in enumerate(bb_results):  # doctest: +SKIP
           ...     enhanced_data[bb_transform.output_name[i]] = series  # doctest: +SKIP

    See Also:

        - :class:`CoreTransform`: General transform base class for multi-input/output operations.
        - :class:`SISOTransform`: Transform base class for single-input, single-output operations.
        - :class:`MISOTransform`: Transform base class for multiple-input, single-output operations.
        - :class:`TechnicalIndicator`: Specialized base class for comprehensive technical analysis indicators.

    References:
        .. _`Bollinger Bands`: https://www.investopedia.com/terms/b/bollingerbands.asp
        .. _`MACD Indicator`: https://www.investopedia.com/terms/m/macd.asp
        .. _`Technical Analysis Patterns`: https://www.wiley.com/en-us/Technical+Analysis+of+the+Financial+Markets%3A+A+Comprehensive+Guide+to+Trading+Methods+and+Applications-p-9780735200661
    """
    def __init__(self, input_col: str, output_cols: Sequence[str]):
        super().__init__(input_col, output_cols)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires[0]} not found in DataFrame")
        return True

    def _prepare_input_nb(self, x: pd.DataFrame) -> NDArray:
        """
        Prepare the input data for numba functions.
        :param x: DataFrame to transform
        :return: Numpy array of the input column
        """
        return x[self.requires[0]].values

    @property
    def output_name(self) -> list[str]:
        """
        Get the output names of the transform.
        For SIMO transforms, the output names are derived from the input column name.
        :return: List of output names
        """
        return [f"{self.requires[0]}_{col}" for col in self.produces]

    def _prepare_output_nb(self, idx: pd.Index, y: tuple[NDArray, ...]) -> tuple[pd.Series, ...]:
        """
        Prepare the output data for numba functions.
        :param idx: index of the original DataFrame
        :param y: Output data from the transform
        :return: Tuple of Series with the same index as the input data
        """
        if len(y) != len(self.produces):
            raise ValueError(f"Expected {len(self.produces)} outputs, got {len(y)}")
        return tuple(pd.Series(y_i, index=idx, name=name) for y_i, name in zip(y, self.output_name))


class MIMOTransform(CoreTransform, ABC):
    r"""Specialized transform for multiple-input, multiple-output operations in advanced financial feature engineering.

    This class extends :class:`CoreTransform` to provide a comprehensive interface for transforms that require
    multiple input columns and produce multiple output columns. This pattern represents the most general and
    powerful transformation capability in quantitative finance, enabling complex multi-dimensional feature
    engineering, cross-sectional analysis, and sophisticated indicator systems that capture relationships
    across multiple time series and produce coordinated output features.

    **MIMO Transform Applications:**

    The Multiple-Input, Multiple-Output pattern enables the most sophisticated analytical operations:

    - **Portfolio Analytics**: Computing multiple risk metrics (VaR, CVaR, Sharpe ratio) from price and volume series
    - **Cross-Asset Analysis**: Generating correlation matrices, beta coefficients, and cointegration vectors from multiple asset prices
    - **Factor Models**: Computing factor loadings, residuals, and explained variance from multiple input series
    - **Advanced Technical Analysis**: Multi-timeframe indicators, regime detection systems, and composite scoring models
    - **Risk Decomposition**: Breaking down portfolio risk into systematic and idiosyncratic components across multiple factors
    - **Statistical Arbitrage**: Computing spread statistics, mean reversion signals, and hedge ratios from pairs or baskets of assets

    **Mathematical Framework:**

    For MIMO transforms operating on input columns :math:`X_1, X_2, \ldots, X_n`, the transformation produces
    multiple outputs :math:`Y_1, Y_2, \ldots, Y_m` through a system of related functions:

    .. math::
        \begin{bmatrix} Y_{1,t} \\ Y_{2,t} \\ \vdots \\ Y_{m,t} \end{bmatrix} =
        \begin{bmatrix} f_1(X_{1,t}, \ldots, X_{n,t}, \theta_1) \\ f_2(X_{1,t}, \ldots, X_{n,t}, \theta_2) \\ \vdots \\ f_m(X_{1,t}, \ldots, X_{n,t}, \theta_m) \end{bmatrix}

    where :math:`\theta_i` represents function-specific parameters. The functions often share computational
    dependencies, enabling efficient batch processing and Numba compilation.

    **Output Naming Strategy:**

    Unlike the previous transform types, MIMO transforms use output names directly as specified in the
    ``output_cols`` parameter. This approach prevents unwieldy concatenated names when dealing with
    multiple inputs and outputs, and allows for semantic naming that clearly describes the transform's purpose.

    The naming philosophy follows: **descriptive and domain-specific names that clearly indicate the
    analytical purpose** rather than mechanical combinations of input column names.

    Args:
        input_cols (Sequence[str]): Names of input columns required for the transformation.
            Order may be significant for certain mathematical operations.
        output_cols (Sequence[str]): Names of output columns produced by the transformation.
            These names are used directly without modification or combination.

    Raises:
        TypeError: If input is not a pandas DataFrame during validation.
        ValueError: If any required input columns are missing from the DataFrame.
        ValueError: If the number of output arrays doesn't match the expected number of outputs.
        NotImplementedError: If abstract methods from :class:`CoreTransform` are not implemented.

    Examples:
        Implementing a portfolio risk decomposition transform:

        .. code-block:: python

           class PortfolioRiskTransform(MIMOTransform):
               def __init__(self, asset_cols: list[str], weights: np.ndarray):
                   output_names = ['portfolio_return', 'total_risk', 'systematic_risk', 'idiosyncratic_risk']
                   super().__init__(asset_cols, output_names)
                   self.weights = weights
                   self.n_assets = len(asset_cols)

               def _pd(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
                   import numpy as np

                   # Extract asset returns
                   returns = x[self.requires].values

                   # Portfolio calculations
                   portfolio_returns = returns @ self.weights

                   # Risk calculations (simplified)
                   rolling_window = 30
                   total_risk = pd.Series(portfolio_returns).rolling(rolling_window).std()

                   # Placeholder for systematic/idiosyncratic decomposition
                   systematic_risk = total_risk * 0.7  # Simplified
                   idiosyncratic_risk = total_risk * 0.3

                   return (
                       pd.Series(portfolio_returns, index=x.index, name=self.output_name[0]),
                       total_risk.rename(self.output_name[1]),
                       systematic_risk.rename(self.output_name[2]),
                       idiosyncratic_risk.rename(self.output_name[3])
                   )

               def _nb(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
                   import numba as nb
                   import numpy as np

                   inputs = self._prepare_input_nb(x)
                   returns_matrix = np.column_stack([inputs[col] for col in self.requires])

                   @nb.jit(nopython=True)
                   def compute_portfolio_risk(returns, weights, window=30):
                       n_periods = returns.shape[0]
                       portfolio_rets = returns @ weights

                       total_risk = np.full(n_periods, np.nan)
                       systematic_risk = np.full(n_periods, np.nan)
                       idiosyncratic_risk = np.full(n_periods, np.nan)

                       for i in range(window-1, n_periods):
                           window_rets = portfolio_rets[i-window+1:i+1]
                           risk_val = np.std(window_rets)
                           total_risk[i] = risk_val
                           systematic_risk[i] = risk_val * 0.7
                           idiosyncratic_risk[i] = risk_val * 0.3

                       return portfolio_rets, total_risk, systematic_risk, idiosyncratic_risk

                   results = compute_portfolio_risk(returns_matrix, self.weights)
                   return self._prepare_output_nb(x.index, results)

        Implementing a multi-asset correlation and cointegration system:

        .. code-block:: python

           class CrossAssetAnalysisTransform(MIMOTransform):
               def __init__(self, asset_cols: list[str], window: int = 60):
                   output_names = ['correlation_12', 'cointegration_stat', 'hedge_ratio', 'spread']
                   super().__init__(asset_cols[:2], output_names)  # Focus on first two assets
                   self.window = window

               def _pd(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
                   asset1, asset2 = self.requires
                   s1, s2 = x[asset1], x[asset2]

                   # Rolling correlation
                   correlation = s1.rolling(self.window).corr(s2)

                   # Simple cointegration test (placeholder)
                   spread = s1 - s2
                   cointegration_stat = spread.rolling(self.window).apply(
                       lambda x: abs(x.mean() / x.std()) if x.std() > 0 else 0
                   )

                   # Hedge ratio from rolling regression
                   hedge_ratio = s1.rolling(self.window).cov(s2) / s2.rolling(self.window).var()

                   return (
                       correlation.rename(self.output_name[0]),
                       cointegration_stat.rename(self.output_name[1]),
                       hedge_ratio.rename(self.output_name[2]),
                       spread.rename(self.output_name[3])
                   )

               def _nb(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
                   import numba as nb
                   import numpy as np

                   inputs = self._prepare_input_nb(x)
                   asset1_prices = inputs[self.requires[0]]
                   asset2_prices = inputs[self.requires[1]]

                   @nb.jit(nopython=True)
                   def compute_cross_asset_metrics(p1, p2, window):
                       n = len(p1)
                       correlation = np.full(n, np.nan)
                       cointegration = np.full(n, np.nan)
                       hedge_ratio = np.full(n, np.nan)
                       spread = p1 - p2

                       for i in range(window-1, n):
                           w1 = p1[i-window+1:i+1]
                           w2 = p2[i-window+1:i+1]
                           w_spread = spread[i-window+1:i+1]

                           # Correlation
                           correlation[i] = np.corrcoef(w1, w2)[0, 1]

                           # Cointegration statistic
                           spread_mean = np.mean(w_spread)
                           spread_std = np.std(w_spread)
                           cointegration[i] = abs(spread_mean / spread_std) if spread_std > 0 else 0

                           # Hedge ratio
                           cov_12 = np.cov(w1, w2)[0, 1]
                           var_2 = np.var(w2)
                           hedge_ratio[i] = cov_12 / var_2 if var_2 > 0 else 0

                       return correlation, cointegration, hedge_ratio, spread

                   results = compute_cross_asset_metrics(asset1_prices, asset2_prices, self.window)
                   return self._prepare_output_nb(x.index, results)

        Using MIMO transforms in complex feature pipelines:

           >>> # doctest: +SKIP
           >>> import pandas as pd
           >>> import numpy as np
           >>> # Sample multi-asset data
           >>> dates = pd.date_range('2023-01-01', periods=100, freq='D')
           >>> np.random.seed(42)
           >>> data = pd.DataFrame({
           ...     'stock_a': 100 + np.random.randn(100).cumsum(),
           ...     'stock_b': 100 + np.random.randn(100).cumsum(),
           ...     'stock_c': 100 + np.random.randn(100).cumsum()
           ... }, index=dates)
           >>>
           >>> # Create portfolio analysis transform
           >>> weights = np.array([0.4, 0.4, 0.2])
           >>> portfolio_transform = PortfolioRiskTransform(['stock_a', 'stock_b', 'stock_c'], weights)  # doctest: +SKIP
           >>> print(f"Input columns: {portfolio_transform.requires}")  # doctest: +SKIP
           Input columns: ['stock_a', 'stock_b', 'stock_c']
           >>> print(f"Output names: {portfolio_transform.output_name}")  # doctest: +SKIP
           Output names: ['portfolio_return', 'total_risk', 'systematic_risk', 'idiosyncratic_risk']
           >>>
           >>> # Apply transform
           >>> portfolio_results = portfolio_transform(data, backend='pd')  # doctest: +SKIP
           >>> print(f"Generated {len(portfolio_results)} output series")  # doctest: +SKIP
           Generated 4 output series

    See Also:

        - :class:`CoreTransform`: General transform base class providing the foundational framework.
        - :class:`SISOTransform`: Transform base class for single-input, single-output operations.
        - :class:`MISOTransform`: Transform base class for multiple-input, single-output operations.
        - :class:`SIMOTransform`: Transform base class for single-input, multiple-output operations.
        - :class:`FactorModel`: Specialized MIMO transform for factor analysis and decomposition.

    References:
        .. _`Portfolio Theory and Risk Management`: https://www.investopedia.com/terms/m/modernportfoliotheory.asp
        .. _`Principal Component Analysis in Finance`: https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis
        .. _`Numba Performance Optimization`: https://numba.pydata.org/numba-doc/latest/user/performance-tips.html
    """
    def __init__(self, input_cols: Sequence[str], output_cols: Sequence[str]):
        super().__init__(input_cols, output_cols)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        missing_cols = [col for col in self.requires if col not in x.columns]
        if missing_cols:
            raise ValueError(f"Input columns {missing_cols} not found in DataFrame")
        return True

    def _prepare_input_nb(self, x: pd.DataFrame) -> dict[str, NDArray]:
        """
        Prepare the input data for numba functions.
        :param x: DataFrame to transform
        :return: Dict of input data for each column
        """
        return {col: x[col].values for col in self.requires}

    @property
    def output_name(self) -> list[str]:
        """
        Get the output names of the transform.
        :return: List of output names
        """
        return self.produces

    def _prepare_output_nb(self, idx: pd.Index, y: tuple[NDArray]) -> tuple[pd.Series, ...]:
        """
        Prepare the output data for numba functions.
        :param idx: index of the original DataFrame
        :param y: Output data from the transform
        :return: Tuple of Series with the same index as the input data
        """
        if len(y) != len(self.produces):
            raise ValueError(f"Expected {len(self.produces)} outputs, got {len(y)}")
        return tuple(pd.Series(y_i, index=idx, name=name) for y_i, name in zip(y, self.output_name))


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
    def output_name(self) -> str|list[str]:
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
    def output_name(self) -> str|list[str]:
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
    def output_name(self) -> str|list[str]:
        if isinstance(self.produces, list) and len(self.produces) == 1:
            return self.produces[0]
        return self.produces

    def __call__(self, x, *, backend="nb"):
        result = self.transform(x, backend=backend)
        result = self.op_func(result)
        result.name = self.output_name
        return result


class MinMaxOpTransform(BaseTransform):
    """Transform that applies min or max operations between two transforms"""
    def __init__(self, left: BaseTransform, right: BaseTransform, op_name: str, op_func: Callable):
        # Combine all input requirements from both transforms
        combined_inputs = list(set(left.requires + right.requires))
        output_name = f"{op_name}({left.output_name},{right.output_name})"
        super().__init__(combined_inputs, output_name)
        self.left = left
        self.right = right
        self.op_func = op_func

    def _validate_input(self, x):
        # min/max operations are valid for SISO and MISO transforms
        if not isinstance(self.left, (SISOTransform, MISOTransform)):
            raise TypeError(f"Left transform must be SISO or MISO for {self.produces[0]} OP, got {type(self.left)}")
        if not isinstance(self.right, (SISOTransform, MISOTransform)):
            raise TypeError(f"Right transform must be SISO or MISO for {self.produces[0]} OP, got {type(self.right)}")
        return self.left._validate_input(x) and self.right._validate_input(x)

    @property
    def output_name(self) -> str|list[str]:
        if isinstance(self.produces, list) and len(self.produces) == 1:
            return self.produces[0]
        return self.produces

    def __call__(self, x, *, backend="nb"):
        left_result = self.left(x, backend=backend)
        right_result = self.right(x, backend=backend)
        result = self.op_func(left_result, right_result)
        result.name = self.output_name
        return result
