from abc import ABC, abstractmethod
from typing import Union, Optional, Sequence, Callable
import pandas as pd
from finmlkit.utils.log import get_logger
import numpy as np
from numpy.typing import NDArray

logger = get_logger(__name__)


class BaseTransform(ABC):
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
        """
        Apply the transform to the input data.
        :param x: DataFrame or Series to transform
        :param backend: Backend to use for the transform. Can be "pd" or "nb". Default is "nb".
        :return:
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
    def __init__(self, input_cols: Union[Sequence[str], str], output_cols: Union[Sequence[str], str]):
        super().__init__(input_cols, output_cols)

    # --- public API ---------------------------------------------------------
    def __call__(self, x: pd.DataFrame, *, backend="nb") -> Union[pd.Series, tuple[pd.Series, ...]]:
        """
        Apply the transform to the input data.
        :param x: DataFrame or Series to transform
        :param backend: Backend to use for the transform. Can be "pd" or "nb". Default is "nb".
        :return:
        """
        self._validate_input(x)

        if backend == "pd":
            return self._pd(x)
        elif backend == "nb":
            return self._nb(x)
        raise ValueError(f"Unknown backend {backend!r}")


    @staticmethod
    def _check_datetime_index(x: pd.DataFrame) -> bool:
        """
        Helper function to check if the input DataFrame has a datetime index. This will be used for time based features.
        :param x: DataFrame
        :return: True if the index is a datetime index
        """
        if isinstance(x, pd.DataFrame):
            if not pd.api.types.is_datetime64_any_dtype(x.index):
                raise ValueError("Input DataFrame must have a datetime index for time-based features.")
        else:
            raise TypeError("Input must be a pandas DataFrame")

        return True

    def _get_timestamps(self, x: pd.DataFrame) -> NDArray[np.int64]:
        """
        Helper function the get timestamps nanoseconds timestamp from the input DataFrame.
        :param x: DataFrame to get timestamps from
        :return: numpy array of timestamps in nanoseconds
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
    """
    Implement a single input, single output transform on a DataFrame.
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
    """
    Implement a multiple input, single output transform on a DataFrame.
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
    """
    Implement a single input, multiple output transform on a DataFrame.
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
    """
    Implement a multiple input, multiple output transform on a DataFrame.
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