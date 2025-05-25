from abc import ABC, abstractmethod
from typing import Union, Optional, Sequence
import pandas as pd
from utils.log import get_logger
import numpy as np
from numpy.typing import NDArray

logger = get_logger(__name__)


class BaseTransform(ABC):
    requires: Union[str, list[str]]       # input column names
    produces: Union[str, list[str]]       # output column name

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
        return pd.Series(y, index=idx, name=self.output_name())


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
        return pd.Series(y, index=idx, name=self.output_name())


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

    def output_name(self) -> list[str]:
        """
        Get the output names of the transform.
        For SIMO transforms, the output names are derived from the input column name.
        :return: List of output names
        """
        return [f"{self.requires[0]}_{col}" for col in self.produces]

    def _prepare_output_nb(self, idx: pd.Index, y: tuple[NDArray]) -> tuple[pd.Series, ...]:
        """
        Prepare the output data for numba functions.
        :param idx: index of the original DataFrame
        :param y: Output data from the transform
        :return: Tuple of Series with the same index as the input data
        """
        if len(y) != len(self.produces):
            raise ValueError(f"Expected {len(self.produces)} outputs, got {len(y)}")
        return tuple(pd.Series(y_i, index=idx, name=name) for y_i, name in zip(y, self.output_name()))


class MIMO(CoreTransform, ABC):
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
        return tuple(pd.Series(y_i, index=idx, name=name) for y_i, name in zip(y, self.output_name()))


class Compose(BaseTransform):
    def __init__(self, *transforms: SISOTransform):
        requires = transforms[0].requires[0]  # First tfs determines the source column
        first_output = transforms[0].output_name()
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
                series_out = tfs(x)
            else:
                # Subsequent transforms on the output of the previous transform
                print(tfs.requires[0])
                series_out = tfs(pd.DataFrame(series_out.values, index=series_out.index, columns=[tfs.requires[0]]), backend=backend)

        # Return the final output Series with the composed name
        series_out.name = self.output_name()

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
