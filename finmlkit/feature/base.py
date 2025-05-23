from abc import ABC, abstractmethod
from typing import Union, Optional
import pandas as pd
from utils.log import get_logger
import numpy as np
from numpy.typing import NDArray

logger = get_logger(__name__)


class BaseTransform(ABC):
    requires: list[str]        # input column names
    produces: str              # output column name

    def __init__(self, input_cols: Optional[Union[list[str], str]], output_col: str):
        self.requires = list(input_cols) if isinstance(input_cols, (list, tuple)) else [input_cols]
        self.produces = output_col

    # --- public API ---------------------------------------------------------
    def __call__(self, x: Union[pd.DataFrame, pd.Series], *, backend="nb") -> Union[pd.Series, tuple[pd.Series]]:
        """
        Apply the transform to the input data.
        :param x: DataFrame or Series to transform
        :param backend: Backend to use for the transform. Can be "pd" or "nb". Default is "nb".
        :return:
        """
        if backend == "pd":
            return self.transform_pandas(x)
        elif backend == "nb":
            return self.transform_numba(x)
        raise ValueError(f"Unknown backend {backend!r}")

    def check_input(self, x: Union[pd.DataFrame, pd.Series]) -> bool:
        """
        Check if the input columns are present in the input DataFrame.
        This method should be called before applying the transform.
        :param x: DataFrame or Series to check
        :return: True if the input is valid
        """
        if isinstance(x, pd.DataFrame):
            if len(self.requires) == 0:
                raise ValueError("No input columns specified")
            missing_cols = [c for c in self.requires if c not in x.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        elif isinstance(x, pd.Series):
            pass
        else:
            raise TypeError("Input must be a pandas DataFrame or Series")

        return True

    @staticmethod
    def _check_datetime_index(x: Union[pd.DataFrame, pd.Series]) -> bool:
        """
        Check if the input DataFrame has a datetime index. This will be used for time based features.
        :param x: DataFrame or Series to check
        :return: True if the index is a datetime index
        """
        if isinstance(x, pd.DataFrame):
            if not pd.api.types.is_datetime64_any_dtype(x.index):
                raise ValueError("Input DataFrame must have a datetime index")
        elif isinstance(x, pd.Series):
            if not pd.api.types.is_datetime64_any_dtype(x.index):
                raise ValueError("Input Series must have a datetime index")
        else:
            raise TypeError("Input must be a pandas DataFrame or Series")

        return True

    def get_input_arrays(self, x: Union[pd.DataFrame, pd.Series]) -> Union[dict[str, NDArray], NDArray]:
        """
        Get the input data for numba. This will be used for numba based transforms.
        :param x: DataFrame or Series to transform
        :return: Dict of input data for DataFrame or array for Series
        """
        self.check_input(x)
        if isinstance(x, pd.DataFrame):
            if len(self.requires) > 1:
                return {c: x[c].values for c in self.requires}
            else:
                return self.get_input_arrays(x[self.requires[0]]) # this will be a Series
        elif isinstance(x, pd.Series):
            return x.values
        else:
            raise TypeError("Input must be a pandas DataFrame or Series")

    def get_series(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Get the input data as a pandas Series. This will be used for pandas based transforms.
        :param x: DataFrame or Series to transform
        :return: Series with the same index as the input data
        """
        self.check_input(x)
        if isinstance(x, pd.DataFrame):
            if len(self.requires) > 1:
                raise ValueError("Input DataFrame has multiple columns, cannot convert to Series")
            return x[self.requires[0]]
        elif isinstance(x, pd.Series):
            return x
        else:
            raise TypeError("Input must be a pandas DataFrame or Series")

    def get_timestamps(self, x: Union[pd.DataFrame, pd.Series]) -> NDArray[np.int64]:
        """
        Get the timestamps from the input DataFrame or Series.
        :param x: DataFrame or Series to get timestamps from
        :return: DatetimeIndex of the input data
        """
        self._check_datetime_index(x)
        return x.index.values.astype(np.int64)

    def to_series(self, x: Union[pd.DataFrame, pd.Series], y: NDArray) -> pd.Series:
        """
        Convert the output of the transform to a pandas Series.
        :param x: DataFrame or Series to transform
        :param y: Output data from the transform
        :return: Series with the same index as the input data
        """
        return pd.Series(y, index=x.index, name=self.produces)


    # --- to be implemented by children --------------------------------------
    @abstractmethod
    def transform_pandas(self, x: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, tuple[pd.Series]]:
        """
        Transform the input data using pandas. For fast prototyping
        :param x: DataFrame or Series to transform
        """
        pass

    @abstractmethod
    def transform_numba(self, x: Union[pd.DataFrame, pd.Series])  -> Union[pd.Series, tuple[pd.Series]]:
        # Fall back to pandas if not overridden
        pass


class Compose(BaseTransform):
    def __init__(self, *transforms: BaseTransform):
        self.transforms = transforms
        produces = "_".join([t.produces for t in transforms])
        super().__init__(transforms[0].requires, produces)

    def transform_pandas(self, x: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, tuple[pd.Series]]:
        result = x
        for t in self.transforms:
            result = t.transform_pandas(result)
        return result

    def transform_numba(self, x: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, tuple[pd.Series]]:
        result = x
        for t in self.transforms:
            result = t.transform_numba(result)
        return result