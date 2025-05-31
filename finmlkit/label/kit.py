""" A API wrapper around the core numba function for better usability"""
from .tbm import triple_barrier
from .weights import average_uniqueness, return_attribution, time_decay, class_balance_weights
from finmlkit.utils.log import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)


class TBMLabel:
    def __init__(self,
                 features: pd.DataFrame,
                 target_ret_col: str,
                 min_ret:float,
                 horizontal_barriers: tuple[float, float],
                 vertical_barrier: float,
                 is_meta: bool = False):
        """
        Triple barrier labeling method

        :param features: The events dataframe (subset of `base_bars`)
            containing the event timestamps (as DatetimeIndex) and features
        :param target_ret_col: The name of the target return column in the `features` dataframe.
            Typically, a volatility estimator output.
            This will be used to determine the horizontal barriers.
            Should be in log-return space.
        :param min_ret: Minimum required return threshold.
            Where `target_col` is below this threshold events will be dropped.
        :param horizontal_barriers: Bottom and Top (SL/TP) horizontal barrier multipliers.
            The return target will be multiplied by these multipliers. Determines the width of the horizontal barriers.
            If you want to disable the barriers, set it to -np.inf or +np.inf, respectively.
        :param vertical_barrier: The temporal barrier in seconds. Set it to np.inf to disable the vertical barrier.
        :param is_meta: Side or meta labeling.
            If `True` `features` must contain `side` column containing the predictions of the primary model.
        """

        # check if target column is in features
        if target_ret_col not in features.columns:
            raise ValueError(f"Target column '{target_ret_col}' not found in features DataFrame.")
        if not isinstance(features.index, pd.DatetimeIndex):
            raise ValueError("Features index must be a DatetimeIndex.")
        if not isinstance(horizontal_barriers, tuple) or len(horizontal_barriers) != 2:
            raise ValueError("Horizontal barriers must be a tuple of two floats (bottom, top).")
        if min_ret < 0.:
            raise ValueError("Minimum return must be non-negative.")
        if is_meta:
            if 'side' not in features.columns:
                raise ValueError("For meta labeling, 'side' column must be present in features DataFrame.")
            if not pd.api.types.is_integer_dtype(features['side']):
                raise ValueError("The 'side' column must be of integer type (e.g., -1, 0, 1).")

        self._orig_features = self._preprocess_features(features, target_ret_col, min_ret)
        self._features = self._orig_features
        self.target_ret_col = target_ret_col
        self.min_ret = min_ret
        self.horizontal_barriers = horizontal_barriers
        self.vertical_barrier = vertical_barrier
        self.is_meta = is_meta

        self._out = None

    @staticmethod
    def _preprocess_features(x: pd.DataFrame, target_ret_col: str, min_ret: float) -> pd.DataFrame:
        # Remove the leading NaNs from the features DataFrame
        first_valid_indices = [x[col].first_valid_index() for col in x.columns if
                               x[col].first_valid_index() is not None]

        if not first_valid_indices:
            raise ValueError("All columns contain only NaN values.")

        # Use the latest of the first valid indices
        start_idx = max(first_valid_indices)
        x = x.loc[start_idx:]

        # Filter out rows where the target return is below the minimum required return threshold
        x = x[x[target_ret_col].abs() >= min_ret]
        if x.empty:
            raise ValueError("No valid events found after filtering by minimum return and removing leading NaNs.")

        # Check there is no nans in the target return column
        if x[target_ret_col].isna().any():
            raise ValueError(f"Target return column '{target_ret_col}' contains NaN values. Please ensure it is fully populated.")

        return x

    @property
    def event_count(self) -> int:
        """
        Get the number of events in the features DataFrame.
        :return: The number of events.
        """
        return len(self._features)

    @property
    def first_event_timestamp(self) -> pd.Timestamp:
        """
        Get the timestamp of the first event.
        :return: The timestamp of the first event.
        """
        return self._features.index[0] if not self._features.empty else None

    @property
    def last_event_timestamp(self) -> pd.Timestamp:
        """
        Get the timestamp of the last event.
        :return: The timestamp of the last event.
        """
        return self._features.index[-1] if not self._features.empty else None

    @property
    def event_range(self) -> str:
        """
        Get the range of event timestamps.
        :return: A string containing the first and last event timestamps.
        """
        event_start, event_end = self.first_event_timestamp, self.last_event_timestamp
        return f"From {event_start} to {event_end} ({self.event_count} events)"

    @property
    def features(self) -> pd.DataFrame:
        """
        Get the features corresponding the generated labels.
        I might be a subset of the original features DataFrame due to TBM evaluation window.
        :return: The features DataFrame.
        """
        return self._features

    @property
    def target_returns(self) -> pd.Series:
        """
        Get the target returns for the events.
        :return: A pandas Series containing the target returns.
        """
        if self.target_ret_col not in self._features.columns:
            raise ValueError(f"Target return column '{self.target_ret_col}' not found in features DataFrame.")
        return self._features[self.target_ret_col]

    @property
    def labels(self) -> pd.Series:
        """
        Get the labels for the events.
        :return: A pandas Series containing the labels.
        """
        if self._out is None:
            raise ValueError("Labels have not been computed yet. Call `compute_labels()` first.")
        return self._out['labels']

    @property
    def sample_weights(self) -> pd.Series:
        """
        Get the sample weights for the events.
        :return: A pandas Series containing the sample weights.
        """
        if self._out is None or 'weights' not in self._out.columns:
            raise ValueError("Sample weights have not been computed yet. Call `compute_weights()` first.")
        return self._out['weights']

    @property
    def sample_avg_uniqueness(self) -> pd.Series:
        """
        Get the average uniqueness weights for the events.
        :return: A pandas Series containing the average uniqueness weights.
        """
        if self._out is None or 'avg_uniqueness' not in self._out.columns:
            raise ValueError("Average uniqueness weights have not been computed yet. Call `compute_weights()` first.")
        return self._out['avg_uniqueness']

    @property
    def event_returns(self) -> pd.Series:
        """
        Get the log returns associated with each event.
        :return: A pandas Series containing the log returns.
        """
        if self._out is None or 'returns' not in self._out.columns:
            raise ValueError("Log returns have not been computed yet. Call `compute_labels()` first.")
        return self._out['returns']

    def _check_base_series(self, close_series: pd.Series) -> bool:
        """
        Checks whether the first event is within the base series.
        :param series: base close series.
        :return: bool
        """
        if not isinstance(close_series, pd.Series):
            raise ValueError("close_series must be a pandas Series.")
        if not isinstance(close_series.index, pd.DatetimeIndex):
            raise ValueError("Base bars index must be a DatetimeIndex.")

        if self.first_event_timestamp < close_series.index[0]:
            logger.warning(f"First event timestamp {self.first_event_timestamp} "
                           f"is before the first base series timestamp {close_series.index[0]}. "
                           "This may lead to incorrect label computation.")
            return False

        return True

    def _drop_trailing_events(self, close_series: pd.Series) -> pd.DataFrame:
        """
        We should drop the trailing events which cannot be evaluated on the full temporal window.
        :param close_series: Base close series on which the events will be evaluated.
        :return: Trimmed features DataFrame with events that are within the base series.
        """
        return self._orig_features[self.features.index + pd.Timedelta(self.vertical_barrier) <= close_series.index[-1]]


    def compute_labels(self, close_series: pd.Series) -> pd.Series:
        """
        Compute the labels for the events using the triple barrier method.
        
        :param close_series: The base price series on which the events will be evaluated
            (its frequency should be greater than the features/event dataframe)
        :return: A pandas Series containing the labels.
        """
        self._check_base_series(close_series)
        self._features = self._drop_trailing_events(close_series)

        # Call the triple_barrier function
        labels, event_idxs, touch_idxs, rets, max_rb_ratios = triple_barrier(
            timestamps=close_series.index.values.astype(np.int64),
            close=close_series.values,
            event_ts=self.features.index.values.astype(np.int64),
            targets=self.target_returns.values,
            horizontal_barriers=self.horizontal_barriers,
            vertical_barrier=self.vertical_barrier,
            side=self.features['side'].values if self.is_meta else None,
            min_ret=self.min_ret
        )

        # Construct the output DataFrame
        self._out = pd.DataFrame({
            'labels': labels,
            'event_idxs': event_idxs,
            'touch_idxs': touch_idxs,
            'returns': rets,
            'vertical_touch_weights': max_rb_ratios
        }, index=self.features.index)

        return self.labels

    def compute_weights(self, close_series: pd.Series,
                        apply_return_attribution: bool = True,
                        time_decay_last_weight: float = 0.5,
                        apply_vertical_touch_weights: bool = True,
                        apply_class_balance: bool = True,
                        ) -> pd.Series:
        """
        Computes the sample weights for the triple barrier labels.
        This method combines average uniqueness (default), return attribution, time decay, and class balance weights.

        :param close_series: The base price series on which the events are evaluated
        :param apply_return_attribution: If True, apply return attribution weights else only average uniqueness weights will be used.
        :param time_decay_last_weight: The weight assigned to the last sample in time decay. Should be in the range [0, 1].
               If 1.0, there is no decay, If negative the oldest portion (n_events * last_weight) will be erased
        :param apply_vertical_touch_weights: If True, apply vertical touch weights to the labels.
        :param apply_class_balance: If True, apply class balance weights to the labels.
        :return:  A pandas Series containing the combined sample weights for the events.
        """
        self._check_base_series(close_series)

        if self._out is None:
            raise ValueError("Labels have not been computed yet. Call `compute_labels()` first.")

        # Compute average uniqueness weights
        avg_u, concurrency = average_uniqueness(
            timestamps=close_series.index.values.astype(np.int64),
            event_idxs=self._out.event_idxs.values,
            touch_idxs=self._out.touch_idxs.values
        )
        self._out["avg_uniqueness"] = avg_u

        if apply_return_attribution:
            # Compute return attribution weights
            info_w = return_attribution(
                event_idxs=self._out.event_idxs.values,
                touch_idxs=self._out.touch_idxs.values,
                close=close_series.values,
                concurrency=concurrency
            )
        else:
            info_w = avg_u

        # Apply time decay
        weight_decay = time_decay(
            avg_uniqueness=avg_u,
            last_weight=time_decay_last_weight
        )

        # Apply vertical touch weights if specified
        vertical_touch_weights = self._out.vertical_touch_weights.values if apply_vertical_touch_weights else 1.0

        # Combine weights
        base_w = info_w * weight_decay * vertical_touch_weights

        # Ensure that the mean of the weights is 1.0
        mean_base_w = base_w.mean()
        if mean_base_w <= 0:
            raise ValueError("Something went wrong! Mean of base weights is zero or negative, cannot normalize.")
        base_w /= mean_base_w

        if apply_class_balance:
            unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(
                labels=self.labels.values,
                base_w=base_w
            )
            logger.info(f"Class balance weights: {dict(zip(unique_labels, class_weights))}")
            logger.info(f"Sum of class weights: {dict(zip(unique_labels, class_weights))}")

            self._out['weights'] = final_weights
        else:
            # If class balance is not applied, just return the base weights
            self._out['weights'] = base_w

        return self.sample_weights
