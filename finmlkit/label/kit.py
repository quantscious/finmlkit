""" A API wrapper around the core numba function for better usability"""
from .tbm import triple_barrier
from .weights import average_uniqueness, return_attribution, time_decay, class_balance_weights
from finmlkit.utils.log import get_logger
from finmlkit.bar.data_model import TradesData
import pandas as pd
import numpy as np
from numpy.typing import NDArray

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

        :param features: The events dataframe containing the event indices ("event_idx" column) and features
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
        if "event_idx" not in features.columns:
            raise ValueError("The 'event_idx' column must be present in features DataFrame.")

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

    @property
    def full_output(self) -> pd.DataFrame:
        """
        Get the full output DataFrame containing labels, event indices, touch indices, returns, and weights.
        :return: A pandas DataFrame containing the full output.
        """
        if self._out is None:
            raise ValueError("Labels have not been computed yet. Call `compute_labels()` and `compute_weights` first.")
        return self._out

    def _drop_trailing_events(self, trades: TradesData) -> pd.DataFrame:
        """
        We should drop the trailing events which cannot be evaluated on the full temporal window.
        :param trades: Raw trades data the events will be evaluated.
        :return: Trimmed features DataFrame with events that are within the base series.
        """
        last_timestamp = pd.Timestamp(trades.data.timestamp.values[-1], unit='ns')
        return self._orig_features[self.features.index + pd.Timedelta(self.vertical_barrier, unit="s") <= last_timestamp]


    def compute_labels(self, trades: TradesData) -> pd.Series:
        """
        Compute the labels for the events using the triple barrier method.
        
        :param trades: The raw trades data the events will be evaluated
        :param event_idxs: The event indices in the trades data.
        :return: A pandas Series containing the labels.
        """
        if not isinstance(trades, TradesData):
            raise ValueError("Trades must be an instance of TradesData.")
        self._features = self._drop_trailing_events(trades)

        # Call the triple_barrier function
        labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
            timestamps=trades.data.timestamp.values,
            close=trades.data.price.values,
            event_idxs=self.features.event_idx.values,
            targets=self.target_returns.values,
            horizontal_barriers=self.horizontal_barriers,
            vertical_barrier=self.vertical_barrier,
            side=self.features['side'].values if self.is_meta else None,
            min_ret=self.min_ret
        )

        # Construct the output DataFrame
        self._out = pd.DataFrame({
            'labels': labels,
            'event_idxs': self.features.event_idx.values,
            'touch_idxs': touch_idxs,
            'returns': rets,
            'vertical_touch_weights': max_rb_ratios
        }, index=self.features.index)

        return self.labels

    def compute_weights(self, trades: TradesData,
                        apply_return_attribution: bool = True,
                        time_decay_last_weight: float = 1.,
                        apply_vertical_touch_weights: bool = True,
                        apply_class_balance: bool = False,
                        ) -> pd.Series:
        """
        Computes the sample weights for the triple barrier labels.
        This method combines average uniqueness (default), return attribution, time decay, and class balance weights.

        :param trades: The raw trades on which the events are evaluated
        :param apply_return_attribution: If True, apply return attribution weights else only average uniqueness weights will be used.
        :param time_decay_last_weight: The weight assigned to the last sample in time decay. Should be in the range [0, 1].
               If 1.0, there is no decay, If negative the oldest portion (n_events * last_weight) will be erased
        :param apply_vertical_touch_weights: If True, apply vertical touch weights to the labels.
        :param apply_class_balance: If True, apply class balance weights to the labels.
        :return:  A pandas Series containing the combined sample weights for the events.
        """

        if self._out is None:
            raise ValueError("Labels have not been computed yet. Call `compute_labels()` first.")

        # Compute average uniqueness weights
        avg_u, concurrency = average_uniqueness(
            timestamps=trades.data.timestamp.values,
            event_idxs=self._out.event_idxs.values,
            touch_idxs=self._out.touch_idxs.values
        )
        self._out["avg_uniqueness"] = avg_u

        if apply_return_attribution:
            # Compute return attribution weights
            info_w = return_attribution(
                event_idxs=self._out.event_idxs.values,
                touch_idxs=self._out.touch_idxs.values,
                close=trades.data.price.values,
                concurrency=concurrency
            )
            self._out["return_attribution"] = info_w
        else:
            info_w = avg_u

        # Apply time decay
        weight_decay = time_decay(
            avg_uniqueness=avg_u,
            last_weight=time_decay_last_weight
        )
        self._out["time_decay_weight"] = weight_decay

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
            # Display class balance weights in a readable format
            weight_info = "\n".join(
                [f"  Class {label}: {weight:.4f}" for label, weight in zip(unique_labels, class_weights)])
            logger.info(f"Class balance weights:\n{weight_info}")

            # Display sum of weights per class
            sum_info = "\n".join(
                [f"  Class {label}: {weight:.4f}" for label, weight in zip(unique_labels, sum_w_class)])
            logger.info(f"Sum of weights per class:\n{sum_info}")

            self._out['weights'] = final_weights
        else:
            # If class balance is not applied, just return the base weights
            self._out['weights'] = base_w

        return self.sample_weights


class SampleWeights:
    """
    A wrapper class for time decay and class balance weights calculation.
    These weights should be run on the training window part of the full dataset.
    """
    def __init__(self, time_decay_intercept: float = 0.5, class_balancing: bool = True):
        self.time_decay_intercept = time_decay_intercept
        self.class_balancing = class_balancing

    def __call__(self, base_w: pd.Series, labels: pd.Series, avg_uniqueness: pd.Series):
        """
        Compute the sample weights based on the base weights, labels, and average uniqueness.

        :param base_w: Base weights to be adjusted.
        :param labels: Labels for class balancing.
        :param avg_uniqueness: Average uniqueness weights.
        :return: A pandas Series containing the computed sample weights.
        """
        # Apply time decay
        weight_decay = time_decay(
            avg_uniqueness=avg_uniqueness.values,
            last_weight=self.time_decay_intercept
        )

        # Combine base weights with time decay
        combined_weights = base_w * weight_decay

        # Normalize the weights to have a mean of 1.0
        mean_combined_weights = combined_weights.mean()
        if mean_combined_weights <= 0:
            raise ValueError("Mean of combined weights is zero or negative, cannot normalize.")

        combined_weights /= mean_combined_weights

        if self.class_balancing:
            unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(
                labels=labels.values,
                base_w=combined_weights
            )
            # Display class balance weights in a readable format
            weight_info = "\n".join(
                [f"  Class {label}: {weight:.4f}" for label, weight in zip(unique_labels, class_weights)])
            logger.info(f"Class balance weights:\n{weight_info}")

            # Display sum of weights per class
            sum_info = "\n".join(
                [f"  Class {label}: {weight:.4f}" for label, weight in zip(unique_labels, sum_w_class)])
            logger.info(f"Sum of weights per class:\n{sum_info}")

            return final_weights
        else:
            return combined_weights