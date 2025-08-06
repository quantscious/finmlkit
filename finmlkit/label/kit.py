""" A API wrapper around the core numba function for better usability"""
from .tbm import triple_barrier
from .weights import average_uniqueness, return_attribution, time_decay, class_balance_weights
from finmlkit.utils.log import get_logger
from finmlkit.bar.data_model import TradesData
import pandas as pd
import numpy as np

logger = get_logger(__name__)


class TBMLabel:
    r"""Implements the Triple Barrier Method (TBM) for labeling financial events, as described by Marcos Lopez de Prado.
    This method assigns labels to events based on whether the price touches an upper barrier (take-profit), lower barrier (stop-loss),
    or a vertical time barrier first. It supports both side labeling and meta-labeling modes.

    The Triple Barrier Method is a technique for labeling outcomes in financial machine learning, particularly useful for
    creating supervised learning datasets from time-series data. It helps mitigate issues like overfitting and improves
    the informativeness of labels by considering profitability thresholds and time horizons.

    For a set of events (e.g., trading signals or cusum events), the method constructs three barriers around each event's
    starting price:

    - **Upper horizontal barrier**: Take-profit level, computed as starting price plus (target return * upper multiplier).
    - **Lower horizontal barrier**: Stop-loss level, computed as starting price minus (target return * lower multiplier).
    - **Vertical barrier**: A time-based barrier after a specified timedelta.

    The label is determined by which barrier is touched first by the price path:

    - +1 if upper barrier is touched first (profitable).
    - -1 if lower barrier is touched first (loss).
    - 0 if vertical barrier is touched first (timeout), or adjustable based on meta-labeling.

    In meta-labeling mode (``is_meta=True``), the method incorporates predictions from a primary model (via the 'side' column).
    Labels are assigned only if the primary model's direction aligns with the barrier outcome, enabling meta-models to learn
    when to trust the primary model.

    Mathematically, for an event at time :math:`t` with starting price :math:`p_t`, target return :math:`r_t` (e.g., volatility estimate),
    and horizontal multipliers :math:`(m_{low}, m_{up})`:

    .. math::
        \text{Upper barrier} = p_t \cdot (1 + r_t \cdot m_{up})

        \text{Lower barrier} = p_t \cdot (1 - r_t \cdot m_{low})

        \text{Vertical barrier} = t + \Delta t

    The label :math:`l` for the event is:

    .. math::
        l = \begin{cases}
        1, & \text{if upper barrier touched first} \\
        -1, & \text{if lower barrier touched first} \\
        0, & \text{if vertical barrier touched first}
        \end{cases}

    .. important::
        In this implementation, we are constructing binary labels: either +1 or -1 for side prediction as recommended in
        Advances in Financial Machine Learning. We introduce "vertical_touch_weights" to decrease the weights of misleading labels associated with a vertical barrier touch.
        Consider the following scenario: vertical barrier is hit slightly **above**/below the initial price resulting in **1**/-1 label,
        but the price path was very close to the **lower**/upper barrier (almost hit it). If the ML model predicted **-1**/1 for this event, we don't want to heavily punish it.

    In meta-labeling, the label is modulated by the primary side :math:`s \in \{-1, 1\}`:

    .. math::
        l_{meta} = \begin{cases}
        1, & \text{if } (s = 1 \land l = 1) \lor (s = -1 \land l = -1) \\
        0, & \text{otherwise}
        \end{cases}

    .. note::
        To disable a horizontal barrier, set its multiplier to :math:`+\infty` or :math:`-\infty`. For the vertical barrier,
        use a very large timedelta (e.g., 1000 years) to effectively disable it.

    .. note::
        This implementation supports computation of sample weights via the related :class:`SampleWeights` class.
        After labeling, use :meth:`compute_weights` to calculate information-driven weights, including:

        - **Label concurrency**: Measures overlap of event durations.
        - **Return attribution**: Attributes returns to overlapping events proportionally to their uniqueness.

        These can be combined with time decay and class balancing for final sample weights in model training using :meth:`SampleWeights.compute_final_weights`.

    .. _`Advances in Financial Machine Learning`: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086

    Args:
        features (pd.DataFrame): The events dataframe containing the return target column and optionally event indices ("event_idx" column) and features. If not provided, event indices will be computed based on timestamps.
        target_ret_col (str): The name of the target return column in the ``features`` dataframe. Typically a volatility estimator output. This is used to scale the horizontal barriers. Should be in log-return space.
        min_ret (float): Minimum required return threshold. Events where the absolute target return (scaled by max horizontal multiplier) is below this threshold will be dropped.
        horizontal_barriers (tuple[float, float]): Bottom and top (stop-loss/take-profit) horizontal barrier multipliers. The target return is multiplied by these to determine barrier widths. Use -inf/+inf to disable.
        vertical_barrier (pd.Timedelta): The temporal barrier duration. Set to a large value (e.g., pd.Timedelta(days=365*1000)) to disable.
        min_close_time (pd.Timedelta, optional): Prevents premature event closure before this minimum time. Default: pd.Timedelta(seconds=1).
        is_meta (bool, optional): Enable meta-labeling mode. If True, ``features`` must contain a 'side' column with primary model predictions (-1, 0, 1). Default: False.

    Raises:
        ValueError: If input validations fail, such as missing columns, invalid types, or empty data after filtering.

    See Also:
        :class:`SampleWeights`: For computing the final sample weights combining and normalizing average uniqueness, return attribution, time decay, and class balancing.
    """
    def __init__(self,
                 features: pd.DataFrame,
                 target_ret_col: str,
                 min_ret:float,
                 horizontal_barriers: tuple[float, float],
                 vertical_barrier: pd.Timedelta,
                 min_close_time: pd.Timedelta = pd.Timedelta(seconds=1),
                 is_meta: bool = False):
        """
        Triple barrier labeling method

        :param features: The events dataframe the return target column and optionally containing the event indices ("event_idx" column) and features. If not it will be computed based on timestamps.
        :param target_ret_col: The name of the target return column in the `features` dataframe.
            Typically, a volatility estimator output.
            This will be used to determine the horizontal barriers.
            Should be in log-return space.
        :param min_ret: Minimum required return threshold.
            Where `target_col` is below this threshold events will be dropped.
        :param horizontal_barriers: Bottom and Top (SL/TP) horizontal barrier multipliers.
            The return target will be multiplied by these multipliers. Determines the width of the horizontal barriers.
            If you want to disable the barriers, set it to -np.inf or +np.inf, respectively.
        :param vertical_barrier: The temporal barrier as timedelta. Set it to a large value to disable the vertical barrier (eg. 1000 years)
        :param min_close_time: This prevents closing the event prematurely before the minimum close time is reached. Default is 1 second.
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

        self._orig_features = self._preprocess_features(features, target_ret_col, min_ret, horizontal_barriers)
        self._features = self._orig_features
        self.target_ret_col = target_ret_col
        self.min_ret = min_ret
        self.horizontal_barriers = horizontal_barriers
        self.vertical_barrier = vertical_barrier.total_seconds()  # get vertical barrier in seconds
        self.min_close_time_sec = min_close_time.total_seconds()
        self.is_meta = is_meta

        self._out = None

    @staticmethod
    def _preprocess_features(x: pd.DataFrame, target_ret_col: str, min_ret: float,
                             horizontal_barriers: tuple[float, float]) -> pd.DataFrame:
        # Remove the leading NaNs from the features DataFrame
        first_valid_indices = [x[col].first_valid_index() for col in x.columns if
                               x[col].first_valid_index() is not None]

        if not first_valid_indices:
            raise ValueError("All columns contain only NaN values.")

        # Use the latest of the first valid indices
        start_idx = max(first_valid_indices)
        x = x.loc[start_idx:]

        # Filter out rows where the target return is below the minimum required return threshold
        max_mult = np.max(horizontal_barriers)
        x = x[x[target_ret_col].abs() * max_mult >= min_ret]
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
    def first_event_timestamp(self) -> pd.Timestamp|None:
        """
        Get the timestamp of the first event.
        :return: The timestamp of the first event.
        """
        return self._features.index[0] if not self._features.empty else None

    @property
    def last_event_timestamp(self) -> pd.Timestamp|None:
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
        last_trade_timestamp = pd.Timestamp(trades.data.timestamp.values[-1], unit='ns')
        return self._orig_features[self._orig_features.index + pd.Timedelta(self.vertical_barrier, unit="s") <= last_trade_timestamp]


    def compute_labels(self, trades: TradesData) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute the labels for the events using the triple barrier method.
        
        :param trades: The raw trades data the events will be evaluated
        :return: A tuple containing:
            - The features DataFrame with the event indices and other features.
            - A dataframe containing labels, event indices, touch indices, returns, and weights.
        """
        if not isinstance(trades, TradesData):
            raise ValueError("Trades must be an instance of TradesData.")
        self._features = self._drop_trailing_events(trades)

        if "event_idx" in self._features.columns:
            event_idx = self._features.event_idx.values
        else:
            event_idx = np.searchsorted(trades.data.timestamp.values, self._features.index.values.astype(np.int64))

        # Call the triple_barrier function
        labels, touch_idx, rets, max_rb_ratios = triple_barrier(
            timestamps=trades.data.timestamp.values,
            close=trades.data.price.values,
            event_idxs=event_idx,
            targets=self.target_returns.values,
            horizontal_barriers=self.horizontal_barriers,
            vertical_barrier=self.vertical_barrier,
            min_close_time_sec=self.min_close_time_sec,
            side=self.features['side'].values.astype(np.int8) if self.is_meta else None,
            min_ret=self.min_ret
        )

        # Construct the output DataFrame
        self._out = pd.DataFrame({
            'touch_time': pd.to_datetime(trades.data.timestamp.values[touch_idx]),
            'event_idx': event_idx,
            'touch_idx': touch_idx,
            'labels': labels,
            'returns': rets,
            'vertical_touch_weights': max_rb_ratios
        }, index=self.features.index)

        return self.features, self.full_output

    def compute_weights(self, trades: TradesData, normalized: bool = False) -> pd.DataFrame:
        """
        Computes the sample average uniqueness and return attribution.
        :param trades: Same Raw trades data passed to `compute_labels()`.
        :param normalized: Whether to normalize the weights.
        :return: DataFrame containing the sample average uniqueness and return attribution.
        """
        return SampleWeights.compute_info_weights(trades, self._out, normalized)


class SampleWeights:
    """
    A wrapper class for time decay and class balance weights calculation.
    These weights should be run on the training window part of the full dataset.
    """

    @staticmethod
    def compute_info_weights(
            trades: TradesData,
            labels: pd.DataFrame,
            normalize: bool = False,
    ) -> pd.DataFrame:
        """
        Computes the average uniqueness and (non-normalized) return attribution for the events.

        :param trades: The raw trades on which the events are evaluated
        :param labels: Labels dataframe containing event indices and touch indices (output of `compute_labels` method).
        :param normalize: Whether to normalize the returned weights.
        :return:  A pandas DataFrame containing the average uniqueness and return attribution and vertical touch weights.
        """

        if not isinstance(trades, TradesData):
            raise ValueError("Trades must be an instance of TradesData.")
        if not isinstance(labels, pd.DataFrame):
            raise ValueError("Events must be a pandas DataFrame.")
        if 'event_idx' not in labels.columns or 'touch_idx' not in labels.columns:
            raise ValueError("Events DataFrame must contain 'event_idx' and 'touch_idxs' columns.")


        # Compute average uniqueness weights
        avg_u, concurrency = average_uniqueness(
            timestamps=trades.data.timestamp.values,
            event_idxs=labels.event_idx.values,
            touch_idxs=labels.touch_idx.values
        )

        out_df = pd.DataFrame({
            'avg_uniqueness': avg_u,
        }, index=labels.index)


        # Compute return attribution weights
        info_w = return_attribution(
            event_idxs=labels.event_idx.values,
            touch_idxs=labels.touch_idx.values,
            close=trades.data.price.values,
            concurrency=concurrency,
            normalize=normalize
        )
        out_df["return_attribution"] = info_w
        # out_df["vertical_touch_weights"] = labels.vertical_touch_weights.values

        return out_df

    @staticmethod
    def compute_final_weights(
            avg_uniqueness: pd.Series,
            time_decay_intercept: float = 1.,
            return_attribution: pd.Series = None,
            vertical_touch_weights: pd.Series = None,
            labels: pd.Series = None
    ) -> pd.DataFrame:
        """
        Compute the time decay and class balance weights based on the average uniqueness and return attribution.
        Normalizes return attribution to sum up to event count.

        :param avg_uniqueness: Average uniqueness weights for the events.
        :param return_attribution: Provide unnormalized return attribution if use this as info weights instead of average uniqueness.
        :param vertical_touch_weights: Provide vertical touch weights if you want to apply them to the final weights.
        :param time_decay_intercept: The intercept for the time decay function. 1.0 means no decay, 0.0 means full decay. Negative values will erase the oldest portion of the weights.
        :param labels: Provide labels if you want to apply class balancing to the final weights.
        :return: A pandas Dataframe containing the weight parts and the combined weights.
        """
        if not isinstance(avg_uniqueness, pd.Series):
            raise ValueError("avg_uniqueness must be a pandas Series.")
        if not isinstance(time_decay_intercept, (int, float)):
            raise ValueError("time_decay_intercept must be a numeric value.")
        if not -1.0 <= time_decay_intercept <= 1.0:
            raise ValueError("time_decay_intercept must lie in [-1, 1]")

            # Check optional parameters only if provided
        if return_attribution is not None and not isinstance(return_attribution, pd.Series):
            raise ValueError("return_attribution must be a pandas Series.")
        if vertical_touch_weights is not None and not isinstance(vertical_touch_weights, pd.Series):
            raise ValueError("vertical_touch_weights must be a pandas Series.")
        if labels is not None and not isinstance(labels, pd.Series):
            raise ValueError("labels must be a pandas Series.")

            # Ensure provided Series have matching indices
        if return_attribution is not None and not avg_uniqueness.index.equals(return_attribution.index):
            raise ValueError("avg_uniqueness and return_attribution must have the same index.")
        if vertical_touch_weights is not None and not avg_uniqueness.index.equals(vertical_touch_weights.index):
            raise ValueError("avg_uniqueness and vertical_touch_weights must have the same index.")
        if labels is not None and not avg_uniqueness.index.equals(labels.index):
            raise ValueError("avg_uniqueness and labels must have the same index.")


        n_events = len(avg_uniqueness)

        # Apply time decay
        time_decay_weights = time_decay(
            avg_uniqueness=avg_uniqueness.values,
            last_weight=time_decay_intercept
        )

        out_df = pd.DataFrame({
            'time_decay_weights': time_decay_weights
        }, index=avg_uniqueness.index)

        # Normalize return attribution to sum up to event count
        if return_attribution is not None:
            if return_attribution.sum() <= 0:
                raise ValueError("Return attribution sum is zero or negative, cannot normalize.")
            return_attribution = return_attribution.values * n_events / return_attribution.sum()

            out_df["return_attribution"] = return_attribution

            # Combine base weights with time decay
            combined_weights = time_decay_weights * return_attribution
        else:
            # If return attribution is not provided, just use time decay weights
            combined_weights = time_decay_weights * avg_uniqueness.values

        if vertical_touch_weights is not None:
            out_df["vertical_touch_weights"] = vertical_touch_weights.values
            combined_weights = combined_weights * vertical_touch_weights.values

        # Normalize the weights to have a mean of 1.0
        mean_combined_weights = combined_weights.mean()
        if mean_combined_weights <= 0:
            raise ValueError("Mean of combined weights is zero or negative, cannot normalize.")

        base_weights = combined_weights / mean_combined_weights

        if labels is not None:
            unique_labels, class_weights, sum_w_class, final_weights = class_balance_weights(
                labels=labels.values,
                base_w=base_weights
            )
            # Display class balance weights in a readable format
            #weight_info = "\n".join(
            #    [f"  Class {label}: {weight:.4f}" for label, weight in zip(unique_labels, class_weights)])
            #logger.info(f"Class balance weights:\n{weight_info}")

            # Display sum of weights per class
            #sum_info = "\n".join(
            #    [f"  Class {label}: {weight:.4f}" for label, weight in zip(unique_labels, sum_w_class)])
            #logger.info(f"Sum of weights per class:\n{sum_info}")
        else:
            final_weights = base_weights

        out_df["weights"] = final_weights

        return out_df