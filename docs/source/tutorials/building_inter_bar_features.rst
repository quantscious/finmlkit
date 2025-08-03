.. _building_inter_bar_features:

Building Inter-Bar Features
===========================

Inter-bar features are derived from aggregated bar data, such as indicators and transforms. This tutorial explains how to compute these features using `FinMLKit`.

Defining Transforms
-------------------

Transforms are reusable operations applied to bar data. For example, to create a rolling standard volatility estimator:

.. code-block:: python

   from finmlkit.feature.transforms import EWMST, ReturnT
   from finmlkit.feature.kit import Compose

   volatility_tfs = Compose(
       ReturnT(window=pd.Timedelta(hours=2), input_col="price"),
       EWMST(half_life=pd.Timedelta(hours=2))
   )

   sigma = volatility_tfs(trades.data)
   print(sigma.tail())

Custom Transforms
-----------------

You can define custom transforms by inheriting from base classes like `SISOTransform`. For example, to compute the trend slope:

.. code-block:: python

   from finmlkit.feature.base import SISOTransform
   from scipy import stats

   class TrendSlope(SISOTransform):
       def __init__(self, window: int = 24, input_col: str = "close"):
           super().__init__(input_col, f"trend_slope_{window}")
           self.window = window

       def _pd(self, x):
           series = x[self.requires[0]]
           log_series = np.log(series)
           result = pd.Series(np.nan, index=series.index, name=self.output_name)
           x_vals = np.arange(self.window)
           for i in range(self.window - 1, len(log_series)):
               window_data = log_series.iloc[i - self.window + 1:i + 1]
               if window_data.isna().any():
                   continue
               slope, _, _, _, _ = stats.linregress(x_vals, window_data.values)
               result.iloc[i] = np.degrees(np.arctan(slope))
           return result

   trend_slope_tfs = TrendSlope(window=24, input_col="close")
   trend_slope_output = trend_slope_tfs(tb5min_klines)
   print(trend_slope_output.tail())

Building Feature Kits
---------------------

Feature kits combine multiple features into a single DataFrame:

.. code-block:: python

   from finmlkit.feature.kit import FeatureKit

   fkit = FeatureKit([
       Feature(trend_slope_tfs),
       Feature(volatility_tfs)
   ])

   feature_df = fkit.build(tb5min_klines)
   print(feature_df.tail())

Next Steps
----------

With inter-bar features computed, you can proceed to build labels for supervised learning. Continue to the next tutorial: :ref:`building_labels`.
