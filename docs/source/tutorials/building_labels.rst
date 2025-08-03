.. _building_labels:

Building Labels
===============

Labels are essential for supervised learning. This tutorial demonstrates how to build labels using the Triple Barrier Method (TBM) in `FinMLKit`.

Setting Up Target Returns
-------------------------

Target returns are often based on volatility estimators. For example:

.. code-block:: python

   from finmlkit.feature.transforms import EWMST, ReturnT
   from finmlkit.feature.kit import Compose

   vola_estim = Compose(
       ReturnT(window=pd.Timedelta(hours=1), is_log=True, input_col="price"),
       EWMST(pd.Timedelta(hours=1))
   )

   sigma = vola_estim(trades.data)
   feature_df = pd.merge_asof(feature_df, sigma, right_index=True, left_index=True, direction="backward")
   feature_df.rename(columns={sigma.name: "sigma"}, inplace=True)

Building TBM Labels
-------------------

The `TBMLabel` class computes labels based on the Triple Barrier Method:

.. code-block:: python

   from finmlkit.label.kit import TBMLabel

   MIN_RET = 0.001  # Minimum target return (e.g., 10 bps)
   vertical_window = pd.Timedelta(hours=2)  # Maximum holding time

   tbm_label = TBMLabel(
       feature_df, target_ret_col="sigma", min_ret=MIN_RET,
       horizontal_barriers=(1.5, 1.5), vertical_barrier=vertical_window
   )

   fts, lbs = tbm_label.compute_labels(trades)
   print(lbs.head())

Sample Weights
--------------

Sample weights account for label concurrence and return attribution:

.. code-block:: python

   info_weights = tbm_label.compute_weights(trades)
   print(info_weights.head())

Final Weights
-------------

Combine various factors to compute final sample weights:

.. code-block:: python

   from finmlkit.label.kit import SampleWeights

   sample_weights = SampleWeights().compute_final_weights(
       info_weights.avg_uniqueness, time_decay_intercept=0.5,
       return_attribution=info_weights.return_attribution, labels=lbs.labels
   )
   print(sample_weights.head())

Next Steps
----------

With features, labels, and sample weights ready, you can now train machine learning models. Explore the API documentation for advanced use cases.
