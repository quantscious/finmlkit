.. _building_intra_bar_features:

Building Intra-Bar Features
===========================

This tutorial provides a comprehensive guide to building intra-bar features using `FinMLKit`. Intra-bar features are derived from raw trade data within a bar, such as OHLCV features, directional features, and footprint data.

Restoring Preprocessed Data
---------------------------

To begin, load the preprocessed trade data from an HDF5 file:

.. code-block:: python

   from finmlkit.bar.data_model import TradesData

   trades = TradesData.load_trades_h5("BTCUSDT.h5")
   print(trades.data.head())

Building Time Bars
------------------

Time bars aggregate trade data into fixed time intervals. For example, to create 5-minute time bars:

.. code-block:: python

   from finmlkit.bar.kit import TimeBarKit

   tb5min_kit = TimeBarKit(trades, period=pd.Timedelta(minutes=5))
   tb5min_klines = tb5min_kit.build_ohlcv()
   print(tb5min_klines.head())

Directional Features
--------------------

Directional features capture the buy/sell imbalance within a bar:

.. code-block:: python

   tb5min_directional = tb5min_kit.build_directional_features()
   print(tb5min_directional.head())

Size Distribution Features
--------------------------

Estimate the typical trade size and compute size distribution features:

.. code-block:: python

   from finmlkit.bar.io import TimeBarReader

   tbd = TimeBarReader("BTCUSDT.h5").read(timeframe="1d")
   typical_trade_size = tbd.median_trade_size.median()

   tb5min_sizedis = tb5min_kit.build_trade_size_features(
       theta=np.ones_like(tb5min_klines.close.values) * typical_trade_size
   )
   print(tb5min_sizedis.head())

Footprint Features
------------------

Footprint features provide insights into volume distribution and imbalances:

.. code-block:: python

   tb5min_fp = tb5min_kit.build_footprints()
   print(tb5min_fp.get_df().head())

Next Steps
----------

With intra-bar features computed, you can proceed to build inter-bar features. Continue to the next tutorial: :ref:`building_inter_bar_features`.
