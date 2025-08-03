.. _processing_raw_data:

Processing Raw Trade Data
=========================

This tutorial demonstrates how to process raw trade data using FinMLKit's `TradesData` class.

Downloading Raw Trade Data
--------------------------

To begin, download raw trade data from Binance:

.. code-block:: bash

   curl -s "https://data.binance.vision/data/futures/um/monthly/trades/BTCUSDT/BTCUSDT-trades-2025-07.zip" -o "BTCUSDT-trades-2025-07.zip"
   curl -s "https://data.binance.vision/data/futures/um/monthly/trades/BTCUSDT/BTCUSDT-trades-2025-07.zip.CHECKSUM" -o "BTCUSDT-trades-2025-07.zip.CHECKSUM"
   shasum -a 256 -c "BTCUSDT-trades-2025-07.zip.CHECKSUM"
   unzip -o "BTCUSDT-trades-2025-07.zip"

Preprocessing the Data
----------------------

Use the `TradesData` class to preprocess the raw data:

.. code-block:: python

   import pandas as pd
   from finmlkit.bar.data_model import TradesData

   df = pd.read_csv("BTCUSDT-trades-2025-07.csv")
   trades = TradesData(
       df.time.values, df.price.values, df.qty.values,
       id=df.id.values, is_buyer_maker=df.is_buyer_maker.values,
       preprocess=True
   )

Key Features of `TradesData`
----------------------------

- **Timestamp Conversion**: Converts timestamps to nanoseconds.
- **Data Integrity Checks**: Identifies missing trades and discontinuities.
- **Trade Merging**: Merges fragmented trades with the same timestamp and price.

Inspect the processed data:

.. code-block:: python

   print(trades.discontinuities)  # Check for discontinuities
   print(trades.data.head())      # View the processed data

Next Steps
----------

Once the data is processed, you can save it for future use. Continue to the next tutorial: :ref:`saving_and_loading_data`.
