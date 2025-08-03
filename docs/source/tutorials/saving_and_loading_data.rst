.. _saving_and_loading_data:

Saving and Loading Data
=======================

This tutorial explains how to save and load preprocessed trade data using FinMLKit.

Saving Data
-----------

Save the processed data to an HDF5 file:

.. code-block:: python

   trades.save_h5("BTCUSDT.h5")

This method also stores any discontinuity information for future reference.

Inspecting the HDF5 File
------------------------

Use the `H5Inspector` class to inspect the saved file:

.. code-block:: python

   from finmlkit.bar.io import H5Inspector

   h5_info = H5Inspector("BTCUSDT.h5")
   print(h5_info.list_keys())
   print(h5_info.get_integrity_summary())

Adding Time Bars
----------------

Add 1-second time bars to the HDF5 file for quick access to aggregated data:

.. code-block:: python

   from finmlkit.bar.io import AddTimeBarH5

   AddTimeBarH5("BTCUSDT.h5").process_key('/trades/2025-07')

Reading Time Bars
-----------------

Read time bars at arbitrary frequencies:

.. code-block:: python

   from finmlkit.bar.io import TimeBarReader

   tb1min = TimeBarReader("BTCUSDT.h5").read(
       start_time="2025-07-01", end_time="2025-07-02", timeframe="1min"
   )
   print(tb1min.head())

Next Steps
----------

With the data saved and time bars added, you can proceed to build intra-bar features. Continue to the next tutorial: :ref:`building_intra_bar_features`.
