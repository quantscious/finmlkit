
Feature pipelines: Compose, FeatureKit and the computation graph
================================================================

This tutorial shows how to build robust feature pipelines using Compose and
FeatureKit, and how to visualize and leverage the underlying dependency graph
(ComputationGraph) to compute features in a valid order automatically.

Prerequisites
-------------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from finmlkit.feature.kit import Feature, FeatureKit, Compose
   from finmlkit.feature.transforms import SMA, EWMA

Dataset
-------

.. code-block:: python

   idx = pd.date_range("2024-01-01", periods=64, freq="D")
   rng = np.random.default_rng(0)
   df = pd.DataFrame({
       "close": 100 + rng.normal(0, 1, len(idx)).cumsum(),
   }, index=idx)

Chaining transforms with Compose
--------------------------------

Compose lets you chain single-output transforms into a linear pipeline. The
first transform determines the input column, and each subsequent transform
consumes the previous output.

.. code-block:: python

   t1 = SMA(3, input_col="close")
   t2 = EWMA(5, input_col=t1.output_name)  # consume SMA output
   pipeline = Compose(t1, t2)

   # Wrap in a Feature for later use in FeatureKit or math ops
   f_pipeline = Feature(pipeline)

   result = f_pipeline(df, backend="pd")
   print(result.name)  # e.g. close_sma3_ewma5

Batch execution and caching with FeatureKit
-------------------------------------------

FeatureKit runs multiple Feature objects against a DataFrame, incrementally
caching results so that dependent features can reuse previously computed
columns.

.. code-block:: python

   f_sma = Feature(SMA(5, input_col="close"))
   f_ewma = Feature(EWMA(10, input_col="close"))
   f_ratio = f_sma / f_ewma  # depends on both above

   kit = FeatureKit([f_ratio, f_sma, f_ewma], retain=["close"])  # intentionally unsorted

   # Compute features in topological order inferred from dependencies
   out = kit.build(df, backend="pd", order="topo")
   print(out.columns)

Visualizing dependencies with ComputationGraph
----------------------------------------------

FeatureKit can build a dependency graph from your features. Input nodes are
prefixed with ``input:`` and feature nodes are the output names of your
Feature objects.

.. code-block:: python

   g = kit.build_graph()
   print(g.visualize())

   # Example output (truncated):
   # ComputationGraph:
   #   input:close -> [close_ewma10, close_sma5, div(close_sma5,close_ewma10)]
   #   close_ewma10 -> [div(close_sma5,close_ewma10)]
   #   close_sma5 -> [div(close_sma5,close_ewma10)]

   # Topological order over features only (input nodes omitted):
   print(kit.topological_order())

Reproducibility: save and load pipeline configurations
------------------------------------------------------

FeatureKit and Feature support JSON-serializable configurations. You can save a
pipeline and reload it later to reproduce the same features.

.. code-block:: python

   kit.save_config("featurekit.json")
   kit2 = FeatureKit.from_config("featurekit.json")
   out2 = kit2.build(df, backend="pd", order="topo")

Tips
----

- Use ``order="topo"`` when your feature list isn’t already dependency-sorted.
- Compose is intended for single-output transforms. For multi-output steps,
  create intermediate Features or manage DataFrame columns explicitly.
- Use the pandas backend (``backend="pd"``) when developing or debugging; switch
  to Numba (``backend="nb"``) for performance once things work.


Integrating external libraries (e.g. TA-Lib) with ExternalFunction
------------------------------------------------------------------

You can integrate third-party Python libraries into your feature pipelines via
``ExternalFunction``. This allows you to call external functions (by object or
import path) as transforms while keeping consistent input/output handling and
full serialization support.

Key points:

- Accepts a Callable (recommended) or an import path string (``"pkg.mod.func"``).
- ``pass_numpy=True`` passes NumPy arrays to the external function (useful for TA-Lib).
- Supports single or multiple outputs. For multi-output functions, provide
  ``output_cols`` with matching length.
- Fully serializable: configurations round-trip via ``FeatureKit.save_config``/
  ``FeatureKit.load_config``.

Example: TA-Lib SMA/RSI using callables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import talib
   import numpy as np
   from finmlkit.feature.kit import Feature, FeatureKit
   from finmlkit.feature.transforms import ExternalFunction

   # Wrap TA-Lib indicators; pass_numpy=True for ndarray inputs
   ext_sma14 = ExternalFunction(talib.SMA, input_cols="close", output_cols="talib_sma14", args=[14], pass_numpy=True)
   ext_rsi14 = ExternalFunction(talib.RSI, input_cols="close", output_cols="talib_rsi14", args=[14], pass_numpy=True)

   f_sma14 = Feature(ext_sma14)
   f_rsi14 = Feature(ext_rsi14)

   kit = FeatureKit([f_sma14, f_rsi14], retain=["close"])  # compute both
   out = kit.build(df, backend="pd", order="topo")

   # Serialize and load back
   kit.save_config("featurekit_talib.json")
   kit2 = FeatureKit.from_config("featurekit_talib.json")
   out2 = kit2.build(df, backend="pd", order="topo")

   assert set(out.columns) == set(out2.columns)

Installation notes for TA-Lib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- TA-Lib may require platform-specific setup. Try:

  - ``pip install TA-Lib``
  - If that fails, consider ``pip install talib-binary`` (prebuilt wheels).

- When using ``pass_numpy=True``, ensure your input columns are numeric and
  free of mixed types for best compatibility.
