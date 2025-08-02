
 ![FinMLKit Header](finmlkit_header.png)
**FinMLKit** is an open-source, lightweight **financial machine learning library** designed to be simple, fast, and easy to contribute to. Whether you’re a seasoned quant or a beginner in finance and programming, FinMLKit welcomes contributions from everyone.
 
The main goal of this library is to provide a solid foundation for financial machine learning, enabling users to process raw trades data, generate different types of bars, intra-bar features (eg. footprints), bar-level features (indicators), and labels for supervised learning.

# ⚒️ Quick Start
To get started with FinMLKit, you can simply install it via pip:
```bash
pip install finmlkit
```

Or clone the repository and install it locally:
```bash
git clone https://github.com/quantscious/finmlkit.git
cd finmlkit
pip install .
```

See the [examples directory](https://github.com/quantscious/finmlkit/tree/main/examples) to learn about the practical usage of the library. How to process trades data and build bars, features, and labels for machine learning models.

## 📖 Documentation
The documentation is available at [finmlkit.readthedocs.io](https://finmlkit.readthedocs.io/).

## 🪵 Logger Configuration
By default, logging is directed to the console, but you can enable file-based logging by setting the appropriate environment variables.
- If `LOG_FILE_PATH` is defined, logs will be written to both the specified file and the console.
- `LOG_FILE_PATH` is not set, logging defaults to console-only output.
- ging levels follow Python’s standard logging module conventions, allowing you to control verbosity as needed.
To apply these settings, export the environment variables in your terminal before running your application:
```bash
export LOG_FILE_PATH=/path/to/your/logfile.log
export FILE_LOGGER_LEVEL=DEBUG
export CONSOLE_LOGGER_LEVEL=WARNING
```
By default, the logger is configured to output to the console with an `INFO` level. If you want to supress console output, you can set the `CONSOLE_LOGGER_LEVEL` to `WARNING`.

# 🧰 Why FinMLKit?
FinMLKit is an **open-source, lightweight** financial data processing library with a focus on preparing data and labels for ML models. It is specialized for High Frequency Trading (HFT) and building on the most granular data level, the price tick data (raw trades data). This enables the building of intra-bar features (e.g., footprints, fow imbalance) that provide additional information to ML models compared to conventional and ubiquitous OHLCV data. Working with large amount of raw data requires a special design approach to ensure speed and efficiency, which is why FinMLKit is built with **Numba** for high-performance computation and parallelization. To illustrate this, if we were to aggregate raw trades data into OHLCV bars using Pandas, it would take around 100x longer than using FinMLKit. A task that would take 1 minute in Pandas would take below 1 second with FinMLKit.

So FinMLKit is built on Python’s Numba for high-performance computation, while using Pandas only as a wrapper for easier handling of data. Numba’s **Just-In-Time (JIT)** compilation allows it to convert Python code into machine code, significantly improving performance, especially in iterative tasks where parallelization can be utilized. In contrast, Pandas, while great for structuring and managing data, is slow and cumbersome for such operations. Therefore, we use Pandas only as a wrapper for handling data, allowing it to shine where it excels, while Numba powers the core algorithmic computations for efficiency and clarity. This way, we can avoid relying on slow and elusive pandas operations and focus on efficient, more explicit codes in the core functions.

Key principles are **Simplicity**, **Speed**, and **Accessibility** (SSA):
- **Simplicity** 🧩 No complex frameworks, no elusive pandas operations, just efficient, explicit, well-documented algorithms.
- **Speed** ⚡ Core functions built with Numba for high-performance computation and parallelization in mind.
- **Accessibility** 🌍 The goal is to make it easy for anyone – regardless of their background – to contribute and enhance the library, fostering an open-source collaborative spirit.

# 🔥 Motivation & Vision
**FinMLKit** provides the essential tools for financial machine learning focusing on the processing of raw trades data. This library offers the core infrastructure that can be applied in countless ways to create custom strategies.
By developing this open-source foundation, contributors from around the world can enhance the tools and create something far more robust than any single entity could achieve alone. This approach allows us to leverage collective expertise and wisdom to build a powerful library that serves everyone.

By **pooling our expertise**, we can create a **stronger foundation** for financial machine learning that benefits the entire community, far exceeding what any small team could achieve alone.


# 📑 Trusted Methodologies
FinMLKit implements methods from **trusted sources**, including renowned **academic papers and books**. We avoid experimental techniques and instead focus on proven methods. Our primary reference is **Marcos Lopez de Prado**’s Advances in Financial Machine Learning, which lays the foundation for many of the algorithms and methods in this package.
We prioritize transparency and accuracy in both the implementation and explanation of these methodologies. Each algorithm is accompanied by **detailed documentation** that:
- **Cites the original sources** from which the methods were derived (papers, books, and other trusted research).
- **Describes the algorithms comprehensively**, explaining the theory behind them and how they are applied in practice.
By ensuring that the algorithms are well-documented, with clear references to their origins, we aim to foster trust and enable users to fully understand the underlying mechanics of the tools they are using. This also makes it easier for contributors to extend the package, knowing exactly how each method works and what references to consult.

# 🤝 Contribution
We aim to make **FinMLKit** as easy to contribute to as possible. Whether it’s fixing bugs, adding new features, 
or improving documentation, **your contribution matters**. Let’s work together to make the financial machine learning space better for everyone!

# 🧱 Project Structure
## Bars
Bars are the primary data structure in FinMLKit – constructed from raw trades data –, 
representing the historical price data of an asset. Bars can be in the form of OHLCV (Open, High, Low, Close, Volume) 
or any other format that includes the necessary information for analysis (e.g. footprint data, directional features). 
Bars are used as input for indicators, strategies, and other components of the library. In summary, **the bars module 
is responsible for processing unstructured raw trades data into structured data that can be used for further analysis**.

**Data Structures:**
- [x] OHLCV bars
- [x] Directional features (e.g. buy/sell tick, volume, dollars, min. cum. volume, max. cum. volume etc.)
- [x] Trade size features  (Are there large trade block prints in the bar?)
- [x] Bar footprints

**Bar Types:**
- [x] Time bars
- [x] Tick bars
- [x] Volume bars
- [x] Dollar bars
- [x] CUSUM bars
- [ ] Imbalance bars
- [ ] Run bars


## Features
Everything that processes bars data (candlestick/OHLCV, directional features, or footprints) and calculates derived values from it is considered a feature. 
This includes moving averages, RSI, MACD, etc. Here we are focusing on more unconventional indicators that are not commonly 
found in other libraries and builds on our advanced data structures like footprints, for example, **volume profile features**.
Features are the building blocks of trading strategies and are used to generate signals for buying or selling assets. 

- [x] Adjusted Exponential Moving Average
- [x] Standard Volatility Estimators
- [x] Volume Profile Indicators: Commitment of Traders (COT), Buy/Sell Imbalance price levels, High Volume Nodes (HVN), Low Volume Nodes (LVN), Point of Control (POC)
- [x] Cusum Monitoring structural break feature _(Chu-Stinchcombe-White CUSUM Test on Levels based on  Homm and Breitung (2011)_


## Labels
Labels are the target values that we want to predict in a supervised learning problem. Currently, Triple Barrier Method is implemented with meta-label support, which is an advanced approach in financial machine learning.

- [x] Triple Barrier Method
- [x] Meta-Labeling
- [x] Label Concurrency weights
- [x] Return Attribution weights
- [x] Class Imbalance weights

## Sampling
- [x] CUSUM Filter

# 📚 Documentation
FinMLKit is designed to be **well-documented**, with detailed explanations of each algorithm, method, and function. It uses reStructured style docstrings to provide clear and concise documentation for each function, class, and module. This makes it easier for users to understand how to use the library and what each function does. It uses Sphinx to generate the documentation and automatically deploy it in readthedocs.io. This way, users can access the documentation online and easily navigate through the library's features and functionalities.

# ⚡ Speed & Performance Tests
FinMLKit is built with speed in mind. We use Numba for high-performance computation, 
allowing us to avoid slow and elusive pandas operations and focus on efficient, 
more explicit codes in the core functions. This way, we can ensure that the library is fast and efficient, even when dealing with large datasets or complex algorithms. 

Some results are collected below to demonstrate the effectiveness of the numba framework:
- Exponentially Weighted Moving Average (EWMA) calculation: __4x speedup__ compared to Pandas function
- Standard Volatility Estimator: __8.12x speedup__ compared to Pandas implementation
- CUSUM monitoring for structural breaks: __6.25x speedup__ with parallelization compared to non-parallelized implementation.
- OHLCV Time Bar generation: **100x speedup** compared to Pandas implementation.

