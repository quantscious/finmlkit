
 ![FinMLKit Header](finmlkit_header.png)

[![Python Versions](https://img.shields.io/pypi/pyversions/finmlkit.svg)](https://pypi.python.org/pypi/finmlkit)
[![Platforms](https://img.shields.io/badge/Platforms-linux--64,win--64,osx--64-orange.svg?style=flat-square)](https://pypi.python.org/pypi/finmlkit)
[![PyPI Version](https://img.shields.io/pypi/v/finmlkit.svg)](https://pypi.python.org/pypi/finmlkit)
[![Build Status](https://img.shields.io/github/actions/workflow/status/quantscious/finmlkit/deploy.yml)](https://github.com/quantscious/finmlkit)
[![Documentation Status](https://readthedocs.org/projects/finmlkit/badge/?version=latest)](https://finmlkit.readthedocs.io/en/latest/?version=latest)
[![codecov](https://codecov.io/gh/quantscious/finmlkit/graph/badge.svg?token=2H6VR817RB)](https://codecov.io/gh/quantscious/finmlkit)
[![Total Downloads](https://static.pepy.tech/badge/finmlkit)](https://pepy.tech/project/finmlkit)
[![GitHub Stars](https://img.shields.io/github/stars/quantscious/finmlkit.svg)](https://github.com/quantscious/finmlkit/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/quantscious/finmlkit.svg)](https://github.com/quantscious/finmlkit/network)
[![GitHub Issues](https://img.shields.io/github/issues/quantscious/finmlkit.svg)](https://github.com/quantscious/finmlkit/issues)
[![Last Commit](https://img.shields.io/github/last-commit/quantscious/finmlkit.svg)](https://github.com/quantscious/finmlkit/commits/main)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-ff69b4)
[![MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://pypi.python.org/pypi/finmlkit)
[![DOI](https://zenodo.org/badge/867537116.svg)](https://doi.org/10.5281/zenodo.16731879)

**FinMLKit** is an open-source, lightweight **financial machine learning library** designed to be simple, blazing fast, and easy to contribute to. Whether you’re a seasoned quant or a beginner in finance and programming, FinMLKit welcomes contributions from everyone.
 
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
The documentation is available at [finmlkit.readthedocs.io](https://finmlkit.readthedocs.io/en/latest/).

## 🪵 Logger Configuration
By default, logging is directed to the console at `INFO` level, but you can change this and also enable file-based logging by setting the appropriate environment variables.
- If `LOG_FILE_PATH` is defined, logs will be written to both the specified file and the console.
- `LOG_FILE_PATH` is not set, logging defaults to console-only output.
- ging levels follow Python’s standard logging module conventions, allowing you to control verbosity as needed.
To apply these settings, export the environment variables in your terminal before running your application:
```bash
export LOG_FILE_PATH=/path/to/your/logfile.log
export FILE_LOGGER_LEVEL=DEBUG
export CONSOLE_LOGGER_LEVEL=WARNING
```
If you want to supress console output, you can set the `CONSOLE_LOGGER_LEVEL`, for example, to `WARNING`.

# 🧰 Why FinMLKit?
**FinMLKit** is an **open-source, lightweight** financial data processing library with a focus on preparing data and labels for ML models. It is specialized for High Frequency Trading (HFT) and building on the most granular data level, the price tick data (raw trades data). This enables the building of intra-bar features (e.g., footprints, flow imbalance) that provide additional information to ML models compared to conventional and ubiquitous OHLCV data. Working with large amount of raw data requires a special design approach to ensure speed and efficiency, which is why FinMLKit is built with **Numba** for high-performance computation and parallelization. To illustrate this, if we were to aggregate raw trades data into OHLCV bars using Pandas, it would take around 100x longer than using FinMLKit. A task that would take 1 minute in Pandas would take below 1 second with FinMLKit. In the [performance test notebook](https://github.com/quantscious/finmlkit/tree/main/examples) we did a fun comparison between **FinMLKit** and [MLFinPy](https://github.com/baobach/mlfinpy) regarding bar construction speed and demonstrated a **more than 600x** speedup. This highlights the efficiency and power of FinMLKit for processing large amount of raw financial data.

So FinMLKit is built on Python’s Numba for high-performance computation, while using Pandas only as a wrapper for easier handling of data. Numba’s **Just-In-Time (JIT)** compilation allows it to convert Python code into machine code, significantly improving performance, especially in iterative tasks where parallelization can be utilized. In contrast, Pandas, while great for structuring and managing data, is slow and cumbersome for such operations. Therefore, we use Pandas only as a wrapper for handling data, allowing it to shine where it excels, while Numba powers the core algorithmic computations for efficiency and clarity. This way, we can avoid relying on slow and elusive pandas operations and focus on efficient, more explicit codes in the core functions.

Key principles are **Simplicity**, **Speed**, and **Accessibility** (SSA):
- **Simplicity** 🧩 No complex frameworks, no elusive pandas operations, just efficient, explicit, well-documented algorithms.
- **Speed** ⚡ Core functions built with Numba for high-performance computation and parallelization in mind.
- **Accessibility** 🌍 The goal is to make it easy for anyone – regardless of their background – to contribute and enhance the library, fostering an open-source collaborative spirit.

# 🔥 Motivation & Vision
**FinMLKit** is an open-source toolkit designed to make advanced, **reproducible financial machine learning** accessible to both researchers and practitioners. 
Many existing pipelines still rely on outdated conventions like time bars, fixed-window labels, and oversimplified features—not because they are optimal, but because better alternatives are often harder to implement and scale. 
FinMLKit addresses this gap by providing a research-grade foundation for working directly with raw trade data, including information-driven bar types, path-aware labeling with the Triple Barrier Method, microstructure features like volume profiles and footprints, and sample weighting for overlapping events—all powered by high-performance, Numba-accelerated internals.

This project aims not only to offer tools, but to foster collaboration. By open-sourcing the core infrastructure, we invite contributors to improve, extend, and build on a shared foundation—raising the methodological standard across both academia and industry. FinMLKit is structured to support **reproducible research**, with clean APIs, modular design, and **citable releases** (see citation info at the bottom). 
Our vision is to **democratize access to advanced techniques**, make rigorous pipelines more practical, and accelerate the adoption of robust, transparent practices in financial ML.


# 🧱 Project Structure

## Data Preprocessing & I/O
The foundation of any financial ML pipeline is robust data handling. FinMLKit provides comprehensive tools for ingesting, preprocessing, validating, and storing high-frequency trading data at scale. **The data preprocessing module transforms raw, inconsistent trade feeds into clean, validated datasets ready for bar construction and analysis**.

**Data Ingestion & Preprocessing:**
- [x] TradesData - Raw trades preprocessing with timestamp normalization, trade merging, and side inference
- [x] Data integrity validation with gap detection and discontinuity analysis
- [x] Multi-format timestamp support (s, ms, μs, ns) with automatic unit inference
- [x] Trade ID validation and missing data percentage calculation
- [x] Memory-efficient processing with chunking support for large datasets

**Storage & Retrieval:**
- [x] HDF5-based storage with monthly partitioning for efficient time-range queries
- [x] Compressed storage with multiple backends (blosc:lz4, blosc:zstd)
- [x] Metadata-driven data discovery and range validation
- [x] Multiprocessing support for large dataset operations
- [x] H5Inspector - Comprehensive HDF5 file analysis and integrity reporting
- [x] AddTimeBarH5 - Automated time bar generation and persistence (extending the raw trade data h5 file)
- [x] TimeBarReader - Efficient time bar loading with flexible resampling capabilities

## Bars
Bars are the primary data structure in FinMLKit – constructed from preprocessed trades data –, representing the historical price data of an asset. Bars can be in the form of OHLCV (Open, High, Low, Close, Volume) or any other format that includes the necessary information for analysis (e.g. footprint data, directional features). Bars are used as input for indicators, strategies, and other components of the library. In summary, **the bars module is responsible for processing structured trades data into analytical data structures optimized for financial machine learning**.

**Data Structures:**
- [x] OHLCV bars with VWAP and trade statistics
- [x] Directional features (e.g. buy/sell tick, volume, dollars, min. cum. volume, max. cum. volume etc.)
- [x] Trade size features (e.g., are there large trade block prints in the bar?)
- [x] Bar footprints with order flow imbalance detection

**Bar Types:**
- [x] Time bars
- [x] Tick bars
- [x] Volume bars
- [x] Dollar bars
- [x] CUSUM bars
- [ ] Imbalance bars
- [ ] Run bars

## Features
Everything that processes bars data (candlestick/OHLCV, directional features, or footprints) and calculates derived values from it is considered a feature. This includes moving averages, RSI, MACD, etc. Here we are focusing on more unconventional indicators that are not commonly found in other libraries and builds on our advanced data structures like footprints, for example, **volume profile features**. Features are the building blocks of trading strategies and are used to generate signals for buying or selling assets.

**FeatureKit Framework:**
- [x] Dual-backend architecture (pandas for development and prototyping ideas, Numba for production)
- [x] SISO, MISO, SIMO, MIMO transform patterns for flexible feature engineering
- [x] Compose class for sequential transform chaining
- [x] Mathematical operations and function composition with Feature wrapper class
- [x] FeatureKit for batch feature computation with performance profiling

**Implemented Features:**
- [x] Adjusted Exponential Moving Average
- [x] Standard Volatility Estimators
- [x] Volume Profile Indicators: Commitment of Traders (COT), Buy/Sell Imbalance price levels, High Volume Nodes (HVN), Low Volume Nodes (LVN), Point of Control (POC)
- [x] Cusum Monitoring structural break feature _(Chu-Stinchcombe-White CUSUM Test on Levels based on Homm and Breitung (2011))_
- [x] And many more... Consult the [documentation](https://finmlkit.readthedocs.io/en/latest/api/finmlkit.feature.transforms.html#module-finmlkit.feature.transforms) for a complete list of implemented transform examples. Feel free to **build your own features** to your specific needs, the **framework design is given**.

## Labels
Labels are the target values that we want to predict in a supervised learning problem. Currently, Triple Barrier Method is implemented with meta-label support, which is an advanced approach in financial machine learning.

- [x] Triple Barrier Method
- [x] Meta-Labeling
- [x] Label Concurrency weights
- [x] Return Attribution weights
- [x] Class Imbalance weights

## Sampling
- [x] CUSUM Filter


# 📑 Trusted Methodologies
FinMLKit implements methods from **trusted sources**, including renowned **academic papers and books**. The primary reference is **Marcos Lopez de Prado**’s Advances in Financial Machine Learning, which lays the foundation for many of the algorithms and methods in this package.
We prioritize transparency and accuracy in both the implementation and explanation of these methodologies. Each algorithm should be accompanied by **detailed documentation** that:
- **Cites the original sources** from which the methods were derived (papers, books, and other trusted research).
- **Describes the algorithms comprehensively**, explaining the theory behind them and how they are applied in practice.
By ensuring that the algorithms are well-documented, with clear references to their origins, we aim to foster trust and enable users to fully understand the underlying mechanics of the tools they are using. This also makes it easier for contributors to extend the package, knowing exactly how each method works and what references to consult.


# 📚 Documentation
**FinMLKit** is designed to be **well-documented**, with detailed explanations of each algorithm, method, and function. It uses `reStructured` style docstrings to provide clear and concise documentation for each function, class, and module. This makes it easier for users to understand how to use the library and what each function does. It uses `Sphinx` to generate the documentation and automatically deploy it to [finmlkit.readthedocs.io](https://finmlkit.readthedocs.io/en/latest/). This way, users can access the documentation online and easily navigate through the library's features and functionalities. This framework also enables the creation of tutorials or in-depth descriptions of the methods.

# 🤝 Contribution
We aim to make **FinMLKit** as easy to contribute to as possible. Whether it’s fixing bugs, adding new features, 
or improving documentation, **your contribution matters**. Let’s work together to make the common ground for financial machine learning!

**Star** the repo, **cite** it in your work, file issues, propose features, and share benchmark results. Let’s make **better defaults** the norm.

# ⚡ Speed & Performance Tests
FinMLKit is built with speed in mind. We use Numba for high-performance computation, 
allowing us to avoid slow and elusive pandas operations and focus on efficient, 
more explicit codes in the core functions. This way, we can ensure that the library is fast and efficient, even when dealing with large datasets or complex algorithms. 

Some results are collected below to demonstrate the effectiveness of the numba framework:
- Exponentially Weighted Moving Average (EWMA) calculation: __4x speedup__ compared to Pandas function
- Standard Volatility Estimator: __8.12x speedup__ compared to Pandas implementation
- CUSUM monitoring for structural breaks: __6.25x speedup__ with parallelization compared to non-parallelized implementation.
- OHLCV Time Bar generation: **100x speedup** compared to Pandas implementation.

# 🔬 Citation

If you use FinMLKit in your research or publications, we kindly ask that you cite it. Use the _"Cite this repository"_ option in the GitHub sidebar for ready-to-use citation details in formats like `BibTeX` and `APA`. For persistent DOIs, check the Zenodo archive linked below.

 [![DOI](https://zenodo.org/badge/867537116.svg)](https://doi.org/10.5281/zenodo.16731879)