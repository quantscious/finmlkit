# FinMLKit
**FinMLKit** is an open-source, lightweight **financial machine learning library** designed to be simple, fast, and easy to contribute to. Whether you’re a seasoned quant or a beginner in finance and programming, FinMLKit welcomes contributions from everyone.

# Why FinMLKit?
FinMLKit is an **open-source, lightweight** alternative to the well-known MLFinLab by Hudson & Thames, but with a focus on being simple, fast, and widely accessible. It is built on Python’s Numba for high-performance computation, while using Pandas only as a wrapper for easier handling of data.
FinMLKit uses Numba over Pandas for complex financial algorithms to ensure speed and precision. Numba’s **Just-In-Time (JIT)** compilation allows it to convert Python code into machine code, significantly improving performance, especially in iterative tasks where parallelization can be utilized. In contrast, Pandas, while great for structuring and managing data, is slow and cumbersome for such operations. Therefore, we use Pandas only as a wrapper for handling data, allowing it to shine where it excels, while Numba powers the core algorithmic computations for efficiency and clarity. This way, we can avoid relying on slow and elusive pandas operations and focus on efficient, more explicit codes in the core functions.

Key principles are **Simplicity**, **Speed**, and **Accessibility**:
- **Simplicity**: No complex frameworks, no elusive pandas operations, just efficient, explicit algorithms.
- **Speed**: Core functions built with Numba for high-performance computation.
- **Accessibility:** The goal is to make it easy for anyone – regardless of their background – to contribute and enhance the library, fostering an open-source collaborative spirit.

# Trusted Methodologies
FinMLKit implements methods from **trusted sources**, including renowned **academic papers and books**. We avoid experimental techniques and instead focus on proven methods. Our primary reference is **Marcos Lopez de Prado**’s Advances in Financial Machine Learning, which lays the foundation for many of the algorithms and methods in this package.
We prioritize transparency and accuracy in both the implementation and explanation of these methodologies. Each algorithm is accompanied by **detailed documentation** that:
- **Cites the original sources** from which the methods were derived (papers, books, and other trusted research).
- **Describes the algorithms comprehensively**, explaining the theory behind them and how they are applied in practice.
By ensuring that the algorithms are well-documented, with clear references to their origins, we aim to foster trust and enable users to fully understand the underlying mechanics of the tools they are using. This also makes it easier for contributors to extend the package, knowing exactly how each method works and what references to consult.

# Motivation & Vision
We argue that **building the core functionality openly does not expose alpha** and, instead, benefits everyone! **FinMLKit** provides the essential tools for financial machine learning without compromising proprietary trading strategies. This library offers the core infrastructure that can be applied in countless ways to create custom strategies, while keeping your alpha private and protected.

Unlike strategies or alphas that can be diluted when shared, the core functionality of financial machine learning can be collaboratively improved without risk. By developing this open-source foundation, contributors from around the world can enhance the tools and create something far more robust than any single entity could achieve alone. This approach allows us to leverage collective expertise and wisdom to build a powerful library that serves everyone, while allowing individual users to maintain their unique competitive edges.

By **pooling our expertise**, we can create a **stronger foundation** for financial machine learning that benefits the entire community, far exceeding what any small team could achieve alone.


# Contribution
We aim to make **FinMLKit** as easy to contribute to as possible. Whether it’s fixing bugs, adding new features, or improving documentation, **your contribution matters**. Let’s work together to make the financial machine learning space better for everyone!

# Project Structure
## Bars
Bars are the primary data structure in FinMLKit – constructed from raw trades data –, representing the historical price data of an asset. Bars can be in the form of OHLCV (Open, High, Low, Close, Volume) or any other format that includes the necessary information for analysis (e.g. footprint data). Bars are used as input for indicators, strategies, and other components of the library.

[x] - OHLCV bars
[x] - Directional features (e.g. buy/sell tick, volume, dollars, min. cum. volume, max. cum. volume etc.)
[x] - Bar footprints

[x] - Time bars
[x] - Tick bars
[x] - Volume bars
[x] - Dollar bars
[] - Imbalance bars
[] - Run bars


## Indicators
Everything that processes bars data (candlestick/OHLCV) and calculates derived values/features from it is considered an indicator. This includes moving averages, RSI, MACD, etc. Indicators are the building blocks of trading strategies and are used to generate signals for buying or selling assets. 