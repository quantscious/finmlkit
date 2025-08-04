Why FinMLKit?
=============
.. important::

   **FinMLKit** is an open-source toolbox for **financial machine learning on raw trades**. It tackles three chronic causes of unreliable results in the field—**time-based sampling bias**, **weak labels**, and **throughput constraints** that make rigorous methods hard to apply at scale—with information-driven bars, robust labeling (Triple Barrier & meta-labeling–ready), rich microstructure features (volume profile & footprint), and **Numba**-accelerated cores. The aim is simple: **help practitioners and researchers produce faster, fairer, and more reproducible studies**.

The problem we're tackling
==========================

Modern financial ML often breaks down before modeling even begins due to 3 chronic obstacles:

1. Time-based sampling bias
---------------------------

Most pipelines aggregate ticks into fixed time bars (e.g., 1-minute). Markets don’t trade information at a constant pace: activity clusters around news, liquidity events, and regime shifts. **Time bars over/under-sample** these bursts, skewing distributions and degrading any statistical assumptions you make downstream. Event-based / information-driven bars (tick, volume, dollar, **imbalance**, **run**) help align sampling with **information flow**, not clock time.

2. Inadequate labeling
----------------------

**Fixed-horizon labels** ignore path dependency and risk symmetry. A “label at *t+N*” can rate a sample as a win even if it **first** slammed through a stop-loss, or vice versa. The **Triple Barrier Method (TBM)** fixes this by assigning outcomes by whichever barrier is hit **first**: take-profit, stop-loss, or a time limit. TBM also plays well with **meta-labeling**, where you learn which primary signals to act on (or skip).

3. Performance bottlenecks
--------------------------

Realistic research needs **millions of ticks** and path-dependent evaluation. Pure-pandas loops crawl; high-granularity features (e.g., footprints), TBM, and event filters become impractical. This slows iteration and quietly biases studies toward simplified—but wrong—setups.

What FinMLKit brings
====================

Three principles
----------------

- **Simplicity** — A small set of composable building blocks: **Bars → Features → Labels → Sample Weights**. Clear inputs/outputs, minimal configuration.
- **Speed** — Hot paths are **Numba-accelerated**; memory-aware array layouts; vectorized data movement.
- **Accessibility** — Typed APIs, Sphinx docs, and examples designed for reproducibility and adoption.

Concrete outcomes
-----------------

- **Sampling bias reduced.** Advanced bar types (tick/volume/dollar/cusum) and CUSUM-like event filters align samples with information arrival rather than wall-clock time.
- **Labels that reflect reality.** TBM (and meta-labeling–ready outputs) use risk-aware, path-dependent rules.
- **Throughput that scales.** Pipelines handle tens of millions of ticks without giving up methodological rigor.

How this advances research
==========================

A lot of academic and applied work still relies on **time bars** and **fixed-window labels** because they’re convenient. That convenience often **invalidates conclusions**: results can disappear out-of-sample when labels ignore path and when sampling amplifies regime effects.

FinMLKit provides **research-grade defaults**:

- **Event-based sampling** as a first-class citizen, not an afterthought.
- **Path-aware labels** (TBM) that reflect realistic trade exits and work cleanly with meta-labeling.
- **Microstructure-informed features** that help models “see” order-flow context, not only bar closes.
- **Transparent speed**: kernels are optimized so correctness does not force you to sacrifice scale.

This combination should make it **easier to publish** and **replicate** studies that move beyond fixed-window labeling and time-bar pipelines—and to test whether reported edges survive under more realistic assumptions.

What’s different from existing libraries
========================================

FinMLKit is **complementary** to existing ecosystems and distills several ideas into a single, coherent, **raw-tick-to-labels** workflow:

A focus on **raw trade ingestion → information/volume-driven bars → microstructure features → TBM/meta-ready labels**.

The goal is to **raise the floor** on research practice by making the correct thing also the easy thing.

Open source philosophy
======================

- **Transparent by default.** Methods, benchmarks, and design choices are documented. Reproduce, critique, and extend.
- **Community-first.** Issues and PRs that add new event filters, bar variants, features, or labeling schemes are welcome.
- **Citable releases.** Archival records and versioned docs support academic use.

Call to action
==============

If you care about **robust financial ML**—and especially if you publish or rely on research—give FinMLKit a try. Run the benchmarks on your data, pressure-test the event filters and labels, and tell us where the pipeline should go next.

- **GitHub:** https://github.com/quantscious/finmlkit
- **Documentation:** https://finmlkit.readthedocs.io/
- **Zenodo (citable release):** https://zenodo.org/records/16734160

Star the repo, file issues, propose features, and share benchmark results. Let’s make **better defaults** the norm.