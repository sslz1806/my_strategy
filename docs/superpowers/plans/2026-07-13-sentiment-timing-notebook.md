# Sentiment Timing Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible sentiment-timing Notebook with correct weekly factor timing and 1/3/5/10-trading-day predictive-effectiveness analysis.

**Architecture:** Put deterministic factor/forward-return/effectiveness functions in a small sibling Python module so they can be unit-tested. Keep the Notebook as an explainable orchestration and visualization layer that uses existing project data readers.

**Tech Stack:** Python 3.9 (`quant`), Polars, Pandas, NumPy, Statsmodels, Matplotlib, Pytest, nbformat/nbclient.

---

### Task 1: Add pure analytical helpers and regression tests

**Files:**

- Create: `因子回测/涨跌停情绪因子/sentiment_timing_analysis.py`
- Create: `tests/test_sentiment_timing_analysis.py`

- [ ] **Step 1: Write failing tests for signed net-limit ratio, real week-end event exclusion, trading-day forward returns, and bearish-factor direction handling.**

- [ ] **Step 2: Run the focused test file and confirm the tests fail because the module is absent.**

Run: `E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_sentiment_timing_analysis.py -q`

Expected: import failure for `sentiment_timing_analysis`.

- [ ] **Step 3: Implement only the helper functions required by the tests.**

Functions: `build_weekly_basic_factors`, `build_market_forward_returns`, and `analyze_factor_effectiveness`.

- [ ] **Step 4: Re-run the focused tests and confirm they pass.**

### Task 2: Rebuild the Notebook around tested functions

**Files:**

- Modify: `因子回测/涨跌停情绪因子/reproduce_sentiment_timing.ipynb`

- [ ] **Step 1: Replace stale cell outputs and duplicate Markdown with a clean nine-section Notebook.**

- [ ] **Step 2: Load only used daily columns, explicitly exclude non-stock/no-limit records, and print a data-quality gate.**

- [ ] **Step 3: Calculate weekly factors with ISO-week keys and event-level next-day returns.**

- [ ] **Step 4: Add a dedicated factor-effectiveness section with tables and heatmaps for 1/3/5/10 trading-day targets.**

- [ ] **Step 5: Correct the final five-factor signal aggregation, normalize the trend signal, compound weekly/index/annual returns, and keep adaptive thresholds in an explicitly labelled appendix.**

### Task 3: Execute and verify research output

**Files:**

- Modify: `因子回测/涨跌停情绪因子/reproduce_output/` (new reproducible CSV/PNG artifacts)

- [ ] **Step 1: Compile every Notebook code cell without executing it.**

- [ ] **Step 2: Execute the Notebook from the project root with the `quant` kernel/environment.**

- [ ] **Step 3: Check that the output data contains all six factors and all four forward-return horizons, is free of non-finite market returns, and has no extreme net-limit ratio values.**

- [ ] **Step 4: Inspect generated tables/figures and report actual factor results rather than stale saved outputs.**
