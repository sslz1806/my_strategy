# Sentiment Timing Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove remaining look-ahead and timing-alignment errors from the sentiment timing research, and publish comparable multi-index effectiveness statistics.

**Architecture:** Keep daily/forward-return mechanics in `sentiment_timing_analysis.py`, where synthetic-data regression tests can exercise them. Keep model rules, external index retrieval, tables, and plots in the Notebook; it imports the shared helpers so the main and robustness backtests share the same next-period alignment.

**Tech Stack:** Python 3.9+, Polars, Pandas, NumPy, Statsmodels, Pytest, Jupyter Notebook.

## Global Constraints

- Use `E:\working\anaconda3\envs\quant\python.exe` for all tests.
- All returns are decimal returns; a 1% return is `0.01`.
- Signals observed at a week end can only earn the following week’s returns.
- Use preceding-trading-day market value as the daily market-return weight.

---

### Task 1: Correct lagged market-cap weighting and reusable forward-return alignment

**Files:**
- Modify: `因子回测/涨跌停情绪因子/sentiment_timing_analysis.py`
- Test: `tests/test_sentiment_timing_analysis.py`

**Interfaces:**
- Produces: `build_market_forward_returns(...)` with `market_daily_ret` weighted by each stock’s preceding observation.
- Produces: `add_next_period_return(data, date_column, return_column, output_column)` for sorted Polars period data.

- [ ] Write synthetic tests where same-day market values would produce a different weighted return, and where a weekly return must be shifted onto the preceding signal week.
- [ ] Run the two tests and verify they fail against the current implementation.
- [ ] Add only the lagged-weight and generic next-period alignment implementation.
- [ ] Re-run the targeted tests and verify they pass.

### Task 2: Apply the shared alignment to all Notebook backtests

**Files:**
- Modify: `因子回测/涨跌停情绪因子/reproduce_sentiment_timing.ipynb`
- Test: `tests/test_sentiment_timing_notebook.py`

**Interfaces:**
- Consumes: `add_next_period_return(...)` from the analysis module.
- Produces: main and broad-index backtests whose positions multiply only the following week’s return; annual attribution uses the realised return week.

- [ ] Add structural regression assertions for shared next-week alignment and realised-return-year attribution.
- [ ] Run the Notebook test and verify it fails.
- [ ] Replace direct shifts and same-week broad-index multiplication with the shared helper.
- [ ] Re-run the Notebook test and verify it passes.

### Task 3: Restore the documented trend thresholds and standard weekly Sharpe

**Files:**
- Modify: `因子回测/涨跌停情绪因子/sentiment_timing_analysis.py`
- Modify: `因子回测/涨跌停情绪因子/reproduce_sentiment_timing.ipynb`
- Test: `tests/test_sentiment_timing_analysis.py`
- Test: `tests/test_sentiment_timing_notebook.py`

**Interfaces:**
- Produces: `add_trend_buy_signal(...)`, using at least 1, 3, and 4 signals in up, sideways, and down markets respectively.
- Produces: performance metrics with weekly mean-excess-return Sharpe annualized by `sqrt(52)`.

- [ ] Write tests for every market-regime threshold and for a known weekly Sharpe series.
- [ ] Run the tests and verify the current Notebook behavior fails.
- [ ] Implement the smallest shared rule helper and call it from `run_backtest`.
- [ ] Re-run the targeted tests and verify they pass.

### Task 4: Add multi-index IC/HAC and group-return output

**Files:**
- Modify: `因子回测/涨跌停情绪因子/sentiment_timing_analysis.py`
- Modify: `因子回测/涨跌停情绪因子/reproduce_sentiment_timing.ipynb`
- Test: `tests/test_sentiment_timing_analysis.py`
- Test: `tests/test_sentiment_timing_notebook.py`

**Interfaces:**
- Produces: `build_close_forward_returns(prices, horizons)` for an index close-price series.
- Produces: per-index 1/3/5/10-day IC, HAC, and G5-G1 CSV output for 沪深300、中证500、中证1000.

- [ ] Write a close-price forward-return test and a Notebook structural test for the three-index effectiveness output.
- [ ] Run the tests and verify they fail.
- [ ] Implement forward close returns and join each index’s date-aligned targets to weekly factors before calling existing effectiveness analysis.
- [ ] Re-run all sentiment timing tests.

### Task 5: Execute and inspect the research Notebook

**Files:**
- Modify only generated Notebook output and `reproduce_output/` artifacts if the full data run succeeds.

- [ ] Execute the Notebook with the `quant` interpreter and a clean kernel.
- [ ] Inspect the exported weekly table and multi-index effectiveness CSV for correct next-period dates and no missing statistical columns.
- [ ] Run `pytest tests/test_sentiment_timing_analysis.py tests/test_sentiment_timing_notebook.py -q`.
