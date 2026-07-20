# Six Sentiment Factors 5-Day Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute one self-contained Notebook that evaluates six five-day rolling limit-up/limit-down sentiment factors with multi-horizon IC and single-factor timing, then export an HTML report.

**Architecture:** The Notebook is a linear research artifact: parameters and factor metadata, daily data preparation, pure factor/benchmark functions, statistical analysis, non-overlapping timing, then reader-facing results. Only the existing daily-data reader is imported; every factor and analysis formula is defined in the Notebook.

**Tech Stack:** Python 3.9, Polars, Pandas, NumPy, SciPy, statsmodels, Matplotlib, nbformat, nbclient, nbconvert; conda environment `quant`.

## Global Constraints

- Do not modify `reproduce_sentiment_timing.ipynb` or inspect/import sibling strategy scripts.
- Keep all six factor formulas and all IC/timing logic in the new Notebook.
- Use clear Chinese Markdown, function docstrings, chart titles, labels, and maintenance comments.
- Do not add dependencies or write external CSV/PNG outputs; the executed Notebook and HTML are the deliverables.
- Preserve the user's existing dirty worktree changes.

---

### Task 1: Create the self-contained Notebook and deterministic checks

**Files:**
- Create: `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb`

**Interfaces:**
- Consumes: `read_day_data(start_date, end_date, fields, file_path)`.
- Produces: `prepare_stock_daily`, `build_daily_sentiment_factors`, `build_value_weighted_benchmark`, `add_forward_returns`, `analyze_ic`, `build_expanding_thresholds`, `run_non_overlapping_timing`, `summarize_timing`.

- [ ] **Step 1: Scaffold the analysis-report sections**

  Create cells in this order: `tl;dr`, context and parameters, data load/quality, calculation functions, synthetic checks, real factor construction, IC results, timing results, takeaways.

- [ ] **Step 2: Add synthetic failing expectations before real-data execution**

  Use a tiny two-stock calendar to assert that a five-day repeated event is counted repeatedly, ratio division by zero uses one, a suspended gap invalidates next-day event return, and all forward returns begin after the factor date.

- [ ] **Step 3: Implement pure calculation functions**

  Implement the exact formulas from the design document with typed signatures and concise Chinese docstrings. Assert unique `(code, trading_date)` keys, sorted calendar alignment, finite denominator, and no factor value before five market dates.

- [ ] **Step 4: Run structural validation**

  Run: `E:\working\anaconda3\envs\quant\python.exe -m nbformat 因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb`

  Expected: the file parses as nbformat v4 with no JSON error.

### Task 2: Add IC and timing research

**Files:**
- Modify: `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb`

**Interfaces:**
- Consumes: a daily table containing the six factor columns, `market_daily_ret`, and `future_return_{1,3,5,10}d`.
- Produces: `ic_summary`, `timing_detail`, `timing_summary`, IC heatmaps, six 2×2 NAV figures.

- [ ] **Step 1: Implement multi-horizon IC**

  Pairwise-drop missing values; compute Pearson, Spearman, direction-adjusted values and OLS with HAC covariance using `max(5, n)-1` lags. Return one row per factor/horizon and check for exactly 24 rows.

- [ ] **Step 2: Implement lagged expanding thresholds**

  Compute `shift(1).expanding(min_periods=252).quantile(0.8)` per factor. Confirm each first threshold uses at least 252 earlier non-null values and never the current observation.

- [ ] **Step 3: Implement non-overlapping timing**

  Anchor all strategies on the first common valid date. For each factor and horizon, rebalance every n dates, apply the decision only to the next n daily returns, exclude incomplete final blocks, and construct strategy/benchmark NAV by daily compounding.

- [ ] **Step 4: Implement metrics and plots**

  Report holding win rate, timing hit rate, exposure, rebalance count, annual return, maximum drawdown, Sharpe, final NAV, benchmark NAV, and excess final NAV. Plot each factor against its same-window benchmark in four horizon panels.

### Task 3: Execute, reconcile, and export HTML

**Files:**
- Modify: `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb`
- Create: `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.html`

**Interfaces:**
- Consumes: local `rq_stock_all_data` through the project reader.
- Produces: executed Notebook and standalone HTML report with embedded tables and charts.

- [ ] **Step 1: Execute from project root**

  Run: `E:\working\anaconda3\envs\quant\python.exe -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1200 因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb`

  Expected: exit code 0 and every code cell has an execution count.

- [ ] **Step 2: Reconcile outputs**

  Confirm six factor columns, 24 IC rows, 24 timing rows, one common anchor, no forward-return lookahead assertion failures, finite NAV values, and a negative raw IC expectation explicitly evaluated for跌停占比 rather than hard-coded as a pass condition.

- [ ] **Step 3: Write evidence-based tl;dr and takeaways**

  Populate the top and bottom narrative from executed `ic_summary` and `timing_summary`; state whether跌停占比 is actually negatively correlated at each horizon and distinguish statistical association from timing utility.

- [ ] **Step 4: Re-execute and export HTML**

  Run: `E:\working\anaconda3\envs\quant\python.exe -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1200 因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb`

  Then run: `E:\working\anaconda3\envs\quant\python.exe -m jupyter nbconvert --to html --output sentiment_factors_5d_research.html 因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb`

  Expected: both commands exit 0; HTML exists beside the Notebook and contains no traceback text.

## Self-Review

- The plan covers all six requested factors, daily rolling five-day windows, IC horizons, threshold directions, four timing horizons, two win rates, NAV comparisons, Chinese clarity, execution, and HTML export.
- Function names and table names are consistent across tasks; no implementation decision is deferred.
- Scope remains one Notebook plus its HTML report; the old Notebook and sibling scripts are untouched.
