# 涨跌停情绪因子简化教学 Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个日频、参数化、无未来函数的教学 Notebook，用五个涨跌停情绪因子量化信号触发后的未来收益、筛选改善和平均规避收益。

**Architecture:** 扩展现有 `sentiment_timing_analysis.py`，新增三个纯计算函数，分别负责日频滚动因子、仅使用历史数据的方向分位信号、条件收益与 HAC 统计；Notebook 只负责参数、数据接口调用、结果展示和图形。所有易错日期逻辑先用合成数据测试，再创建并从头执行 Notebook。

**Tech Stack:** Python 3.9+、Polars 1.33、Pandas、NumPy、statsmodels、Matplotlib、nbformat、nbconvert、pytest；使用项目 `quant` 环境。

---

## 文件结构

- Modify: `因子回测/涨跌停情绪因子/sentiment_timing_analysis.py`：新增日频纯计算接口，保留现有周度接口。
- Modify: `tests/test_sentiment_timing_analysis.py`：锁定事件日期、窗口加权、历史阈值和规避收益。
- Create: `tests/test_simple_sentiment_timing_notebook.py`：锁定教学结构和禁用内容。
- Create: `因子回测/涨跌停情绪因子/simple_sentiment_timing.ipynb`：读者直接运行的教学 Notebook。
- Create: `因子回测/涨跌停情绪因子/simple_output/`：新 Notebook 的 CSV 和图形，不覆盖原输出。

## Task 1: 日频滚动因子

**Files:**
- Modify: `tests/test_sentiment_timing_analysis.py`
- Modify: `因子回测/涨跌停情绪因子/sentiment_timing_analysis.py`

- [ ] **Step 1: 运行现有测试作为基线**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_sentiment_timing_analysis.py -v
```

Expected: 现有 7 个测试全部通过。

- [ ] **Step 2: 编写失败测试**

在导入列表加入 `build_daily_sentiment_factors`，追加：

```python
def test_daily_factors_use_yesterday_event_and_weight_all_events() -> None:
    day_1, day_2, day_3 = date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)
    raw = pl.DataFrame(
        [
            _row("SHSE.600001", day_1, close=11.0, pct=10.0),
            _row("SHSE.600002", day_1),
            _row("SHSE.600001", day_2, close=11.0, pct=2.0),
            _row("SHSE.600002", day_2, close=11.0, pct=0.0),
            _row("SHSE.600001", day_3, pct=1.0),
            _row("SHSE.600002", day_3, pct=3.0),
        ]
    )
    last = build_daily_sentiment_factors(raw, window=2).row(-1, named=True)
    assert last["limit_up_ratio"] == pytest.approx(2 / 4)
    assert last["limit_down_ratio"] == 0.0
    assert last["limit_up_next_ret"] == pytest.approx(0.02)


def test_daily_factors_do_not_treat_suspension_gap_as_next_day() -> None:
    day_1, day_2, day_3 = date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)
    raw = pl.DataFrame(
        [
            _row("SHSE.600001", day_1, close=11.0, pct=10.0),
            _row("SHSE.600001", day_3, pct=5.0),
            _row("SHSE.600002", day_1),
            _row("SHSE.600002", day_2),
            _row("SHSE.600002", day_3),
        ]
    )
    last = build_daily_sentiment_factors(raw, window=1).row(-1, named=True)
    assert last["limit_up_event_count"] == 0
    assert last["limit_up_next_ret"] is None


def test_daily_factors_are_prefix_invariant() -> None:
    start = date(2025, 1, 6)
    rows = [
        _row(
            "SHSE.600001",
            start + timedelta(days=offset),
            close=11.0 if offset % 2 == 0 else 10.0,
            pct=float(offset),
        )
        for offset in range(5)
    ]
    full = build_daily_sentiment_factors(pl.DataFrame(rows), window=2)
    truncated = build_daily_sentiment_factors(pl.DataFrame(rows[:4]), window=2)
    assert full.head(4).to_dicts() == truncated.to_dicts()


def test_daily_factor_window_must_be_positive() -> None:
    raw = pl.DataFrame([_row("SHSE.600001", date(2025, 1, 6))])
    with pytest.raises(ValueError, match="window 必须是正整数"):
        build_daily_sentiment_factors(raw, window=0)
```

- [ ] **Step 3: 确认测试因新函数不存在而失败**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_sentiment_timing_analysis.py -k "daily_factor" -v
```

Expected: collection error 或 FAIL，指出 `build_daily_sentiment_factors` 不存在。

- [ ] **Step 4: 实现日频因子函数**

在 `_add_code_prefix` 后加入：

```python
def build_daily_sentiment_factors(
    daily_raw: pl.DataFrame,
    window: int = 5,
) -> pl.DataFrame:
    """构造截至每日收盘可得的五个日频滚动情绪因子。"""
    normalized_window = int(window)
    if normalized_window <= 0:
        raise ValueError("window 必须是正整数")
    _check_required_columns(daily_raw, _BASIC_REQUIRED_COLUMNS)

    eligible = (
        _add_code_prefix(daily_raw)
        .filter(
            pl.col("_code_prefix").is_in(_A_SHARE_PREFIXES)
            & pl.col("is_st").fill_null(True).not_()
            & pl.col("is_suspended").fill_null(True).not_()
            & pl.col("limit_up").is_finite()
            & pl.col("limit_down").is_finite()
            & (pl.col("limit_up") > 0)
            & (pl.col("limit_down") > 0)
            & pl.col("close").is_finite()
            & pl.col("pre_close").is_finite()
            & pl.col("pct").is_finite()
        )
        .with_columns(
            [
                ((pl.col("close") >= pl.col("limit_up") - 0.01)
                 & (pl.col("close") > pl.col("pre_close"))).alias("_is_limit_up"),
                ((pl.col("close") <= pl.col("limit_down") + 0.01)
                 & (pl.col("close") < pl.col("pre_close"))).alias("_is_limit_down"),
                (pl.col("pct") / 100.0).alias("_current_return"),
            ]
        )
    )
    if eligible.is_empty():
        raise ValueError("没有满足股票池条件的有效日线记录")

    market_calendar = (
        daily_raw.select("trading_date").unique().sort("trading_date")
        .with_columns(pl.col("trading_date").shift(1).alias("_previous_market_date"))
    )
    previous_events = eligible.select(
        ["code", "trading_date", "_is_limit_up", "_is_limit_down"]
    ).rename(
        {
            "trading_date": "_previous_market_date",
            "_is_limit_up": "_was_limit_up",
            "_is_limit_down": "_was_limit_down",
        }
    )
    aligned = (
        eligible.join(market_calendar, on="trading_date", how="left")
        .join(previous_events, on=["code", "_previous_market_date"], how="left")
        .with_columns(
            [
                pl.col("_was_limit_up").fill_null(False),
                pl.col("_was_limit_down").fill_null(False),
            ]
        )
    )
    daily = (
        aligned.group_by("trading_date")
        .agg(
            [
                pl.len().alias("eligible_stock_count"),
                pl.col("_is_limit_up").cast(pl.Int64).sum().alias("limit_up_count"),
                pl.col("_is_limit_down").cast(pl.Int64).sum().alias("limit_down_count"),
                pl.col("_was_limit_up").cast(pl.Int64).sum().alias("limit_up_event_count"),
                pl.col("_was_limit_down").cast(pl.Int64).sum().alias("limit_down_event_count"),
                pl.col("_current_return").filter(pl.col("_was_limit_up")).sum().alias("limit_up_return_sum"),
                pl.col("_current_return").filter(pl.col("_was_limit_down")).sum().alias("limit_down_return_sum"),
            ]
        )
        .sort("trading_date")
    )
    rolling_columns = [
        "eligible_stock_count", "limit_up_count", "limit_down_count",
        "limit_up_event_count", "limit_down_event_count",
        "limit_up_return_sum", "limit_down_return_sum",
    ]
    daily = daily.with_columns(
        [
            pl.col(column)
            .rolling_sum(window_size=normalized_window, min_samples=normalized_window)
            .alias(f"_window_{column}")
            for column in rolling_columns
        ]
    )
    return daily.with_columns(
        [
            (pl.col("_window_limit_up_count") / pl.col("_window_eligible_stock_count")).alias("limit_up_ratio"),
            (pl.col("_window_limit_down_count") / pl.col("_window_eligible_stock_count")).alias("limit_down_ratio"),
            ((pl.col("_window_limit_up_count") - pl.col("_window_limit_down_count"))
             / pl.col("_window_eligible_stock_count")).alias("net_limit_ratio"),
            pl.when(pl.col("_window_limit_up_event_count") > 0)
            .then(pl.col("_window_limit_up_return_sum") / pl.col("_window_limit_up_event_count"))
            .otherwise(None).alias("limit_up_next_ret"),
            pl.when(pl.col("_window_limit_down_event_count") > 0)
            .then(pl.col("_window_limit_down_return_sum") / pl.col("_window_limit_down_event_count"))
            .otherwise(None).alias("limit_down_next_ret"),
        ]
    ).drop([f"_window_{column}" for column in rolling_columns])
```

- [ ] **Step 5: 运行全部核心测试**

Run: `E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_sentiment_timing_analysis.py -v`

Expected: 新旧测试全部通过。

- [ ] **Step 6: 提交日频因子**

```powershell
git add -- tests/test_sentiment_timing_analysis.py '因子回测/涨跌停情绪因子/sentiment_timing_analysis.py'
git commit -m "feat: 新增日频涨跌停情绪因子"
```

## Task 2: 历史方向分位信号

**Files:**
- Modify: `tests/test_sentiment_timing_analysis.py`
- Modify: `因子回测/涨跌停情绪因子/sentiment_timing_analysis.py`

- [ ] **Step 1: 编写失败测试**

导入 `build_historical_quantile_signals`，追加：

```python
def test_historical_quantile_signal_uses_only_prior_rows() -> None:
    days = [date(2025, 1, 6) + timedelta(days=i) for i in range(5)]
    factors = pl.DataFrame(
        {"trading_date": days, "optimism": [1., 2., 3., 100., 4.], "fear": [5., 4., 3., -100., 2.]}
    )
    result = build_historical_quantile_signals(
        factors, {"optimism": 1, "fear": -1}, lookback=4, quantile=0.75, min_history=3
    )
    row = result.row(3, named=True)
    assert row["threshold_optimism"] == pytest.approx(2.5)
    assert row["threshold_fear"] == pytest.approx(3.5)
    assert row["signal_optimism"] == 1
    assert row["signal_fear"] == 1
    assert result["signal_optimism"][:3].null_count() == 3


def test_historical_quantile_signals_are_prefix_invariant() -> None:
    days = [date(2025, 1, 6) + timedelta(days=i) for i in range(8)]
    factors = pl.DataFrame({"trading_date": days, "factor": list(map(float, range(8)))})
    kwargs = {"factor_directions": {"factor": 1}, "lookback": 4, "quantile": 0.75, "min_history": 3}
    full = build_historical_quantile_signals(factors, **kwargs)
    truncated = build_historical_quantile_signals(factors.head(6), **kwargs)
    assert full.head(6).to_dicts() == truncated.to_dicts()


@pytest.mark.parametrize(
    ("lookback", "quantile", "min_history", "message"),
    [(0, 0.7, 3, "lookback"), (4, 0.5, 3, "quantile"), (4, 0.7, 5, "min_history")],
)
def test_historical_quantile_signal_validates_parameters(
    lookback: int, quantile: float, min_history: int, message: str
) -> None:
    data = pl.DataFrame({"trading_date": [date(2025, 1, 6)], "factor": [1.0]})
    with pytest.raises(ValueError, match=message):
        build_historical_quantile_signals(
            data, {"factor": 1}, lookback=lookback, quantile=quantile, min_history=min_history
        )
```

- [ ] **Step 2: 确认测试失败**

Run: `E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_sentiment_timing_analysis.py -k "historical_quantile" -v`

Expected: FAIL，指出新函数不存在。

- [ ] **Step 3: 实现历史分位信号**

在 `build_market_forward_returns` 后加入：

```python
def build_historical_quantile_signals(
    factor_data: pl.DataFrame,
    factor_directions: Mapping[str, int],
    lookback: int = 252,
    quantile: float = 0.70,
    min_history: int = 60,
) -> pl.DataFrame:
    """用 t-1 及以前的滚动分位数生成看多信号。"""
    if lookback <= 0:
        raise ValueError("lookback 必须是正整数")
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile 必须位于 (0.5, 1.0)")
    if min_history <= 0 or min_history > lookback:
        raise ValueError("min_history 必须位于 [1, lookback]")
    if not factor_directions:
        raise ValueError("factor_directions 不能为空")
    _check_required_columns(factor_data, {"trading_date", *factor_directions})

    result = factor_data.sort("trading_date")
    thresholds = []
    for factor, direction in factor_directions.items():
        if direction not in {-1, 1}:
            raise ValueError(f"{factor} 的方向必须是 1 或 -1")
        q = quantile if direction == 1 else 1.0 - quantile
        thresholds.append(
            pl.col(factor).shift(1).rolling_quantile(
                q, interpolation="linear", window_size=lookback, min_samples=min_history
            ).alias(f"threshold_{factor}")
        )
    result = result.with_columns(thresholds)

    signals = []
    for factor, direction in factor_directions.items():
        threshold = f"threshold_{factor}"
        condition = (
            pl.col(factor) >= pl.col(threshold)
            if direction == 1
            else pl.col(factor) <= pl.col(threshold)
        )
        signals.append(
            pl.when(pl.col(factor).is_null() | pl.col(threshold).is_null())
            .then(pl.lit(None, dtype=pl.Int8))
            .when(condition).then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias(f"signal_{factor}")
        )
    return result.with_columns(signals)
```

- [ ] **Step 4: 运行全模块测试并提交**

Run: `E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_sentiment_timing_analysis.py -v`

Expected: 全部通过；完整样本与截断样本的历史信号一致。

```powershell
git add -- tests/test_sentiment_timing_analysis.py '因子回测/涨跌停情绪因子/sentiment_timing_analysis.py'
git commit -m "feat: 新增无未来函数的历史分位信号"
```

## Task 3: 条件收益与规避收益统计

**Files:**
- Modify: `tests/test_sentiment_timing_analysis.py`
- Modify: `因子回测/涨跌停情绪因子/sentiment_timing_analysis.py`

- [ ] **Step 1: 编写失败测试**

导入 `analyze_daily_signal_effectiveness`，追加：

```python
def test_daily_signal_effectiveness_reports_selection_and_avoided_return() -> None:
    data = pl.DataFrame(
        {
            "optimism": [0., 1., 2., 3., 4., 5.],
            "signal_optimism": [0, 0, 0, 1, 1, 1],
            "future_return_1d": [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03],
        }
    )
    row = analyze_daily_signal_effectiveness(data, {"optimism": 1}, horizons=(1,)).iloc[0]
    assert row["n_obs"] == 6
    assert row["trigger_count"] == 3
    assert row["trigger_mean_return"] == pytest.approx(0.02)
    assert row["not_trigger_mean_return"] == pytest.approx(-0.02)
    assert row["selection_improvement"] == pytest.approx(0.04)
    assert row["avoided_return"] == pytest.approx(0.02)
    assert row["trigger_win_rate"] == pytest.approx(1.0)
    assert row["directional_spearman_ic"] > 0


def test_daily_signal_effectiveness_adjusts_bearish_direction() -> None:
    data = pl.DataFrame(
        {
            "fear": [5., 4., 3., 2., 1.],
            "signal_fear": [0, 0, 0, 1, 1],
            "future_return_1d": [-0.03, -0.02, -0.01, 0.01, 0.02],
        }
    )
    result = analyze_daily_signal_effectiveness(data, {"fear": -1}, horizons=(1,))
    assert result.loc[0, "directional_spearman_ic"] > 0
```

- [ ] **Step 2: 确认测试失败**

Run: `E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_sentiment_timing_analysis.py -k "daily_signal_effectiveness" -v`

Expected: FAIL，指出统计函数不存在。

- [ ] **Step 3: 实现统计函数**

```python
def analyze_daily_signal_effectiveness(
    data: pl.DataFrame,
    factor_directions: Mapping[str, int],
    horizons: Sequence[int] = (1, 3, 5, 10, 20),
) -> pd.DataFrame:
    """统计方向信号触发与未触发时的未来收益和平均规避收益。"""
    normalized_horizons = tuple(dict.fromkeys(int(h) for h in horizons))
    if not normalized_horizons or any(h <= 0 for h in normalized_horizons):
        raise ValueError("horizons 必须是正整数序列")
    if not factor_directions:
        raise ValueError("factor_directions 不能为空")
    required = set(factor_directions)
    required.update(f"signal_{factor}" for factor in factor_directions)
    required.update(f"future_return_{h}d" for h in normalized_horizons)
    _check_required_columns(data, required)
    frame = data.select(sorted(required)).to_pandas()

    rows = []
    for factor, direction in factor_directions.items():
        if direction not in {-1, 1}:
            raise ValueError(f"{factor} 的方向必须是 1 或 -1")
        for horizon in normalized_horizons:
            target, signal = f"future_return_{horizon}d", f"signal_{factor}"
            valid = frame[[factor, signal, target]].rename(
                columns={factor: "x", signal: "signal", target: "y"}
            ).replace([np.inf, -np.inf], np.nan).dropna()
            triggered = valid.loc[valid["signal"] == 1, "y"]
            untriggered = valid.loc[valid["signal"] == 0, "y"]
            n_obs = len(valid)
            row = {
                "factor": factor, "horizon": horizon, "n_obs": n_obs,
                "directional_spearman_ic": np.nan,
                "trigger_count": len(triggered),
                "trigger_rate": len(triggered) / n_obs if n_obs else np.nan,
                "trigger_mean_return": triggered.mean() if len(triggered) else np.nan,
                "not_trigger_mean_return": untriggered.mean() if len(untriggered) else np.nan,
                "selection_improvement": np.nan,
                "avoided_return": -untriggered.mean() if len(untriggered) else np.nan,
                "trigger_win_rate": (triggered > 0).mean() if len(triggered) else np.nan,
                "hac_beta": np.nan, "hac_t": np.nan, "hac_pvalue": np.nan,
            }
            if n_obs >= 3 and valid["x"].nunique() >= 2:
                row["directional_spearman_ic"] = direction * valid["x"].corr(valid["y"], method="spearman")
            if len(triggered) and len(untriggered):
                row["selection_improvement"] = triggered.mean() - untriggered.mean()
                fit = sm.OLS(valid["y"], sm.add_constant(valid["signal"], has_constant="add")).fit(
                    cov_type="HAC", cov_kwds={"maxlags": max(0, horizon - 1)}
                )
                row.update(
                    hac_beta=float(fit.params["signal"]),
                    hac_t=float(fit.tvalues["signal"]),
                    hac_pvalue=float(fit.pvalues["signal"]),
                )
            rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: 运行测试并提交**

Run: `E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_sentiment_timing_analysis.py -v`

Expected: 全部通过；测试样本平均规避收益为 `+2%`。

```powershell
git add -- tests/test_sentiment_timing_analysis.py '因子回测/涨跌停情绪因子/sentiment_timing_analysis.py'
git commit -m "feat: 量化情绪信号的筛选与规避收益"
```

## Task 4: Notebook 结构测试与教学 Notebook

**Files:**
- Create: `tests/test_simple_sentiment_timing_notebook.py`
- Create: `因子回测/涨跌停情绪因子/simple_sentiment_timing.ipynb`

- [ ] **Step 1: 编写结构失败测试**

```python
from pathlib import Path
import nbformat

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "因子回测" / "涨跌停情绪因子" / "simple_sentiment_timing.ipynb"

def test_simple_notebook_keeps_only_daily_factor_timing_workflow() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    code_cells = [c.source for c in notebook.cells if c.cell_type == "code"]
    code = "\n".join(code_cells)
    for name in ("FACTOR_WINDOW", "FORWARD_DAYS", "SIGNAL_LOOKBACK", "SIGNAL_QUANTILE", "MIN_HISTORY"):
        assert name in code
    for name in ("build_daily_sentiment_factors", "build_historical_quantile_signals", "analyze_daily_signal_effectiveness"):
        assert name in code
    assert code.count("read_day_data(") == 1
    assert "平均规避收益" in markdown and "t-1" in markdown and "重叠" in markdown
    assert len(code_cells) <= 8
    for excluded in ("read_min_data", "build_weekly_basic_factors", "chase_ret", "trend_position", "RqData", "quality_gate"):
        assert excluded not in code

def test_simple_notebook_has_no_execution_errors_when_outputs_exist() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert errors == []
```

- [ ] **Step 2: 确认测试因 Notebook 不存在而失败**

Run: `E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_simple_sentiment_timing_notebook.py -v`

Expected: `FileNotFoundError`。

- [ ] **Step 3: 用 `nbformat` 创建 Notebook**

使用 `nbformat.v4.new_notebook()`、`new_markdown_cell()` 和 `new_code_cell()` 生成，不直接编辑 JSON。单元内容固定为：

1. Markdown：标题、研究目标，以及“t 日收盘形成信号；阈值只用 t-1 及以前；未来收益只作标签”。
2. Code：项目路径、导入、`DATA_START_DATE/ANALYSIS_START_DATE/END_DATE`、五个集中参数、`FACTOR_LABELS` 和 `FACTOR_DIRECTIONS`。方向依次为 `1,-1,1,1,1`。
3. Markdown + Code：仅一次 `read_day_data(DATA_START_DATE, END_DATE, fields=DAILY_FIELDS, file_path=DATA_SOURCE)`，只打印行数和日期范围。
4. Markdown + Code：解释五个公式和事件加权；调用 `build_daily_sentiment_factors(..., window=FACTOR_WINDOW)`，再与 `build_market_forward_returns(..., horizons=FORWARD_DAYS)` 按日期连接。
5. Markdown + Code：解释方向阈值；调用 `build_historical_quantile_signals`，参数完整传入 `SIGNAL_LOOKBACK/SIGNAL_QUANTILE/MIN_HISTORY`。
6. Markdown + Code：调用 `analyze_daily_signal_effectiveness`，展示并保存以下列：`directional_spearman_ic, trigger_count, trigger_rate, trigger_mean_return, not_trigger_mean_return, selection_improvement, avoided_return, trigger_win_rate, hac_pvalue`。
7. Code：绘制“触发后平均收益”和“平均规避收益”两张热图并保存。
8. Code：`signal_{factor}.shift(1) * market_daily_ret` 生成五条单因子次日择时净值，与基准同图比较并保存。
9. Markdown：说明多日未来收益重叠、条件均值不可复利、规避收益正负含义，以及方向或显著性不一致时不能宣称有效。

关键代码必须使用以下参数和调用，避免隐藏默认值：

```python
FACTOR_WINDOW = 5
FORWARD_DAYS = [1, 3, 5, 10, 20]
SIGNAL_LOOKBACK = 252
SIGNAL_QUANTILE = 0.70
MIN_HISTORY = 60

daily_factors = build_daily_sentiment_factors(daily_raw, window=FACTOR_WINDOW)
market_returns = build_market_forward_returns(daily_raw, horizons=FORWARD_DAYS)
research = daily_factors.join(market_returns, on="trading_date", how="inner")
research = build_historical_quantile_signals(
    research,
    factor_directions=FACTOR_DIRECTIONS,
    lookback=SIGNAL_LOOKBACK,
    quantile=SIGNAL_QUANTILE,
    min_history=MIN_HISTORY,
)
effectiveness = analyze_daily_signal_effectiveness(
    research.filter(pl.col("trading_date") >= ANALYSIS_START_DATE),
    factor_directions=FACTOR_DIRECTIONS,
    horizons=FORWARD_DAYS,
)
```

热图单元把 `effectiveness.pivot(index="因子", columns="horizon", values=metric)` 分别用于 `trigger_mean_return` 与 `avoided_return`；次日净值单元必须先在完整预热数据上执行信号 `shift(1)`，再过滤 `ANALYSIS_START_DATE`。

- [ ] **Step 4: 运行结构测试并提交**

Run: `E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_simple_sentiment_timing_notebook.py -v`

Expected: 两个结构测试通过。

```powershell
git add -- tests/test_simple_sentiment_timing_notebook.py '因子回测/涨跌停情绪因子/simple_sentiment_timing.ipynb'
git commit -m "feat: 新增简化情绪择时教学 notebook"
```

## Task 5: 执行 Notebook 并完成回归验证

**Files:**
- Modify: `因子回测/涨跌停情绪因子/simple_sentiment_timing.ipynb`
- Create: `因子回测/涨跌停情绪因子/simple_output/daily_signal_effectiveness.csv`
- Create: `因子回测/涨跌停情绪因子/simple_output/daily_signal_heatmaps.png`
- Create: `因子回测/涨跌停情绪因子/simple_output/single_factor_timing_nav.png`

- [ ] **Step 1: 从项目根目录完整执行**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 '因子回测/涨跌停情绪因子/simple_sentiment_timing.ipynb'
```

Expected: exit code 0；所有代码单元有执行序号，无 error output；生成一个 CSV 和两张 PNG。

- [ ] **Step 2: 检查实际结果完整性**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -c "import pandas as pd; p=r'因子回测/涨跌停情绪因子/simple_output/daily_signal_effectiveness.csv'; d=pd.read_csv(p); assert len(d)==25; assert d['trigger_count'].gt(0).all(); assert d['trigger_rate'].between(0,1).all(); assert d['trigger_mean_return'].notna().all(); assert d['avoided_return'].notna().all(); print(d[['因子','horizon','trigger_mean_return','avoided_return']].to_string(index=False))"
```

Expected: 25 行（5 因子 × 5 期限），关键数值有效；打印实际正负，不预设研究结论。

- [ ] **Step 3: 运行相关回归测试**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_sentiment_timing_analysis.py tests/test_sentiment_timing_notebook.py tests/test_simple_sentiment_timing_notebook.py -v
```

Expected: 全部通过，原周度 Notebook 的结构测试仍通过。

- [ ] **Step 4: 核对差异范围**

Run:

```powershell
git diff --check
git status --short
```

Expected: 无空白错误；本任务没有改写原 `reproduce_sentiment_timing.ipynb` 的既有用户改动。

- [ ] **Step 5: 提交执行结果**

```powershell
git add -- '因子回测/涨跌停情绪因子/simple_sentiment_timing.ipynb' '因子回测/涨跌停情绪因子/simple_output'
git commit -m "test: 执行情绪择时 notebook 并保存结果"
```

- [ ] **Step 6: 最终核验**

Run:

```powershell
git show --stat --oneline HEAD
git status --short
```

Expected: 最后提交只包含已执行 Notebook 和新输出；用户原有日志与原 Notebook 改动仍保持原状。
