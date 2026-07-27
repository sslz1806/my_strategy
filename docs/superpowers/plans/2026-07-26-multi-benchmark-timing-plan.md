# 多基准单因子择时测试 实现计划

> **For agentic workers:** 使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` 按任务实施。步骤使用 checkboxes (`- [ ]`) 跟踪。

**目标:** 将 `sentiment_factors_5d_research.ipynb` 中的择时回测函数抽取为可复用模块，支持对全 A 市值加权、中证 500/1000/2000 等基准进行单因子择时对比。

**架构:** 两个 .py 模块 + 精简 notebook。`benchmark_loader.py` 负责基准收益加载，`timing_engine.py` 负责择时逻辑。Notebook 中只保留数据读取、因子计算和编排。

**Tech Stack:** Python 3.9+, Polars, Pandas, NumPy, Matplotlib

## 全局约束

- 所有新增文件放在 `因子回测/涨跌停情绪因子/` 下
- 函数签名与原 notebook 保持兼容（不破坏已有测试）
- 指数收益统一为 `trading_date` 日期列 + `market_daily_ret` 小数收益列
- 中证 500 本地数据路径：`E:/working/stock_data/barra_cne5/benchmark_zz500.parquet`（含 `trading_date`, `close`, `ret_1d`，仅覆盖 2021-01-04 起）
- 中证 1000 (`000852`) 和中证 2000 (`932000.CSI`) 默认走掘金 API `stock_api.gm_get_index_day_data`

---

## 文件结构

| 文件 | 类型 | 职责 |
|---|---|---|
| `因子回测/涨跌停情绪因子/benchmark_loader.py` | 新建 | `load_benchmark()` 统一加载各基准日收益 |
| `因子回测/涨跌停情绪因子/timing_engine.py` | 新建 | 择时核心函数（从 notebook 抽取） + 多基准新增函数 |
| `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb` | 修改 | 精简为 5-6 个细胞，导入模块后跑多基准 |
| `因子回测/涨跌停情绪因子/multi_benchmark_timing_comparison.ipynb` | 新建（可选） | 纯对比 notebook，直接读因子数据 + 跑多基准 |

---

### Task 1: 创建 `benchmark_loader.py` — 基准收益加载

**Files:**
- Create: `因子回测/涨跌停情绪因子/benchmark_loader.py`
- （无外部依赖，但需要运行时确认 stock_api 的 gm 接口可用）

**Interfaces:**
- Consumes: `stock_api.gm_get_index_day_data(index_code, start_date, end_date)` 返回 pandas DataFrame
- Consumes: `E:/working/stock_data/barra_cne5/benchmark_zz500.parquet` Polars DataFrame
- Produces: `load_benchmark(name, start_date, end_date, prepared_daily, calendar, source="auto")` → pandas DataFrame（`trading_date` + `market_daily_ret`）

**代码：** 该文件包含以下结构：

```python
"""
benchmark_loader.py — 统一加载各宽基指数的日频收益序列。

支持的基准名称:
- "all_a_value_weight": 全 A 市值加权（需传入 prepared_daily + calendar）
- "zz500":  中证 500  (000905.SH)
- "zz1000": 中证 1000 (000852.SH)
- "zz2000": 中证 2000 (932000.CSI)
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# 中证 500 本地缓存路径
_ZZ500_LOCAL_PATH = Path("E:/working/stock_data/barra_cne5/benchmark_zz500.parquet")

# 指数代码映射：基准名 → (gm_code, ts_code)
_BENCHMARK_CODE_MAP = {
    "zz500":  ("SHSE.000905", "000905.SH"),
    "zz1000": ("SHSE.000852", "000852.SH"),
    "zz2000": ("CSI.932000",  "932000.CSI"),
}


def load_benchmark(
    name: str,
    start_date: date,
    end_date: date,
    prepared_daily: pl.DataFrame | None = None,
    calendar: pl.DataFrame | None = None,
    source: str = "auto",
) -> pd.DataFrame:
    """
    加载指定宽基指数的日频收益序列。
    
    返回的 DataFrame 包含 trading_date 和 market_daily_ret 两列，
    与 timing_engine.build_value_weighted_benchmark 输出格式一致。
    收益为小数形式（0.01 = 1%）。
    """
    if name == "all_a_value_weight":
        return _load_all_a_value_weight(prepared_daily, calendar, start_date, end_date)
    elif name in _BENCHMARK_CODE_MAP:
        return _load_index_benchmark(name, start_date, end_date, source)
    else:
        raise ValueError(f"未知基准名称: {name}，可选: all_a_value_weight, zz500, zz1000, zz2000")


def _load_all_a_value_weight(prepared_daily, calendar, start_date, end_date):
    """从 prepared_daily 计算全 A 市值加权收益。"""
    # 直接延时导入避免循环依赖
    from 因子回测.涨跌停情绪因子.timing_engine import build_value_weighted_benchmark
    result = build_value_weighted_benchmark(prepared_daily, calendar)
    # 按日期范围过滤
    result = result[
        (result["trading_date"] >= pd.Timestamp(start_date))
        & (result["trading_date"] <= pd.Timestamp(end_date))
    ].reset_index(drop=True)
    return result


def _load_index_benchmark(name, start_date, end_date, source="auto"):
    """从本地或 API 加载指数收益。
    
    数据来源优先级:
    1. local:  仅 zz500 有本地 parquet 缓存
    2. gm:     掘金 API (stock_api.gm_get_index_day_data)
    3. ts:     Tushare API (stock_api.ts.index_daily)
    """
    # 尝试本地（仅 zz500）
    if source in ("auto", "local") and name == "zz500" and _ZZ500_LOCAL_PATH.exists():
        local = pl.read_parquet(_ZZ500_LOCAL_PATH)
        result = local.select(
            pl.col("trading_date"),
            pl.col("ret_1d").alias("market_daily_ret"),
        ).filter(
            pl.col("trading_date").is_between(
                pl.lit(start_date).cast(pl.Date),
                pl.lit(end_date).cast(pl.Date),
            )
        ).sort("trading_date").to_pandas()
        result["trading_date"] = pd.to_datetime(result["trading_date"])
        return result.reset_index(drop=True)

    # 回退到掘金 API
    if source in ("auto", "gm"):
        try:
            gm_code = _BENCHMARK_CODE_MAP[name][0]
            from my_utils.stock_api import stock_api
            api = stock_api()
            df = api.gm_get_index_day_data(gm_code, start_date, end_date)
            if df is not None and len(df) > 0:
                result = df[["trading_date", "pct"]].copy()
                result["trading_date"] = pd.to_datetime(result["trading_date"])
                result["market_daily_ret"] = result["pct"] / 100.0  # 百分比 → 小数
                return result.sort_values("trading_date")[["trading_date", "market_daily_ret"]].reset_index(drop=True)
        except Exception:
            pass

    # 最后回退到 Tushare
    if source in ("auto", "ts"):
        try:
            ts_code = _BENCHMARK_CODE_MAP[name][1]
            from my_utils.stock_api import stock_api
            api = stock_api()
            ts = api.ts
            df = ts.index_daily(ts_code=ts_code,
                                start_date=start_date.strftime("%Y%m%d"),
                                end_date=end_date.strftime("%Y%m%d"))
            if df is not None and len(df) > 0:
                result = df.rename(columns={"trade_date": "trading_date", "pct_chg": "market_daily_ret"})
                result["trading_date"] = pd.to_datetime(result["trading_date"])
                result["market_daily_ret"] = result["market_daily_ret"] / 100.0
                return result.sort_values("trading_date")[["trading_date", "market_daily_ret"]].reset_index(drop=True)
        except Exception:
            pass

    raise RuntimeError(f"无法加载基准 {name}：所有数据源均失败")


def list_available_benchmarks() -> list[str]:
    """返回当前可用的基准名称列表（用于 notebook 中遍历）。"""
    return ["all_a_value_weight", "zz500", "zz1000", "zz2000"]
```

- [ ] **Step 1: 创建 benchmark_loader.py**

```bash
touch 因子回测/涨跌停情绪因子/benchmark_loader.py
```

- [ ] **Step 2: 写入上述完整代码**

- [ ] **Step 3: 验证加载可用**

```bash
E:\working\anaconda3\envs\quant\python.exe -c "
from datetime import date
from 因子回测.涨跌停情绪因子.benchmark_loader import load_benchmark, list_available_benchmarks
print('可用基准:', list_available_benchmarks())
df = load_benchmark('zz500', date(2021,1,4), date(2021,12,31))
print(f'zz500: {len(df)} rows, cols={df.columns.tolist()}, head={df.head(3).to_string()}')
"
```

- [ ] **Step 4: 提交**

```bash
git add 因子回测/涨跌停情绪因子/benchmark_loader.py
git commit -m "feat: add benchmark_loader for multi-benchmark timing tests"
```

---

### Task 2: 创建 `timing_engine.py` — 抽取择时核心函数

**Files:**
- Create: `因子回测/涨跌停情绪因子/timing_engine.py`

**Interfaces:**
- Consumes: 原 notebook 中函数签名
- Produces: 以下所有函数供 notebook 和下游使用

**函数列表（从 notebook 原样抽取，仅以下两处调整）：**

| 函数 | 来源 | 改动 |
|---|---|---|
| `build_value_weighted_benchmark` | notebook cell `46497cd1` | 原样（移除了对 notebook 全局变量的依赖） |
| `compute_threshold` | notebook cell `fb544037` | 原样（默认参数保持不变，但改为使用模块级常量） |
| `annualized_metrics` | notebook cell `fb544037` | 原样 |
| `run_timing` | notebook cell `fb544037` | 原样 |
| `summarize_timing` | notebook cell `fb544037` | 新增 `factor_labels` 参数，不强制依赖全局 FACTOR_LABELS |
| `plot_timing_nav_comparison` | notebook cell `6629847e` | 新增 `factor_labels` 参数 + `benchmark_label` 参数（支持多基准时改图例文字） |

**关键改动说明：**

```python
# summarize_timing 增加 factor_labels 参数
def summarize_timing(daily, blocks, factor, horizon, factor_labels=None):
    if factor_labels is None:
        factor_labels = {}
    # 不再依赖全局 FACTOR_LABELS
    factor_label = factor_labels.get(factor, factor)
    # ... 与原函数相同 ...
```

```python
# plot_timing_nav_comparison 增加 factor_labels + benchmark_label
def plot_timing_nav_comparison(daily_results, factor_columns, horizons,
                                start_date=None, factor_labels=None,
                                benchmark_label="全A市值加权基准"):
    if factor_labels is None:
        factor_labels = {f: f for f in factor_columns}
    # ... 绘图时用 factor_labels 替代全局 FACTOR_LABELS ...
    # ... 用 benchmark_label 替代硬编码 "全 A 市值加权基准" ...
```

**模块级常量：** 定义与 notebook 相同的默认值

```python
# ============================================================
# 默认参数（与 sentiment_factors_5d_research.ipynb 保持一致）
# ============================================================
THRESHOLD_QUANTILE = 0.60
MIN_HISTORY = 252
HORIZONS_DEFAULT = (1, 3, 5, 10)
PRICE_TOLERANCE = 1e-6
```

- [ ] **Step 1: 创建 timing_engine.py**

```bash
touch 因子回测/涨跌停情绪因子/timing_engine.py
```

- [ ] **Step 2: 从 notebook 提取所有函数到 timing_engine.py**

核心逻辑：完全复制 notebook 中 `cell 46497cd1` 的 `build_value_weighted_benchmark`、`cell fb544037` 的 `compute_threshold`、`annualized_metrics`、`run_timing`、`summarize_timing`、`plot_timing_nav_comparison`，做上述两处参数化改动。

注意函数体内引用的全局变量：
- `summarize_timing` 中引用 `FACTOR_LABELS` → 改为 `factor_labels` 参数
- `plot_timing_nav_comparison` 中引用 `FACTOR_LABELS` 和 `FACTOR_COLUMNS`、`HORIZONS` → 改为参数
- `compute_threshold` 引用 `THRESHOLD_QUANTILE`、`MIN_HISTORY` → 改为模块级常量
- `annualized_metrics` 干净，无外部引用
- `run_timing` 干净，无外部引用
- `build_value_weighted_benchmark` 干净，无外部引用

- [ ] **Step 3: 验证抽取后函数可导入**

```bash
E:\working\anaconda3\envs\quant\python.exe -c "
from 因子回测.涨跌停情绪因子.timing_engine import (
    build_value_weighted_benchmark, compute_threshold,
    annualized_metrics, run_timing, summarize_timing,
    plot_timing_nav_comparison
)
print('所有函数导入成功')
"
```

- [ ] **Step 4: 提交**

```bash
git add 因子回测/涨跌停情绪因子/timing_engine.py
git commit -m "feat: extract timing engine functions from notebook"
```

---

### Task 3: 新增 `run_multi_benchmark_timing` + `plot_multi_benchmark_summary`

**Files:**
- Modify: `因子回测/涨跌停情绪因子/timing_engine.py`（追加到文件末尾）

**Interfaces:**
- Consumes: `load_benchmark`（从 benchmark_loader 导入）
- Consumes: 已抽取的 `compute_threshold`, `run_timing`, `summarize_timing`, `annualized_metrics`
- Produces: `run_multi_benchmark_timing(factor_daily, benchmarks, prepared_daily, calendar, ...)` → dict
- Produces: `plot_multi_benchmark_summary(multi_results, factor_columns, factor_labels, ...)` → 显示图

**`run_multi_benchmark_timing` 代码：**

```python
def run_multi_benchmark_timing(
    factor_daily: pl.DataFrame,
    benchmarks: list[str],
    prepared_daily: pl.DataFrame,
    calendar: pl.DataFrame,
    start_date: date,
    end_date: date,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    lower_quantile: float = 0.65,
    upper_quantile: float = 1.0,
    min_history: int = 252,
    factor_columns: list[str] = None,
    factor_labels: dict[str, str] = None,
    benchmark_data_source: str = "auto",
) -> dict:
    """
    对多个基准运行同一套情绪因子的择时回测。

    返回:
        summary: pd.DataFrame — 每行一个(基准, 因子, 持有期)的绩效汇总
        daily: dict — {(基准, 因子, 持有期): 逐日明细}
    """
    from 因子回测.涨跌停情绪因子.benchmark_loader import load_benchmark

    if factor_columns is None:
        cols = [c for c in factor_daily.columns if c not in ("trading_date",)]
        factor_columns = cols
    if factor_labels is None:
        factor_labels = {f: f for f in factor_columns}

    factor_pd = factor_daily.to_pandas()
    factor_pd["trading_date"] = pd.to_datetime(factor_pd["trading_date"])

    daily_results = {}

    for bench_name in benchmarks:
        bench_ret = load_benchmark(
            bench_name, start_date, end_date,
            prepared_daily=prepared_daily, calendar=calendar,
            source=benchmark_data_source,
        )
        data = factor_pd.merge(bench_ret, on="trading_date", how="inner")

        from 因子回测.alpha import add_future_return
        data = add_future_return(data, ret_col="market_daily_ret", horizons=horizons)

        data = compute_threshold(data, factor_columns,
                                  lower_quantile=lower_quantile,
                                  upper_quantile=upper_quantile,
                                  min_history=min_history)
        for factor in factor_columns:
            data[f"signal_{factor}"] = (
                (data[factor] >= data[f"lower_{factor}"])
                & (data[factor] <= data[f"upper_{factor}"])
            ).astype(float)

        threshold_cols = [f"lower_{f}" for f in factor_columns]
        common_valid = data[threshold_cols].notna().all(axis=1)
        common_anchor = pd.Timestamp(data.loc[common_valid.idxmax(), "trading_date"])

        for factor in factor_columns:
            for horizon in horizons:
                daily, _ = run_timing(data, signal_column=f"signal_{factor}",
                                      horizon=horizon, anchor_date=common_anchor)
                daily_results[(bench_name, factor, horizon)] = daily

    summary_list = []
    for (bench_name, factor, horizon), daily_detail in daily_results.items():
        if len(daily_detail) == 0:
            continue
        strat_m = annualized_metrics(daily_detail["strategy_daily_ret"])
        bench_m = annualized_metrics(daily_detail["market_daily_ret"])
        summary_list.append({
            "benchmark": bench_name,
            "factor": factor,
            "factor_label": factor_labels.get(factor, factor),
            "horizon": horizon,
            "holding_ratio": daily_detail["position"].mean(),
            "annual_return": strat_m["annual_return"],
            "benchmark_annual_return": bench_m["annual_return"],
            "annual_excess_return": strat_m["annual_return"] - bench_m["annual_return"],
            "sharpe": strat_m["sharpe"],
            "max_drawdown": strat_m["max_drawdown"],
            "final_nav": daily_detail["strategy_nav"].iloc[-1],
            "benchmark_final_nav": daily_detail["benchmark_nav"].iloc[-1],
            "relative_final_nav": (
                daily_detail["strategy_nav"].iloc[-1]
                / daily_detail["benchmark_nav"].iloc[-1] - 1
            ),
        })

    summary = pd.DataFrame(summary_list)
    return {"summary": summary, "daily": daily_results}
```

**`plot_multi_benchmark_summary` 伪代码：**

```python
def plot_multi_benchmark_summary(
    multi_results: dict,
    factor_columns: list[str],
    factor_labels: dict[str, str],
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    benchmarks: list[str] = None,
    benchmark_labels: dict[str, str] = None,
):
    """
    绘制多基准择时对比。
    
    1. 每个 (因子, 持有期) 画一张多折线净值对比图
    2. 一张热力图：row=基准, column=因子(按持有期分面)，颜色=年化超额收益
    """
    daily = multi_results["daily"]
    if benchmark_labels is None:
        benchmark_labels = {b: b for b in benchmarks} if benchmarks else {}
    
    colors = ["#C44E52", "#1f77b4", "#2ca02c", "#9467bd"]
    
    for factor in factor_columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        for ax, horizon in zip(axes.ravel(), horizons):
            for idx, bench in enumerate(benchmarks or []):
                key = (bench, factor, horizon)
                if key not in daily or len(daily[key]) == 0:
                    continue
                detail = daily[key]
                ax.plot(
                    detail["trading_date"],
                    detail["benchmark_nav"],
                    color=colors[idx % len(colors)],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.5,
                )
                ax.plot(
                    detail["trading_date"],
                    detail["strategy_nav"],
                    label=benchmark_labels.get(bench, bench),
                    color=colors[idx % len(colors)],
                    linewidth=1.6,
                )
            ax.set_title(f"{horizon} 日持有期")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(f"{factor_labels.get(factor, factor)}：多基准择时净值对比", fontsize=15)
        plt.show()
    
    # 热力图：基准 × 因子（按持有期分面）
    summary = multi_results["summary"]
    n_horizons = len(horizons)
    fig, axes = plt.subplots(1, n_horizons, figsize=(5 * n_horizons, 5), constrained_layout=True)
    if n_horizons == 1:
        axes = [axes]
    for ax, horizon in zip(axes, horizons):
        sub = summary[summary["horizon"] == horizon]
        pivot = sub.pivot(index="benchmark", columns="factor", values="annual_excess_return")
        if benchmarks:
            pivot = pivot.reindex(index=[b for b in benchmarks if b in pivot.index])
        if factor_columns:
            pivot = pivot.reindex(columns=factor_columns)
        vals = pivot.to_numpy(dtype=float)
        limit = max(0.01, float(np.nanmax(np.abs(vals)))) if vals.size else 0.05
        im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(pivot.columns)), 
                       [factor_labels.get(c, c) for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)),
                       [benchmark_labels.get(b, b) for b in pivot.index])
        ax.set_title(f"{horizon}日持有期 | 年化超额收益")
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.show()
```

- [ ] **Step 1: 将上述两个新函数追加到 timing_engine.py 末尾**

- [ ] **Step 2: 验证新函数可导入**

```bash
E:\working\anaconda3\envs\quant\python.exe -c "
from 因子回测.涨跌停情绪因子.timing_engine import run_multi_benchmark_timing, plot_multi_benchmark_summary
print('函数导入成功')
"
```

- [ ] **Step 3: 提交**

```bash
git add 因子回测/涨跌停情绪因子/timing_engine.py
git commit -m "feat: add run_multi_benchmark_timing and plot_multi_benchmark_summary"
```

---

### Task 4: 精简 `sentiment_factors_5d_research.ipynb`

**Files:**
- Modify: `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb`

**目标：** 从约 13 个代码细胞精简为约 6 个代码细胞，删除重复的择时函数定义，改为导入模块。

**保留：**
- 细胞 0（导入依赖 + 全局配置 + 因子标签定义）— 小幅修改，增加 `from timing_engine import ...`
- 细胞 1（`prepare_stock_daily` + `build_daily_sentiment_factors` 定义 + 数据读取 + 因子计算 + 全A基准计算）— 原样保留
- 细胞 2（IC 分析 + 展示 + 滚动 IC）— 原样保留（IC 分析函数未迁移）

**修改：**
- 删除细胞 3（`compute_threshold`, `annualized_metrics`, `run_timing`, `summarize_timing`, `plot_timing_nav_comparison` 函数定义）— 改为从 timing_engine 导入
- 删除细胞 4（执行择时回测）— 精简为调用 `run_multi_benchmark_timing`
- 删除细胞 5（净值对比图）— 精简为调用 `plot_multi_benchmark_summary` 或原 `plot_timing_nav_comparison`

**改造后结构：**

| 细胞 | 内容 |
|---|---|
| #0 导入 + 配置 | 导入依赖 + 新增 `from timing_engine import ...` + 从 `benchmark_loader import load_benchmark, list_available_benchmarks` + 因子标签/方向/全局参数 |
| #1 因子管线 | `prepare_stock_daily` + `build_daily_sentiment_factors` 定义 + 读数据 + 执行因子管线 + 显示尾部 |
| #2 IC 分析 | `analyze_ic` + `report_ic_summary` + `compute_rolling_ic` + ...（保持现状，IC 分析未迁移到模块） |
| #3 单基准择时（保留原有） | 导入 `compute_threshold` + `run_timing` + `summarize_timing` → 跑全 A 单基准择时并展示 |
| #4 多基准择时（新增） | 调用 `run_multi_benchmark_timing` + `plot_multi_benchmark_summary` → 展示多基准择时对比 |
| #5 结论 | 与目前相同（IC 均值 vs 择时绩效交叉对比） |

**具体代码修改：**

细胞 0 末尾增加：

```python
# 新增：导入择时引擎和基准加载器
from 因子回测.涨跌停情绪因子.timing_engine import (
    build_value_weighted_benchmark, compute_threshold,
    annualized_metrics, run_timing, summarize_timing,
    plot_timing_nav_comparison,
    run_multi_benchmark_timing, plot_multi_benchmark_summary,
)
from 因子回测.涨跌停情绪因子.benchmark_loader import load_benchmark, list_available_benchmarks
```

细胞 3（原来是函数定义）改为：

```python
# ============================================================
# 全 A 单基准择时（函数定义已迁移到 timing_engine.py）
# ============================================================

# 1. 计算各因子的区间阈值
timing_input = compute_threshold(research_data, ...)  # 代码与原来相同

# 2. 信号生成（与原来相同）
for factor in FACTOR_COLUMNS:
    timing_input[f"signal_{factor}"] = ...

# 3. 锚点
common_anchor = pd.Timestamp(...)

# 4. 回测
timing_daily = {}
timing_summary_rows = []
for factor in FACTOR_COLUMNS:
    for horizon in HORIZONS:
        daily, blocks = run_timing(timing_input, f"signal_{factor}", horizon, anchor_date=common_anchor)
        timing_daily[(factor, horizon)] = daily
        timing_summary_rows.append(summarize_timing(daily, blocks, factor, horizon, FACTOR_LABELS))

# 5. 展示
timing_summary = pd.DataFrame(timing_summary_rows)
...  # 与原来相同

# 6. 绘图
plot_timing_nav_comparison(timing_daily, FACTOR_COLUMNS, HORIZONS,
                           start_date=common_anchor, factor_labels=FACTOR_LABELS)
```

细胞 4（新增多基准对比）：

```python
# ============================================================
# 多基准择时对比
# ============================================================

BENCHMARKS = list_available_benchmarks()  # ["all_a_value_weight", "zz500", "zz1000", "zz2000"]
BENCHMARK_LABELS = {
    "all_a_value_weight": "全A市值加权",
    "zz500": "中证500",
    "zz1000": "中证1000",
    "zz2000": "中证2000",
}

multi_results = run_multi_benchmark_timing(
    factor_daily=factor_daily,
    benchmarks=BENCHMARKS,
    prepared_daily=prepared_daily,
    calendar=trading_calendar,
    start_date=START_DATE,
    end_date=END_DATE,
    horizons=HORIZONS,
    factor_columns=FACTOR_COLUMNS,
    factor_labels=FACTOR_LABELS,
)

# 展示汇总表
summary = multi_results["summary"]
display(
    summary.style.format(
        {
            "holding_ratio": "{:.2%}", "annual_return": "{:.2%}",
            "benchmark_annual_return": "{:.2%}", "annual_excess_return": "{:+.2%}",
            "max_drawdown": "{:.2%}", "sharpe": "{:.2f}",
            "final_nav": "{:.3f}", "benchmark_final_nav": "{:.3f}",
            "relative_final_nav": "{:+.2%}",
        }
    )
)

# 画多基准净值对比图
plot_multi_benchmark_summary(
    multi_results,
    factor_columns=FACTOR_COLUMNS,
    factor_labels=FACTOR_LABELS,
    horizons=HORIZONS,
    benchmarks=BENCHMARKS,
    benchmark_labels=BENCHMARK_LABELS,
)
```

- [ ] **Step 1: 按上述方案修改 notebook 的导入细胞（#0）和第一段择时细胞（原 #3）**

注意：使用 `nbformat` 或 JSON 修改 notebook，或者直接在 Jupyter 中编辑。建议先做 JSON 级备份：

```bash
cp 因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb 因子回测/涨跌停情绪因子/sentiment_factors_5d_research_backup.ipynb
```

- [ ] **Step 2: 新增多基准对比细胞（#4）**

- [ ] **Step 3: 运行 notebook 验证全 A 基准择时结果与重构前一致**

```bash
jupyter nbconvert --to script --execute 因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb --stdout > /dev/null
```

或直接批量执行关键细胞验证。

- [ ] **Step 4: 运行多基准对比，确认 zz500 基准数据可正常加载**

- [ ] **Step 5: 提交**

```bash
git add 因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb 因子回测/涨跌停情绪因子/timing_engine.py
git commit -m "refactor: simplify notebook with multi-benchmark timing module"
```

---

### Task 5（可选）: 新建 `multi_benchmark_timing_comparison.ipynb`

**Files:**
- Create: `因子回测/涨跌停情绪因子/multi_benchmark_timing_comparison.ipynb`

纯对比 notebook，3 个代码细胞：

```python
# 细胞 1: 导入 + 配置 + 读取因子数据
from datetime import date
from pathlib import Path
import sys

from my_utils.fun import read_day_data
from 因子回测.涨跌停情绪因子.timing_engine import (
    build_value_weighted_benchmark, run_multi_benchmark_timing,
    plot_multi_benchmark_summary,
)
from 因子回测.涨跌停情绪因子.benchmark_loader import list_available_benchmarks

START_DATE = date(2018, 1, 2)
END_DATE = date(2026, 7, 20)
HORIZONS = (1, 3, 5, 10)
FACTOR_LABELS = {
    "limit_up_ratio": "涨停占比", "limit_down_ratio": "跌停占比",
    "net_limit_ratio": "净涨停占比", "limit_up_down_ratio": "涨跌停比值",
    "limit_up_next_ret": "涨停次日收益", "limit_down_next_ret": "跌停次日收益",
}
FACTOR_COLUMNS = list(FACTOR_LABELS)

# 假设已有预处理因子数据文件，或直接读日线重新算
daily_raw = read_day_data(...)  # 复用与主 notebook 相同的数据读取
```

```python
# 细胞 2: 跑多基准择时
BENCHMARKS = list_available_benchmarks()
BENCHMARK_LABELS = {"all_a_value_weight": "全A", "zz500": "中证500",
                    "zz1000": "中证1000", "zz2000": "中证2000"}

results = run_multi_benchmark_timing(
    factor_daily=factor_daily,
    benchmarks=BENCHMARKS,
    prepared_daily=prepared_daily,
    calendar=trading_calendar,
    start_date=START_DATE,
    end_date=END_DATE,
    horizons=HORIZONS,
    factor_columns=FACTOR_COLUMNS,
    factor_labels=FACTOR_LABELS,
)
```

```python
# 细胞 3: 展示
display(results["summary"].style.format(...))
plot_multi_benchmark_summary(results, FACTOR_COLUMNS, FACTOR_LABELS,
                              horizons=HORIZONS, benchmarks=BENCHMARKS,
                              benchmark_labels=BENCHMARK_LABELS)
```

- [ ] **Step 1: 创建 notebook 文件**

- [ ] **Step 2: 写入 3 个细胞**

- [ ] **Step 3: 验证可正常运行**

- [ ] **Step 4: 提交**

```bash
git add 因子回测/涨跌停情绪因子/multi_benchmark_timing_comparison.ipynb
git commit -m "feat: add multi-benchmark timing comparison notebook"
```
