# 多基准单因子择时设计

## 1. 目标

将 `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb` 中的择时回测核心函数抽成可复用模块，支持对多个宽基指数基准（全 A 市值加权、中证 500、中证 1000、中证 2000）运行同一套情绪因子信号的择时回测，横向比较情绪因子对不同市值风格的有效性。

## 2. 设计决策

| 决策项 | 选择 | 说明 |
|---|---|---|
| 模块位置 | `因子回测/涨跌停情绪因子/` 下两个 .py 文件 | 与现有该因子研究放在一起，不泛化到 `my_utils/` |
| 模块拆分 | `timing_engine.py` + `benchmark_loader.py` | 择时逻辑与基准数据加载解耦 |
| 函数命名 | 保持与原 notebook 一致 + 新增 `run_multi_benchmark_timing` 等组合函数 | 复用现有已测试代码，避免重命名引入 bug |
| 基准来源 | 指数优先走本地 Parquet，回退到掘金 API / Tushare 实时拉取 | 中证 500 已有本地缓存；1000/2000 需在线获取 |
| 信号逻辑 | 信号仍由全 A 情绪因子计算，不切换成分股 | 方案 A：同一套因子择时不同基准 |
| Notebook 风格 | 极简编排，3~5 个细胞完成全流程 | 导入 → 读数据 → 算因子 → 跑多基准回测 → 展示结果 |

## 3. 模块设计

### 3.1 `benchmark_loader.py` — 基准收益加载

```python
def load_benchmark(
    name: str,
    start_date: date,
    end_date: date,
    prepared_daily: pl.DataFrame | None = None,
    source: str = "auto",
) -> pd.DataFrame:
    """
    加载指定宽基指数的日频收益序列。

    Parameters
    ----------
    name : str
        基准名称。可选：
        - "all_a_value_weight"：全A市值加权（需传入 prepared_daily）
        - "zz500"：中证500 (000905.SH)
        - "zz1000"：中证1000 (000852.SH)
        - "zz2000"：中证2000 (932000.CSI)
    start_date, end_date : date
        起止日期。
    prepared_daily : pl.DataFrame, optional
        prepare_stock_daily 的输出，仅 "all_a_value_weight" 需要。
    source : str, default "auto"
        数据来源。auto = 按 local → gm → ts 优先级自动选择。

    Returns
    -------
    pd.DataFrame
        含 trading_date 和 market_daily_ret 两列的日频基准表。
        与 build_value_weighted_benchmark 的输出格式一致。
    """
```

内部代码映射：

```python
_BENCHMARK_CODE_MAP = {
    "zz500":  ("SHSE.000905", "000905.SH"),
    "zz1000": ("SHSE.000852", "000852.SH"),
    "zz2000": ("CSI.932000",  "932000.CSI"),
}
```

数据来源优先级：
1. `local`：检查 `E:\working\stock_data\barra_cne5\benchmark_zz500.parquet`（仅中证 500）
2. `gm`：`stock_api().gm_get_index_day_data(index_code, start_date, end_date)`
3. `ts`：`stock_api().ts.index_daily(ts_code, start_date, end_date)`
4. `rq`：`RqData().get_price(index_code, ...)`

### 3.2 `timing_engine.py` — 择时引擎

从 `sentiment_factors_5d_research.ipynb` 抽取以下已有函数（签名不变）：

- `compute_threshold(data, factor_columns, quantile, lower_quantile, upper_quantile, min_history)`
- `run_timing(data, signal_column, horizon, anchor_date, require_complete_exit)`
- `summarize_timing(daily, blocks, factor, horizon)`
- `annualized_metrics(daily_returns)`
- `plot_timing_nav_comparison(daily_results, factor_columns, horizons, start_date)`

**新增函数：**

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
    benchmark_data_source: str = "auto",
) -> dict:
    """
    对多个基准运行同一套情绪因子的择时回测。

    Parameters
    ----------
    factor_daily : pl.DataFrame
        build_daily_sentiment_factors 输出的日频因子表。
    benchmarks : list[str]
        基准名称列表，如 ["all_a_value_weight", "zz500", "zz1000", "zz2000"]。
    prepared_daily : pl.DataFrame
        prepare_stock_daily 的输出，用于全A基准计算。
    calendar : pl.DataFrame
        prepare_stock_daily 返回的日历表。
    start_date, end_date : date
        研究起止日期。
    horizons : tuple[int, ...]
        持有周期列表。
    lower_quantile, upper_quantile : float
        区间择时阈值分位。
    min_history : int
        扩展窗口最少预热天数。
    benchmark_data_source : str
        指数数据来源，传给 load_benchmark。

    Returns
    -------
    dict
        summary: pd.DataFrame — 汇总表，每行一个 (基准, 因子, 持有期)
        daily: dict — {(基准, 因子, 持有期): run_timing 返回的逐日明细}
    """
```


```python
def plot_multi_benchmark_summary(
    multi_results: dict,
    factor_columns: list[str],
    factor_labels: dict[str, str],
    horizons: tuple[int, ...],
    benchmarks: list[str],
    benchmark_labels: dict[str, str] | None = None,
) -> None:
    """
    绘制多基准择时对比汇总图。

    每张图对应一个 (因子, 持有期) 组合，画出一个多折线图：
    - x 轴：时间
    - y 轴：累计净值
    - 每条线 = 不同基准的策略择时净值

    并输出一个热力图（row=基准，column=因子，分面=持有期），
    颜色映射年化超额收益或夏普比率。
    """
```

## 4. Notebook 改造后样式

改造后 `sentiment_factors_5d_research.ipynb` 只有 **5 个细胞**（不含纯 markdown）：

| # | 类型 | 内容 |
|---|---|---|
| 1 | code | 导入依赖、定义常量（因子配置、基准列表、参数） |
| 2 | code | 读日线、清洗、算因子（同一段代码） |
| 3 | code | 多基准择时：`result = run_multi_benchmark_timing(...)` |
| 4 | code | 展示汇总表 |
| 5 | code | 画对比图 |

其中细胞 1~2 保持现有代码，细胞 3~5 是新加/精简后的代码。

## 5. 新增 Notebook：多基准择时对比分析（可选）

在 `因子回测/涨跌停情绪因子/` 下新建 `multi_benchmark_timing_comparison.ipynb`，直接展示跨基准对比：

```python
from timing_engine import run_multi_benchmark_timing, plot_multi_benchmark_summary
from benchmark_loader import load_benchmark

# 配置
BENCHMARKS = ["all_a_value_weight", "zz500", "zz1000", "zz2000"]
HORIZONS = (1, 3, 5, 10)

# 读数据 + 算因子（与 sentiment_factors_5d_research 共享同一段代码）
...

# 多基准择时
results = run_multi_benchmark_timing(
    factor_daily=factor_daily,
    benchmarks=BENCHMARKS,
    prepared_daily=prepared_daily,
    start_date=START_DATE,
    end_date=END_DATE,
    horizons=HORIZONS,
)

# 汇总表
display(results["summary"].style.format(...))

# 对比图
plot_multi_benchmark_summary(results, ...)
```

## 6. 实现要点

### 6.1 全 A 市值加权基准的复用

`build_value_weighted_benchmark` 原本只在 notebook 内定义，抽取到 `timing_engine.py` 中（保持签名不变），同时在 `benchmark_loader.py` 中通过以下方式引用：

```python
# benchmark_loader.py
def _load_all_a_value_weight(prepared_daily, calendar):
    from 因子回测.涨跌停情绪因子.timing_engine import build_value_weighted_benchmark
    return build_value_weighted_benchmark(prepared_daily, calendar)
```

### 6.2 指数收益对齐规则

指数日行情的 `pct` 列可能名称不同：

| 来源 | 收益列 | 日期列 |
|---|---|---|
| 全A自建 | `market_daily_ret`（小数） | `trading_date` |
| 掘金 API | `pct`（百分比） | `trading_date` |
| Tushare | `pct_chg`（百分比） | `trade_date` |

`load_benchmark` 内部统一转成 `market_daily_ret`（小数），日期统一为 `trading_date`，保证下游和 `build_value_weighted_benchmark` 的输出格式完全一致。

### 6.3 日期对齐

因子和基准按 `trading_date` inner join，确保信号和基准收益在同一组日期上比较，避免因指数停市日不同导致样本错位。

### 6.4 返回格式

`run_multi_benchmark_timing` 返回的 `summary` 包含列：

| 列名 | 说明 |
|---|---|
| `benchmark` | 基准名称 (str) |
| `factor` | 因子名称 (str) |
| `horizon` | 持有周期 (int) |
| `holding_ratio` | 持仓比例 |
| `annual_return` | 策略年化收益 |
| `benchmark_annual_return` | 基准年化收益 |
| `annual_excess_return` | 年化超额收益 |
| `sharpe` | 策略夏普比率 |
| `max_drawdown` | 策略最大回撤 |
| `timing_hit_rate` | 择时命中率 |
| `relative_final_nav` | 相对期末净值（策略/基准 - 1） |

## 7. 验证方案

1. **全A基准一致性**：用抽取后的 `build_value_weighted_benchmark` 跑原 notebook 的数据，输出应与 notebook 中直接计算的 `market_daily_ret` 一致。
2. **指数基准加载**：对 `zz500` 分别用 local 和 gm source 加载并比较日收益序列（只比较有重叠的日期）。
3. **回测结果可复现**：用抽取后的 `run_multi_benchmark_timing` 对全 A 基准跑一次，结果与原 notebook 一致。
4. **错误处理**：中证 2000 本地无数据时，能否自动回退到 API；API 不可用时是否报清晰错误。

## 8. 不纳入范围

- 不修改 `my_utils/` 下的通用模块。
- 不影响 `reproduce_sentiment_timing.ipynb`（周频模式不在此次改动范围）。
- 不涉及成分股筛选、指数复制成本或滑点。
