# 质量动量因子 Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `因子回测/质量动量因子.ipynb` 中实现并回测示例代码版和 63 日半衰期版质量动量因子。

**Architecture:** 因子函数保留在研究 Notebook 内，NumPy 负责单股票 26 日滑动窗口的加权回归，Polars 负责全市场分组、收益构造与长表对齐。两个综合得分共享同一输入样本，分别交给现有 `因子回测.alpha.analyze_factor` 完成 IC 与分组回测。

**Tech Stack:** Python 3.9、NumPy、Polars、Jupyter/nbformat、pytest、项目现有 `analyze_factor`

## Global Constraints

- Python、pytest 和 Notebook 验证统一使用 `E:\working\anaconda3\envs\quant\python.exe`。
- 只修改 `因子回测/质量动量因子.ipynb`，并新增 `tests/test_quality_momentum_factor_notebook.py`；不改公共模块。
- 主因子严格保留示例代码的线性权重、`np.polyfit` 权重语义和 R² 公式。
- 半衰期因子使用 63 日指数观测权重、标准 WLS 和加权均值 R²。
- 因子使用截至当日的 26 个收盘价，回测收益从下一交易日开始；不做中性化或缩尾。
- 使用 `apply_patch` 修改文件；Notebook 修改后必须通过 nbformat 结构校验。
- 保留工作区中所有无关改动，暂存和提交时只指定本计划涉及的文件。

---

### Task 1: 用失败测试锁定两个因子版本

**Files:**
- Create: `tests/test_quality_momentum_factor_notebook.py`
- Test: `因子回测/质量动量因子.ipynb`

**Interfaces:**
- Expects: `quality_momentum_weights(window, mode, half_life=63.0) -> np.ndarray`
- Expects: `calc_momentum_score(prices, mode="example", annualization=250, half_life=63.0) -> tuple[float, float, float]`
- Expects: `add_quality_momentum_factors(data, window=26, annualization=250, half_life=63.0) -> pl.DataFrame`

- [ ] **Step 1: 创建 Notebook 行为测试**

使用 `apply_patch` 创建测试文件。测试加载器遍历代码单元，只执行函数定义，避免访问真实行情：

```python
from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import nbformat
import numpy as np
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "因子回测" / "质量动量因子.ipynb"


def load_notebook_functions() -> dict:
    notebook = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    function_nodes = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = "\n".join(
            line for line in cell.source.splitlines()
            if not line.lstrip().startswith("%")
        )
        tree = ast.parse(source)
        function_nodes.extend(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    module = ast.Module(body=function_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "pl": pl}
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def test_example_score_matches_literal_reference_values():
    calc_score = load_notebook_functions()["calc_momentum_score"]
    x = np.arange(26, dtype=float)
    prices = 100.0 * np.exp(0.002 * x + 0.00008 * x**2)
    actual = calc_score(prices, mode="example")
    expected = (1.8578009654578476, 0.9831522318751459, 1.8265011655696837)
    assert np.allclose(actual, expected)


def test_exponential_trend_has_known_return_and_perfect_fit():
    calc_score = load_notebook_functions()["calc_momentum_score"]
    prices = 100.0 * np.exp(0.001 * np.arange(26, dtype=float))
    expected_annual_ret = 0.2840254166877415
    for mode in ("example", "halflife"):
        annual_ret, r2, score = calc_score(prices, mode=mode)
        assert np.isclose(annual_ret, expected_annual_ret)
        assert np.isclose(r2, 1.0)
        assert np.isclose(score, expected_annual_ret)


def test_constant_prices_return_zero_metrics():
    calc_score = load_notebook_functions()["calc_momentum_score"]
    for mode in ("example", "halflife"):
        assert np.allclose(calc_score(np.full(26, 10.0), mode=mode), (0.0, 0.0, 0.0))


def test_halflife_weights_and_invalid_parameters():
    weights = load_notebook_functions()["quality_momentum_weights"]
    halflife = weights(26, mode="halflife", half_life=63.0)
    assert np.isclose(halflife[0], 0.5 ** (25 / 63))
    assert np.isclose(halflife[-1], 1.0)
    assert np.all(np.diff(halflife) > 0)
    with pytest.raises(ValueError, match="window"):
        weights(1, mode="example")
    with pytest.raises(ValueError, match="half_life"):
        weights(26, mode="halflife", half_life=0)
    with pytest.raises(ValueError, match="mode"):
        weights(26, mode="unknown")


def test_rolling_factor_waits_for_26_prices_and_stays_within_code():
    add_factors = load_notebook_functions()["add_quality_momentum_factors"]
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(30)]
    rows = []
    for code in ("B", "A"):
        for index, trading_date in enumerate(dates):
            close = 100.0 * np.exp(0.001 * index)
            if code == "B" and index == 4:
                close = 0.0
            rows.append({"code": code, "trading_date": trading_date, "close": close})
    source = pl.DataFrame(rows).reverse()
    result = add_factors(source, window=26)
    a_rows = result.filter(pl.col("code") == "A").sort("trading_date")
    b_rows = result.filter(pl.col("code") == "B").sort("trading_date")
    assert a_rows["momentum_score"].head(25).null_count() == 25
    assert a_rows["momentum_score"].tail(5).null_count() == 0
    assert np.allclose(a_rows["momentum_r2"].tail(5), np.ones(5))
    assert b_rows["momentum_score"].null_count() == 30
    assert result.select("code", "trading_date").rows() == (
        source.sort("code", "trading_date").select("code", "trading_date").rows()
    )


def test_rolling_factor_rejects_missing_input_columns():
    add_factors = load_notebook_functions()["add_quality_momentum_factors"]
    frame = pl.DataFrame({"code": ["A"], "trading_date": [date(2024, 1, 1)]})
    with pytest.raises(ValueError, match="close"):
        add_factors(frame)


def test_all_python_cells_compile_after_removing_ipython_magics():
    notebook = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = "\n".join(
            line for line in cell.source.splitlines()
            if not line.lstrip().startswith("%")
        )
        compile(source, f"{NOTEBOOK_PATH}:cell_{cell_index}", "exec")


def test_backtest_cell_analyzes_both_score_columns_with_same_parameters():
    notebook = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    backtest_cell = [cell for cell in notebook.cells if cell.cell_type == "code"][-1]
    calls = []

    def fake_analyze_factor(**kwargs):
        calls.append(kwargs)
        return {
            "ic_stats": pl.DataFrame({"window": [1], "ic_mean": [0.1]}),
            "group_stats": pl.DataFrame({"window": [1], "group": ["G1"]}),
        }

    namespace = {
        "analysis_data": pl.DataFrame(
            {
                "trading_date": [date(2024, 1, 1)],
                "code": ["A"],
                "momentum_score": [1.0],
                "momentum_score_halflife": [1.0],
                "daily_ret": [0.01],
                "benchmark_ret": [0.01],
            }
        ),
        "analyze_factor": fake_analyze_factor,
    }
    exec(backtest_cell.source, namespace)
    assert [call["factor_col"] for call in calls] == [
        "momentum_score",
        "momentum_score_halflife",
    ]
    assert all(call["ret_windows"] == [1, 3, 5] for call in calls)
    assert all(call["ic_windows"] == [1, 3, 5] for call in calls)
    assert all(call["group_num"] == 5 for call in calls)
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_quality_momentum_factor_notebook.py -v
```

Expected: FAIL，当前 Notebook 只有 Markdown，三个预期函数和双因子回测代码均不存在。

- [ ] **Step 3: 提交失败测试作为本地 TDD 检查点**

不要单独提交红灯测试；保持测试文件未提交，直接进入 Task 2 的最小实现。

---

### Task 2: 实现因子并接入双回测

**Files:**
- Modify: `因子回测/质量动量因子.ipynb`
- Test: `tests/test_quality_momentum_factor_notebook.py`

**Interfaces:**
- Produces: 六列因子结果、`analysis_data`、`factor_results`
- Consumes: `read_day_data(...)`、`analyze_factor(...)`

- [ ] **Step 1: 用 apply_patch 将 Notebook 组织为一个 Markdown 和三个代码单元**

第一个代码单元按参考 Notebook 风格读取预热数据：

```python
# 获取数据
%reload_ext autoreload
%autoreload 2
import sys

sys.path.append(r"C:\Users\20561\Desktop\策略")
from my_utils.fun import read_day_data
from 因子回测.alpha import analyze_factor
import polars as pl
import numpy as np
import datetime as dt

start_date = dt.date(2021, 1, 1)
end_date = dt.date(2026, 1, 1)
window = 26
annualization = 250
half_life = 63.0
data_start_date = start_date - dt.timedelta(days=60)

stock_data = read_day_data(
    start_date=data_start_date,
    end_date=end_date,
    fields=["trading_date", "code", "close", "pre_close"],
)
```

- [ ] **Step 2: 在第二个代码单元实现权重和单窗口公式**

```python
def quality_momentum_weights(window: int, mode: str, half_life: float = 63.0) -> np.ndarray:
    """生成示例版线性权重或标准半衰期观测权重。"""
    if window < 2:
        raise ValueError("window 必须大于等于 2")
    if half_life <= 0:
        raise ValueError("half_life 必须大于 0")
    if mode == "example":
        return np.linspace(1.0, 2.0, window)
    if mode == "halflife":
        age = np.arange(window - 1, -1, -1, dtype=float)
        return 0.5 ** (age / half_life)
    raise ValueError("mode 必须是 'example' 或 'halflife'")


def calc_momentum_score(
    prices,
    mode: str = "example",
    annualization: int = 250,
    half_life: float = 63.0,
) -> tuple[float, float, float]:
    """计算一个窗口的年化趋势收益、R² 与质量动量得分。"""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices 必须是一维价格序列")
    weights = quality_momentum_weights(len(prices), mode, half_life)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        return np.nan, np.nan, np.nan
    y = np.log(prices)
    if np.ptp(y) <= 1e-12:
        return 0.0, 0.0, 0.0
    x = np.arange(len(y), dtype=float)
    fit_weights = weights if mode == "example" else np.sqrt(weights)
    slope, intercept = np.polyfit(x, y, deg=1, w=fit_weights)
    annual_ret = np.expm1(slope * annualization)
    y_fit = np.polyval([slope, intercept], x)
    center = y.mean() if mode == "example" else np.average(y, weights=weights)
    ss_res = np.sum(weights * (y - y_fit) ** 2)
    ss_tot = np.sum(weights * (y - center) ** 2)
    r2 = 0.0 if ss_tot <= 1e-10 else 1.0 - ss_res / ss_tot
    if mode == "halflife":
        r2 = float(np.clip(r2, 0.0, 1.0))
    return float(annual_ret), float(r2), float(annual_ret * r2)
```

- [ ] **Step 3: 在同一代码单元加入向量化滚动实现**

```python
def _calculate_momentum_windows(price_windows, mode, annualization, half_life):
    """批量计算一个股票的全部滚动窗口。"""
    window_size = price_windows.shape[1]
    metrics = np.full((len(price_windows), 3), np.nan, dtype=float)
    valid = np.isfinite(price_windows).all(axis=1) & (price_windows > 0).all(axis=1)
    if not valid.any():
        return metrics
    y = np.log(price_windows[valid])
    x = np.arange(window_size, dtype=float)
    weights = quality_momentum_weights(window_size, mode, half_life)
    regression_weights = weights**2 if mode == "example" else weights
    weight_sum = regression_weights.sum()
    x_mean = np.sum(regression_weights * x) / weight_sum
    x_centered = x - x_mean
    denominator = np.sum(regression_weights * x_centered**2)
    slopes = np.sum(y * (regression_weights * x_centered), axis=1) / denominator
    intercepts = np.sum(y * regression_weights, axis=1) / weight_sum - slopes * x_mean
    fitted = slopes[:, None] * x + intercepts[:, None]
    annual_ret = np.expm1(slopes * annualization)
    centers = y.mean(axis=1) if mode == "example" else np.sum(y * weights, axis=1) / weights.sum()
    ss_res = np.sum(weights * (y - fitted) ** 2, axis=1)
    ss_tot = np.sum(weights * (y - centers[:, None]) ** 2, axis=1)
    r2 = np.zeros(len(y), dtype=float)
    non_constant = ss_tot > 1e-10
    r2[non_constant] = 1.0 - ss_res[non_constant] / ss_tot[non_constant]
    constant = np.ptp(y, axis=1) <= 1e-12
    annual_ret[constant] = 0.0
    r2[constant] = 0.0
    if mode == "halflife":
        r2 = np.clip(r2, 0.0, 1.0)
    metrics[valid] = np.column_stack((annual_ret, r2, annual_ret * r2))
    return metrics


def add_quality_momentum_factors(data, window=26, annualization=250, half_life=63.0):
    """按股票追加示例版和半衰期版质量动量因子。"""
    required = {"code", "trading_date", "close"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"缺少必要列: {', '.join(missing)}")
    quality_momentum_weights(window, "example", half_life)
    mode_columns = {
        "example": ("momentum_annual_ret", "momentum_r2", "momentum_score"),
        "halflife": (
            "momentum_annual_ret_halflife",
            "momentum_r2_halflife",
            "momentum_score_halflife",
        ),
    }

    def add_group(group):
        group = group.sort("trading_date")
        output = {
            column: np.full(group.height, np.nan)
            for columns in mode_columns.values()
            for column in columns
        }
        if group.height >= window:
            windows = np.lib.stride_tricks.sliding_window_view(
                group["close"].to_numpy(), window_shape=window
            )
            for mode, columns in mode_columns.items():
                metrics = _calculate_momentum_windows(
                    windows, mode, annualization, half_life
                )
                for metric_index, column in enumerate(columns):
                    output[column][window - 1:] = metrics[:, metric_index]
        return group.with_columns(
            pl.Series(column, values) for column, values in output.items()
        )

    factor_columns = [column for columns in mode_columns.values() for column in columns]
    return (
        data.sort("code", "trading_date")
        .group_by("code", maintain_order=True)
        .map_groups(add_group)
        .with_columns(pl.col(column).fill_nan(None) for column in factor_columns)
        .sort("code", "trading_date")
    )
```

- [ ] **Step 4: 构造与参考框架一致的分析长表**

```python
stock_data = add_quality_momentum_factors(
    stock_data, window=window, annualization=annualization, half_life=half_life
)

# daily_ret[t] 表示 t-1 收盘到 t 收盘；analyze_factor 从 t+1 开始评价 t 日因子。
analysis_data = (
    stock_data.with_columns(
        (pl.col("close") / pl.col("pre_close") - 1).alias("daily_ret")
    )
    .filter(pl.col("trading_date").is_between(start_date, end_date))
    .select(
        "trading_date", "code",
        "momentum_annual_ret", "momentum_r2", "momentum_score",
        "momentum_annual_ret_halflife", "momentum_r2_halflife",
        "momentum_score_halflife", "daily_ret",
    )
    .drop_nulls(["momentum_score", "momentum_score_halflife", "daily_ret"])
    .filter(
        pl.col("momentum_score").is_finite()
        & pl.col("momentum_score_halflife").is_finite()
        & pl.col("daily_ret").is_finite()
    )
    .with_columns(
        pl.col("daily_ret").mean().over("trading_date").alias("benchmark_ret")
    )
)

analysis_data.head()
```

- [ ] **Step 5: 在第三个代码单元回测两个因子**

```python
factor_columns = {
    "示例代码版": "momentum_score",
    "63日半衰期版": "momentum_score_halflife",
}
factor_results = {}
for factor_name, factor_col in factor_columns.items():
    print(f"\n===== {factor_name}：{factor_col} =====")
    factor_results[factor_name] = analyze_factor(
        data=analysis_data,
        factor_col=factor_col,
        ret_col="daily_ret",
        ret_windows=[1, 3, 5],
        ic_windows=[1, 3, 5],
        group_num=5,
        plot=True,
        save_result=False,
    )

{
    factor_name: {
        "ic_stats": result["ic_stats"],
        "group_stats": result["group_stats"],
    }
    for factor_name, result in factor_results.items()
}
```

将 Notebook metadata 设置为 `kernelspec.name = "quant"`、`display_name = "Python (quant)"`，并清除旧输出与执行序号。

- [ ] **Step 6: 运行测试并确认 GREEN**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_quality_momentum_factor_notebook.py -v
```

Expected: 8 tests passed，0 failed。

- [ ] **Step 7: 提交实现与测试**

```powershell
git add -- '因子回测/质量动量因子.ipynb' 'tests/test_quality_momentum_factor_notebook.py'
git commit -m "feat: reproduce quality momentum factor variants"
```

---

### Task 3: 自顶向下执行与最终验证

**Files:**
- Modify: `因子回测/质量动量因子.ipynb`（只写入执行输出）
- Test: `tests/test_quality_momentum_factor_notebook.py`

**Interfaces:**
- Verifies: Notebook 结构、合成数据行为、真实行情执行、双因子回测结果

- [ ] **Step 1: 校验 Notebook 结构**

Run:

```powershell
@'
from pathlib import Path
import nbformat
path = Path("因子回测") / "质量动量因子.ipynb"
notebook = nbformat.read(str(path), as_version=4)
nbformat.validate(notebook)
print(len(notebook.cells), sum(cell.cell_type == "code" for cell in notebook.cells))
'@ | & 'E:\working\anaconda3\envs\quant\python.exe' -
```

Expected: exit code 0，输出 `4 3`。

- [ ] **Step 2: 运行定向测试**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_quality_momentum_factor_notebook.py -v
```

Expected: 8 tests passed。

- [ ] **Step 3: 使用 quant 内核完整执行 Notebook**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=quant '因子回测\质量动量因子.ipynb'
```

Expected: exit code 0，两个 `analyze_factor` 调用均完成。

- [ ] **Step 4: 检查执行输出没有异常**

Run:

```powershell
@'
from pathlib import Path
import nbformat
path = Path("因子回测") / "质量动量因子.ipynb"
notebook = nbformat.read(str(path), as_version=4)
errors = [
    output
    for cell in notebook.cells if cell.cell_type == "code"
    for output in cell.get("outputs", []) if output.output_type == "error"
]
print(f"errors={len(errors)}")
print([cell.execution_count for cell in notebook.cells if cell.cell_type == "code"])
assert not errors
'@ | & 'E:\working\anaconda3\envs\quant\python.exe' -
```

Expected: `errors=0`，三个代码单元均有执行序号。

- [ ] **Step 5: 运行相关回归测试与差异检查**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_analyze_factor.py tests/test_quality_momentum_factor_notebook.py -v
git diff --check -- '因子回测/质量动量因子.ipynb' 'tests/test_quality_momentum_factor_notebook.py'
git status --short -- '因子回测/质量动量因子.ipynb' 'tests/test_quality_momentum_factor_notebook.py'
```

Expected: 所有测试通过；差异检查无错误；状态只包含计划内两个文件。

- [ ] **Step 6: 提交执行后的 Notebook 输出**

```powershell
git add -- '因子回测/质量动量因子.ipynb' 'tests/test_quality_momentum_factor_notebook.py'
git commit -m "test: execute quality momentum factor notebook"
```
