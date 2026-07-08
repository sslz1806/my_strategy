# RQ Factor Style Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `因子回测/learn/米筐官方因子收益率_风格趋势.ipynb` 改造成投资者可读的 A 股风格雷达，看清显式风格因子的市场表现、方向含义和切换迹象。

**Architecture:** 把可测试的因子方向、表现表、图表和自动解读逻辑抽到 `因子回测/learn/rq_style_dashboard.py`。Notebook 只负责取数、调用看板函数和展示，原有 v1/v2 对照、自建 Barra、行业因子内容保留为研究附录。

**Tech Stack:** Python 3.9+, pandas, numpy, matplotlib, seaborn, pytest, Jupyter notebook JSON；验证命令优先使用 `E:\working\anaconda3\envs\quant\python.exe`。

## Global Constraints

- 默认回复和注释使用中文，关键业务逻辑、数据口径、边界条件需要中文注释。
- 不重写米筐数据拉取接口，不重新实现 Barra 截面回归。
- 主分析使用米筐 v2 显式风格因子，排除 `comovement` 和行业因子。
- `comovement` 只作为市场整体温度计，不进入风格排名。
- 行业因子单独作为行业轮动视角，不与风格因子混在一起。
- 大类聚合只做辅助摘要，不作为风格判断主依据。
- 修改已有 dirty worktree 时，只暂存和提交本任务涉及文件，不改动无关文件。
- 所有验证命令优先直接调用 `E:\working\anaconda3\envs\quant\python.exe`。

---

## File Structure

- Create: `因子回测/learn/rq_style_dashboard.py`
  - 责任：封装因子中文名称、方向解释、多窗口收益表、市场结论和核心图表函数。
  - 边界：不调用米筐 API，不管理缓存，只消费已有 `pd.DataFrame` 因子收益。

- Create: `因子回测/learn/test_rq_style_dashboard.py`
  - 责任：用小型合成 DataFrame 验证因子筛选、方向解释、窗口收益、中文结论和图表 smoke test。
  - 边界：不联网、不依赖米筐账号。

- Modify: `因子回测/learn/米筐官方因子收益率_风格趋势.ipynb`
  - 责任：在 Part 1 取数后插入“投资者风格雷达”主视图，并把原有研究内容标为附录。
  - 边界：不删除原有研究图，尽量通过新增 markdown/code cells 实现。

---

### Task 1: Factor Metadata And Direction Table

**Files:**
- Create: `因子回测/learn/rq_style_dashboard.py`
- Create: `因子回测/learn/test_rq_style_dashboard.py`

**Interfaces:**
- Produces: `FACTOR_INFO: dict[str, dict[str, str]]`
- Produces: `select_explicit_style_factors(columns: Iterable[str]) -> list[str]`
- Produces: `build_factor_direction_table(style_factors: Sequence[str], latest_returns: pd.Series | None = None) -> pd.DataFrame`
- Later tasks consume these names without modification.

- [ ] **Step 1: Write failing tests for metadata and style factor selection**

Add `因子回测/learn/test_rq_style_dashboard.py` with:

```python
import pandas as pd

from rq_style_dashboard import (
    FACTOR_INFO,
    build_factor_direction_table,
    select_explicit_style_factors,
)


def test_select_explicit_style_factors_excludes_comovement_and_chinese_industries():
    columns = [
        "comovement",
        "momentum",
        "beta",
        "residual_volatility",
        "银行",
        "食品饮料",
    ]

    result = select_explicit_style_factors(columns)

    assert result == ["beta", "momentum", "residual_volatility"]


def test_factor_direction_table_contains_investor_interpretation():
    latest_returns = pd.Series(
        {"momentum": 0.03, "residual_volatility": -0.02},
        name="latest",
    )

    table = build_factor_direction_table(
        ["momentum", "residual_volatility"],
        latest_returns=latest_returns,
    )

    assert list(table["因子"]) == ["momentum", "residual_volatility"]
    assert table.loc[0, "正收益代表"] == FACTOR_INFO["momentum"]["positive"]
    assert table.loc[1, "当前方向"] == FACTOR_INFO["residual_volatility"]["negative"]
    assert "低波动" in table.loc[1, "当前方向"]
```

- [ ] **Step 2: Run tests to verify they fail because the module does not exist**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest '因子回测/learn/test_rq_style_dashboard.py' -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rq_style_dashboard'`.

- [ ] **Step 3: Implement metadata and direction table**

Create `因子回测/learn/rq_style_dashboard.py` with:

```python
"""米筐风险因子收益率的投资者风格看板工具。

本模块只消费已经取好的因子收益 DataFrame，不连接米筐，也不重写缓存逻辑。
核心口径：因子收益为正表示高暴露方向跑赢低暴露方向；因子收益为负则相反。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


FACTOR_INFO: dict[str, dict[str, str]] = {
    "momentum": {
        "name": "动量",
        "group": "动量/反转",
        "positive": "过去强势股继续跑赢",
        "negative": "弱势股或反转方向占优",
    },
    "longterm_reversal": {
        "name": "长期反转",
        "group": "动量/反转",
        "positive": "长期反转暴露占优",
        "negative": "长期趋势延续方向占优",
    },
    "size": {
        "name": "规模",
        "group": "规模",
        "positive": "大市值暴露占优",
        "negative": "小市值暴露占优",
    },
    "mid_cap": {
        "name": "中盘",
        "group": "规模",
        "positive": "中市值暴露占优",
        "negative": "非中市值暴露占优",
    },
    "beta": {
        "name": "Beta",
        "group": "风险偏好",
        "positive": "高 beta、高弹性股票跑赢",
        "negative": "低 beta、防御股票跑赢",
    },
    "residual_volatility": {
        "name": "残差波动",
        "group": "风险偏好",
        "positive": "高特质波动股票跑赢",
        "negative": "低波动股票跑赢",
    },
    "liquidity": {
        "name": "流动性",
        "group": "交易活跃度",
        "positive": "高换手、高交易活跃度股票跑赢",
        "negative": "低换手、低交易活跃度股票跑赢",
    },
    "book_to_price": {
        "name": "账面市值比",
        "group": "价值/股息/成长",
        "positive": "高账面市值比、低估值股票跑赢",
        "negative": "低账面市值比、高估值股票跑赢",
    },
    "earnings_yield": {
        "name": "盈利收益",
        "group": "价值/股息/成长",
        "positive": "高盈利收益股票跑赢",
        "negative": "低盈利收益股票跑赢",
    },
    "dividend_yield": {
        "name": "股息率",
        "group": "价值/股息/成长",
        "positive": "高股息股票跑赢",
        "negative": "低股息股票跑赢",
    },
    "growth": {
        "name": "成长",
        "group": "价值/股息/成长",
        "positive": "成长暴露占优",
        "negative": "非成长或稳态暴露占优",
    },
    "profitability": {
        "name": "盈利能力",
        "group": "质量",
        "positive": "高盈利能力股票跑赢",
        "negative": "低盈利能力或投机方向跑赢",
    },
    "earnings_quality": {
        "name": "盈利质量",
        "group": "质量",
        "positive": "高盈利质量股票跑赢",
        "negative": "低盈利质量股票跑赢",
    },
    "investment_quality": {
        "name": "投资质量",
        "group": "质量",
        "positive": "高投资质量股票跑赢",
        "negative": "低投资质量股票跑赢",
    },
    "earnings_variability": {
        "name": "盈利波动",
        "group": "质量",
        "positive": "高盈利波动暴露占优",
        "negative": "盈利更稳定方向占优",
    },
    "leverage": {
        "name": "杠杆",
        "group": "杠杆",
        "positive": "高杠杆股票跑赢",
        "negative": "低杠杆股票跑赢",
    },
}


def select_explicit_style_factors(columns: Iterable[str]) -> list[str]:
    """从米筐因子收益列中筛出投资者主看板使用的显式风格因子。

    只保留 FACTOR_INFO 中有方向解释的英文风格因子；自然排除中文行业因子、
    `comovement` 以及暂未写入口径字典的其他列，避免混入不可解释信号。
    """
    return sorted([col for col in columns if col in FACTOR_INFO])


def _direction_text(factor: str, value: float | int | None) -> str:
    info = FACTOR_INFO[factor]
    if value is None or pd.isna(value):
        return "样本不足"
    if value > 0:
        return info["positive"]
    if value < 0:
        return info["negative"]
    return "高低暴露方向接近持平"


def build_factor_direction_table(
    style_factors: Sequence[str],
    latest_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """构建因子方向说明表，供 notebook 在所有图之前展示。

    latest_returns 可传入任意窗口收益，例如 60 日收益或年初至今收益；
    它只决定“当前方向”文案，不影响正负收益的固定解释。
    """
    rows: list[dict[str, object]] = []
    for factor in style_factors:
        if factor not in FACTOR_INFO:
            continue
        info = FACTOR_INFO[factor]
        latest_value = None if latest_returns is None else latest_returns.get(factor, np.nan)
        rows.append(
            {
                "因子": factor,
                "中文名": info["name"],
                "风格组": info["group"],
                "正收益代表": info["positive"],
                "负收益代表": info["negative"],
                "当前方向": _direction_text(factor, latest_value),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify metadata passes**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest '因子回测/learn/test_rq_style_dashboard.py' -q
```

Expected: PASS for 2 tests.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add -- '因子回测/learn/rq_style_dashboard.py' '因子回测/learn/test_rq_style_dashboard.py'
git commit -m "feat: add rq style factor metadata"
```

Expected: commit contains only the two files above.

---

### Task 2: Performance Table And Market Commentary

**Files:**
- Modify: `因子回测/learn/rq_style_dashboard.py`
- Modify: `因子回测/learn/test_rq_style_dashboard.py`

**Interfaces:**
- Consumes: `FACTOR_INFO`, `select_explicit_style_factors`
- Produces: `calc_cumulative_return(series: pd.Series, window: int | None = None) -> float`
- Produces: `build_style_performance_table(fr: pd.DataFrame, style_factors: Sequence[str], windows: Sequence[int] = (20, 60, 120), rank_window: int = 60) -> pd.DataFrame`
- Produces: `generate_market_style_commentary(perf_table: pd.DataFrame) -> str`

- [ ] **Step 1: Write failing tests for window returns and commentary**

Append to `因子回测/learn/test_rq_style_dashboard.py`:

```python
from rq_style_dashboard import (
    build_style_performance_table,
    calc_cumulative_return,
    generate_market_style_commentary,
)


def test_calc_cumulative_return_uses_tail_window():
    series = pd.Series([0.10, -0.05, 0.02])

    result = calc_cumulative_return(series, window=2)

    assert round(result, 6) == round((1 - 0.05) * (1 + 0.02) - 1, 6)


def test_build_style_performance_table_sorts_by_rank_window_and_explains_direction():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    fr = pd.DataFrame(
        {
            "momentum": [0.01, 0.01, 0.01, 0.01, 0.01],
            "residual_volatility": [-0.01, -0.01, -0.01, -0.01, -0.01],
            "beta": [0.00, 0.00, 0.02, 0.02, 0.02],
        },
        index=dates,
    )

    table = build_style_performance_table(
        fr,
        ["momentum", "residual_volatility", "beta"],
        windows=(2, 3),
        rank_window=3,
    )

    assert list(table.columns) == [
        "中文名",
        "风格组",
        "2日收益%",
        "3日收益%",
        "年初至今收益%",
        "最新5日收益%",
        "当前方向",
        "投资者解读",
        "信号强度",
    ]
    assert table.index[0] == "beta"
    assert "高 beta" in table.loc["beta", "当前方向"]
    assert "低波动" in table.loc["residual_volatility", "当前方向"]


def test_generate_market_style_commentary_mentions_strong_and_weak_factors():
    dates = pd.date_range("2026-01-01", periods=65, freq="D")
    fr = pd.DataFrame(
        {
            "momentum": [0.002] * 65,
            "beta": [0.001] * 65,
            "residual_volatility": [-0.0015] * 65,
            "liquidity": [-0.001] * 65,
        },
        index=dates,
    )
    table = build_style_performance_table(
        fr,
        ["momentum", "beta", "residual_volatility", "liquidity"],
        windows=(20, 60),
        rank_window=60,
    )

    text = generate_market_style_commentary(table)

    assert "当前市场风格" in text
    assert "动量" in text
    assert "低波动" in text
    assert "高换手" in text or "低换手" in text
```

- [ ] **Step 2: Run tests to verify they fail because functions are missing**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest '因子回测/learn/test_rq_style_dashboard.py' -q
```

Expected: FAIL with missing import or missing attribute for the new functions.

- [ ] **Step 3: Implement performance table and commentary**

Append to `因子回测/learn/rq_style_dashboard.py`:

```python
def calc_cumulative_return(series: pd.Series, window: int | None = None) -> float:
    """计算因子收益的复利累计收益。

    window 为 None 时使用全样本；传入整数时使用尾部 window 行。
    样本不足时返回 NaN，避免用 30 天数据伪装 120 天信号。
    """
    clean = series.dropna()
    if window is not None:
        if len(clean) < window:
            return float("nan")
        clean = clean.tail(window)
    if clean.empty:
        return float("nan")
    return float((1.0 + clean).prod() - 1.0)


def _strength_label(value: float) -> str:
    if pd.isna(value):
        return "样本不足"
    abs_value = abs(value)
    if abs_value >= 0.05:
        return "强"
    if abs_value >= 0.02:
        return "中"
    if abs_value >= 0.005:
        return "弱"
    return "噪音"


def build_style_performance_table(
    fr: pd.DataFrame,
    style_factors: Sequence[str],
    windows: Sequence[int] = (20, 60, 120),
    rank_window: int = 60,
) -> pd.DataFrame:
    """生成投资者主看板的显式风格因子表现表。

    排序默认使用 60 日收益，这个窗口比 20 日更稳定，又比 120 日更能捕捉切换。
    """
    valid_factors = [factor for factor in style_factors if factor in fr.columns and factor in FACTOR_INFO]
    rows: list[dict[str, object]] = []
    for factor in valid_factors:
        info = FACTOR_INFO[factor]
        row: dict[str, object] = {
            "因子": factor,
            "中文名": info["name"],
            "风格组": info["group"],
        }
        for window in windows:
            row[f"{window}日收益%"] = calc_cumulative_return(fr[factor], window=window) * 100
        ytd_return = calc_cumulative_return(fr[factor], window=None)
        latest5_return = calc_cumulative_return(fr[factor], window=min(5, len(fr[factor].dropna())))
        row["年初至今收益%"] = ytd_return * 100
        row["最新5日收益%"] = latest5_return * 100
        rank_value = calc_cumulative_return(fr[factor], window=rank_window)
        row["当前方向"] = _direction_text(factor, rank_value)
        row["投资者解读"] = f"{info['name']}：{_direction_text(factor, rank_value)}"
        row["信号强度"] = _strength_label(rank_value)
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    rank_col = f"{rank_window}日收益%"
    if rank_col in table.columns:
        table = table.sort_values(rank_col, ascending=False, na_position="last")
    return table.set_index("因子")


def _top_line(table: pd.DataFrame, ascending: bool) -> list[str]:
    col = "60日收益%" if "60日收益%" in table.columns else table.filter(like="日收益%").columns[0]
    ranked = table.sort_values(col, ascending=ascending).head(3)
    parts: list[str] = []
    for factor, row in ranked.iterrows():
        parts.append(f"{row['中文名']}({row[col]:+.2f}%，{row['当前方向']})")
    return parts


def generate_market_style_commentary(perf_table: pd.DataFrame) -> str:
    """基于表现表生成面向投资者的中文市场风格结论。"""
    if perf_table.empty:
        return "当前市场风格：没有可用的显式风格因子数据。"

    strongest = "；".join(_top_line(perf_table, ascending=False))
    weakest = "；".join(_top_line(perf_table, ascending=True))
    observations: list[str] = []

    def direction_of(factor: str) -> str | None:
        if factor not in perf_table.index:
            return None
        return str(perf_table.loc[factor, "当前方向"])

    momentum = direction_of("momentum")
    if momentum:
        observations.append(f"动量：{momentum}")
    size = direction_of("size")
    if size:
        observations.append(f"规模：{size}")
    beta = direction_of("beta")
    residual_vol = direction_of("residual_volatility")
    if beta and residual_vol:
        observations.append(f"风险偏好：{beta}，同时{residual_vol}")
    liquidity = direction_of("liquidity")
    if liquidity:
        observations.append(f"交易活跃度：{liquidity}")

    warnings: list[str] = []
    if "20日收益%" in perf_table.columns and "120日收益%" in perf_table.columns:
        for factor, row in perf_table.iterrows():
            r20 = row["20日收益%"]
            r120 = row["120日收益%"]
            if pd.notna(r20) and pd.notna(r120) and r20 * r120 < 0:
                warnings.append(f"{row['中文名']}20日与120日方向相反，可能处于切换期")
    if "最新5日收益%" in perf_table.columns and "60日收益%" in perf_table.columns:
        for factor, row in perf_table.iterrows():
            r5 = row["最新5日收益%"]
            r60 = row["60日收益%"]
            if pd.notna(r5) and pd.notna(r60) and abs(r60) >= 5 and r5 * r60 < 0:
                warnings.append(f"{row['中文名']}60日较强但最新5日反向，短期有反转风险")

    text = [
        f"当前市场风格：最强方向为 {strongest}。",
        f"当前受压方向为 {weakest}。",
    ]
    if observations:
        text.append("关键观察：" + "；".join(observations) + "。")
    if warnings:
        text.append("风险提醒：" + "；".join(warnings[:3]) + "。")
    return "\n".join(text)
```

- [ ] **Step 4: Run tests to verify performance table passes**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest '因子回测/learn/test_rq_style_dashboard.py' -q
```

Expected: PASS for all tests in the file.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add -- '因子回测/learn/rq_style_dashboard.py' '因子回测/learn/test_rq_style_dashboard.py'
git commit -m "feat: summarize rq style factor performance"
```

Expected: commit contains only the module and its test.

---

### Task 3: Plotting Functions

**Files:**
- Modify: `因子回测/learn/rq_style_dashboard.py`
- Modify: `因子回测/learn/test_rq_style_dashboard.py`

**Interfaces:**
- Consumes: `build_style_performance_table`
- Produces: `plot_style_heatmap(perf_table: pd.DataFrame, ax=None)`
- Produces: `plot_style_rank_bar(perf_table: pd.DataFrame, rank_col: str = "60日收益%", ax=None)`
- Produces: `plot_key_style_groups(fr: pd.DataFrame, groups: Mapping[str, Sequence[str]] | None = None, ax=None)`
- Produces: `plot_market_temperature(fr: pd.DataFrame, ax=None, comovement_col: str = "comovement")`

- [ ] **Step 1: Write failing plotting smoke tests**

Append to `因子回测/learn/test_rq_style_dashboard.py`:

```python
import matplotlib

matplotlib.use("Agg")

from rq_style_dashboard import (
    plot_key_style_groups,
    plot_market_temperature,
    plot_style_heatmap,
    plot_style_rank_bar,
)


def test_plotting_functions_return_axes_objects():
    dates = pd.date_range("2026-01-01", periods=130, freq="D")
    fr = pd.DataFrame(
        {
            "momentum": [0.001] * 130,
            "beta": [0.0005] * 130,
            "residual_volatility": [-0.0008] * 130,
            "size": [0.0002] * 130,
            "comovement": [0.0015] * 130,
        },
        index=dates,
    )
    table = build_style_performance_table(
        fr,
        ["momentum", "beta", "residual_volatility", "size"],
        windows=(20, 60, 120),
        rank_window=60,
    )

    assert plot_style_heatmap(table).get_title()
    assert plot_style_rank_bar(table).get_title()
    assert plot_key_style_groups(fr).get_title()
    assert plot_market_temperature(fr).get_title()
```

- [ ] **Step 2: Run tests to verify they fail because plotting functions are missing**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest '因子回测/learn/test_rq_style_dashboard.py' -q
```

Expected: FAIL with missing import or missing attribute for plotting functions.

- [ ] **Step 3: Implement plotting functions**

Append imports near the top of `因子回测/learn/rq_style_dashboard.py`:

```python
from collections.abc import Mapping

import matplotlib.pyplot as plt
import seaborn as sns
```

Append functions:

```python
DEFAULT_KEY_GROUPS: dict[str, tuple[str, ...]] = {
    "大小盘": ("size", "mid_cap"),
    "动量/反转": ("momentum", "longterm_reversal"),
    "风险偏好": ("beta", "residual_volatility"),
    "价值/成长/股息": ("book_to_price", "earnings_yield", "dividend_yield", "growth"),
    "质量": ("profitability", "earnings_quality", "investment_quality"),
}


def plot_style_heatmap(perf_table: pd.DataFrame, ax=None):
    """画多窗口风格热力图，颜色代表因子收益正负和强弱。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(4, len(perf_table) * 0.38)))
    value_cols = [col for col in perf_table.columns if col.endswith("收益%") and col != "最新5日收益%"]
    data = perf_table[value_cols].copy()
    data.index = [f"{row['中文名']} ({idx})" for idx, row in perf_table.iterrows()]
    vmax = np.nanmax(np.abs(data.values)) if data.size else 1.0
    vmax = max(vmax, 1.0)
    sns.heatmap(
        data,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "累计收益%"},
    )
    ax.set_title("显式风格因子多窗口收益热力图（正值=高暴露方向跑赢）")
    ax.set_xlabel("观察窗口")
    ax.set_ylabel("风格因子")
    return ax


def plot_style_rank_bar(perf_table: pd.DataFrame, rank_col: str = "60日收益%", ax=None):
    """画当前风格强弱排名柱状图。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, len(perf_table) * 0.36)))
    data = perf_table.sort_values(rank_col, ascending=True)
    labels = [f"{row['中文名']} ({idx})" for idx, row in data.iterrows()]
    colors = ["#d62728" if value < 0 else "#2ca02c" for value in data[rank_col]]
    ax.barh(labels, data[rank_col], color=colors, alpha=0.82)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"当前风格强弱排名：{rank_col}（正值=高暴露方向跑赢）")
    ax.set_xlabel("累计收益%")
    ax.grid(axis="x", alpha=0.25)
    return ax


def plot_key_style_groups(
    fr: pd.DataFrame,
    groups: Mapping[str, Sequence[str]] | None = None,
    ax=None,
):
    """画关键风格组的累计净值，用少量线条观察冲突关系。"""
    groups = groups or DEFAULT_KEY_GROUPS
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 7))
    for group_name, factors in groups.items():
        valid = [factor for factor in factors if factor in fr.columns]
        for factor in valid:
            label = f"{group_name}-{FACTOR_INFO[factor]['name']}"
            (1 + fr[factor].dropna()).cumprod().plot(ax=ax, linewidth=1.4, label=label)
    ax.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("关键风格对照：少量因子看市场偏好冲突")
    ax.set_ylabel("累计净值")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    return ax


def plot_market_temperature(fr: pd.DataFrame, ax=None, comovement_col: str = "comovement"):
    """单独画 comovement，作为市场整体温度，不纳入风格排名。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    if comovement_col not in fr.columns:
        ax.text(0.5, 0.5, "当前数据缺少 comovement 列", ha="center", va="center", transform=ax.transAxes)
    else:
        (1 + fr[comovement_col].dropna()).cumprod().plot(ax=ax, color="#1f77b4", linewidth=1.8)
        ax.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("市场整体温度：comovement（不参与风格排名）")
    ax.set_ylabel("累计净值")
    ax.grid(alpha=0.25)
    return ax
```

- [ ] **Step 4: Run plotting tests**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest '因子回测/learn/test_rq_style_dashboard.py' -q
```

Expected: PASS for all tests in the file.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add -- '因子回测/learn/rq_style_dashboard.py' '因子回测/learn/test_rq_style_dashboard.py'
git commit -m "feat: plot rq style factor dashboard"
```

Expected: commit contains only the module and its test.

---

### Task 4: Notebook Integration

**Files:**
- Modify: `因子回测/learn/米筐官方因子收益率_风格趋势.ipynb`
- Modify: `因子回测/learn/test_rq_style_dashboard.py`

**Interfaces:**
- Consumes: all functions from `rq_style_dashboard.py`
- Produces: Notebook section `## Part 2 · 投资者风格雷达（v2 显式风格因子）`
- Produces: Notebook section `## 研究附录 · 原始风格趋势、v1/v2 对照与行业因子`

- [ ] **Step 1: Add an integration test that checks dashboard cells exist after modification**

Append to `因子回测/learn/test_rq_style_dashboard.py`:

```python
import json
from pathlib import Path


def test_notebook_contains_investor_dashboard_section():
    nb_path = Path(__file__).with_name("米筐官方因子收益率_风格趋势.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "投资者风格雷达" in all_source
    assert "build_style_performance_table" in all_source
    assert "generate_market_style_commentary" in all_source
    assert "研究附录" in all_source
```

- [ ] **Step 2: Run the new test to verify it fails before notebook integration**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest '因子回测/learn/test_rq_style_dashboard.py::test_notebook_contains_investor_dashboard_section' -q
```

Expected: FAIL because the notebook does not yet contain the investor dashboard section.

- [ ] **Step 3: Insert dashboard cells after current Part 1 loading cell**

Use `E:\working\anaconda3\envs\quant\python.exe` to run a one-off notebook editing script from the repository root. The script must:

1. Read `因子回测/learn/米筐官方因子收益率_风格趋势.ipynb`.
2. Insert the following markdown and code cells immediately after the code cell that defines `STYLE_V2_NO_COMM`.
3. Rename the old `## Part 2 · v1 模型风格趋势` markdown to `## 研究附录 · 原始风格趋势、v1/v2 对照与行业因子`.
4. Leave all existing code cells intact.

Notebook cells to insert:

```python
dashboard_markdown = """## Part 2 · 投资者风格雷达（v2 显式风格因子）

这一部分是主看板：只看 v2 显式风格因子，不把 `comovement` 和行业因子混进风格排名。

读图口径：

- 因子收益为正：高暴露方向跑赢低暴露方向。
- 因子收益为负：低暴露方向跑赢高暴露方向。
- `comovement` 单独作为市场整体温度，不代表风格分化。
"""

dashboard_code = """# ==================== 投资者风格雷达：v2 显式风格因子 ====================
from rq_style_dashboard import (
    build_factor_direction_table,
    build_style_performance_table,
    generate_market_style_commentary,
    plot_key_style_groups,
    plot_market_temperature,
    plot_style_heatmap,
    plot_style_rank_bar,
    select_explicit_style_factors,
)

# 主看板只看有明确正负方向解释的显式风格因子。
# 这一步会自然排除中文行业因子、comovement 和暂未写入口径字典的列。
PRIMARY_MODEL = 'v2'
STYLE_WINDOWS = (20, 60, 120)
DASHBOARD_LOOKBACK_FOR_RANK = 60
PRIMARY_STYLE_FACTORS = select_explicit_style_factors(fr_v2.columns)

missing_from_v2 = [factor for factor in PRIMARY_STYLE_FACTORS if factor not in fr_v2.columns]
if missing_from_v2:
    print(f'以下风格因子在 v2 数据中缺失，将不会进入看板: {missing_from_v2}')

print(f'主看板使用 v2 显式风格因子 ({len(PRIMARY_STYLE_FACTORS)}): {PRIMARY_STYLE_FACTORS}')
print(f'数据区间: {fr_v2.index.min().date()} ~ {fr_v2.index.max().date()}，样本 {len(fr_v2)} 个交易日')

style_perf_table = build_style_performance_table(
    fr_v2,
    PRIMARY_STYLE_FACTORS,
    windows=STYLE_WINDOWS,
    rank_window=DASHBOARD_LOOKBACK_FOR_RANK,
)

direction_table = build_factor_direction_table(
    PRIMARY_STYLE_FACTORS,
    latest_returns=style_perf_table['60日收益%'].div(100) if '60日收益%' in style_perf_table.columns else None,
)

print('\\n=== 因子方向解释表（先看这个，再看图）===')
display(direction_table)

print('\\n=== 显式风格因子表现总览（默认按60日收益排序）===')
display(style_perf_table.style.format({
    col: '{:+.2f}' for col in style_perf_table.columns if col.endswith('收益%')
}))

print('\\n=== 自动市场风格结论 ===')
print(generate_market_style_commentary(style_perf_table))
"""

dashboard_plot_code = """# ==================== 投资者风格雷达图表 ====================
fig, axes = plt.subplots(2, 2, figsize=(18, 13))

plot_style_heatmap(style_perf_table, ax=axes[0, 0])
plot_style_rank_bar(style_perf_table, rank_col='60日收益%', ax=axes[0, 1])
plot_key_style_groups(fr_v2[PRIMARY_STYLE_FACTORS], ax=axes[1, 0])
plot_market_temperature(fr_v2, ax=axes[1, 1])

plt.tight_layout()
plt.show()
"""
```

The one-off script body:

```python
import json
from pathlib import Path

nb_path = Path("因子回测/learn/米筐官方因子收益率_风格趋势.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }

for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and source.startswith("## Part 2 · v1 模型风格趋势"):
        cell["source"] = [
            "## 研究附录 · 原始风格趋势、v1/v2 对照与行业因子\n",
            "\n",
            "以下内容保留原 notebook 的研究视角：v1/v2 对比、自建 Barra 对照、大类聚合和行业因子轮动。日常看市场风格时，优先使用上方“投资者风格雷达”。\n",
        ]

insert_at = None
for idx, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "STYLE_V2_NO_COMM" in source and "load_factor_return_with_cache" in source:
        insert_at = idx + 1
        break

if insert_at is None:
    raise RuntimeError("找不到 Part 1 数据加载 cell，无法插入投资者风格雷达")

existing_source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
if "投资者风格雷达（v2 显式风格因子）" not in existing_source:
    nb["cells"][insert_at:insert_at] = [
        md_cell(dashboard_markdown),
        code_cell(dashboard_code),
        code_cell(dashboard_plot_code),
    ]

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
```

- [ ] **Step 4: Run notebook integration test**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest '因子回测/learn/test_rq_style_dashboard.py::test_notebook_contains_investor_dashboard_section' -q
```

Expected: PASS.

- [ ] **Step 5: Validate notebook JSON**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -c "import json, pathlib; p=pathlib.Path('因子回测/learn/米筐官方因子收益率_风格趋势.ipynb'); json.loads(p.read_text(encoding='utf-8')); print('notebook json ok')"
```

Expected output includes `notebook json ok`.

- [ ] **Step 6: Run all dashboard tests**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest '因子回测/learn/test_rq_style_dashboard.py' -q
```

Expected: PASS for all tests.

- [ ] **Step 7: Smoke-test module against local cached v2 parquet**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -c "import sys, pathlib, pandas as pd; sys.path.insert(0, str(pathlib.Path('因子回测/learn').resolve())); from rq_style_dashboard import select_explicit_style_factors, build_style_performance_table, generate_market_style_commentary; fr=pd.read_parquet('因子回测/learn/saved_data/rq_factor_return_v2.parquet'); facs=select_explicit_style_factors(fr.columns); table=build_style_performance_table(fr, facs); print(table.head(3).to_string()); print(generate_market_style_commentary(table))"
```

Expected: prints top rows of the performance table and a Chinese market style conclusion.

- [ ] **Step 8: Commit Task 4**

Run:

```powershell
git add -- '因子回测/learn/米筐官方因子收益率_风格趋势.ipynb' '因子回测/learn/test_rq_style_dashboard.py'
git commit -m "feat: add investor rq style dashboard notebook"
```

Expected: commit contains only the notebook integration and updated test.

---

### Task 5: Final Verification And Handoff

**Files:**
- Inspect: `因子回测/learn/rq_style_dashboard.py`
- Inspect: `因子回测/learn/test_rq_style_dashboard.py`
- Inspect: `因子回测/learn/米筐官方因子收益率_风格趋势.ipynb`

**Interfaces:**
- Consumes: artifacts from Tasks 1-4
- Produces: final verification result for the user

- [ ] **Step 1: Run full dashboard test suite**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest '因子回测/learn/test_rq_style_dashboard.py' -q
```

Expected: all tests pass.

- [ ] **Step 2: Validate notebook JSON once more**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -c "import json, pathlib; p=pathlib.Path('因子回测/learn/米筐官方因子收益率_风格趋势.ipynb'); nb=json.loads(p.read_text(encoding='utf-8')); print(len(nb['cells']))"
```

Expected: prints an integer greater than the original 28 because the dashboard cells were inserted.

- [ ] **Step 3: Confirm git diff scope**

Run:

```powershell
git status --short -- '因子回测/learn/rq_style_dashboard.py' '因子回测/learn/test_rq_style_dashboard.py' '因子回测/learn/米筐官方因子收益率_风格趋势.ipynb'
```

Expected: no uncommitted changes for these three files after Task 4 commit.

- [ ] **Step 4: Report final outcome**

Final response must mention:

- 新增 `rq_style_dashboard.py`，封装方向解释、表现表、图表和自动结论。
- Notebook 新增“投资者风格雷达”主视图。
- `comovement` 和行业因子没有混入风格排名。
- 已运行的验证命令和结果。
- 如果没有执行整本 notebook，说明原因是米筐联网/权限可能影响完整执行，已用本地缓存做核心验证。

---

## Self-Review

- Spec coverage: Tasks 1-4 cover因子方向解释、多窗口表现表、热力图、60 日排名、关键风格对照、市场温度、自动中文结论、研究附录标记和验证方式。
- Scope check: The plan only touches one helper module, one test file, and one notebook. It does not modify米筐取数接口、实盘交易、数据更新或自建 Barra 逻辑。
- Type consistency: `build_style_performance_table` returns a DataFrame indexed by factor; plotting and commentary functions all consume that shape.
- Verification: Tests avoid network and use synthetic data; final smoke test uses本地 `saved_data/rq_factor_return_v2.parquet`。
