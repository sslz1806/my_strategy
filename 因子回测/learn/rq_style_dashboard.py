"""米筐风险因子收益率的投资者风格看板工具。

本模块只消费已经取好的因子收益 DataFrame，不连接米筐，也不重写缓存逻辑。
核心口径：因子收益为正表示高暴露方向跑赢低暴露方向；因子收益为负则相反。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
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
        latest5_window = min(5, len(fr[factor].dropna()))
        latest5_return = calc_cumulative_return(fr[factor], window=latest5_window)
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
    for _factor, row in ranked.iterrows():
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
        for _factor, row in perf_table.iterrows():
            r20 = row["20日收益%"]
            r120 = row["120日收益%"]
            if pd.notna(r20) and pd.notna(r120) and r20 * r120 < 0:
                warnings.append(f"{row['中文名']}20日与120日方向相反，可能处于切换期")
    if "最新5日收益%" in perf_table.columns and "60日收益%" in perf_table.columns:
        for _factor, row in perf_table.iterrows():
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
    image = ax.imshow(data.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.figure.colorbar(image, ax=ax, label="累计收益%")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data.iat[row_idx, col_idx]
            text = "" if pd.isna(value) else f"{value:.1f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8)
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
