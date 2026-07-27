"""
择时引擎核心函数。

从 sentiment_factors_5d_research.ipynb 抽取，供多基准择时测试使用。

包含模块：
    - build_value_weighted_benchmark: 构造全A市值加权基准收益
    - compute_threshold: 扩展窗口分位阈值生成
    - annualized_metrics: 年化绩效指标计算
    - run_timing: 信号驱动的连续持仓择时回测
    - summarize_timing: 择时绩效汇总
    - plot_timing_nav_comparison: 逐因子择时净值对比图
"""

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

# ============================================================
# 默认参数（与 sentiment_factors_5d_research.ipynb 保持一致）
# ============================================================
THRESHOLD_QUANTILE = 0.60
MIN_HISTORY = 252
HORIZONS_DEFAULT = (1, 3, 5, 10)
PRICE_TOLERANCE = 1e-6


# ============================================================
# build_value_weighted_benchmark：构造全A市值加权基准收益
# ============================================================
def build_value_weighted_benchmark(
    prepared: pl.DataFrame,
    calendar: pl.DataFrame,
) -> pd.DataFrame:
    """
    使用前一交易日总市值构造沪深 A 股市值加权日收益。

    Parameters
    ----------
    prepared : pl.DataFrame
        prepare_stock_daily 的输出，含 benchmark_weight 和原始 pct 字段。
    calendar : pl.DataFrame
        全市场交易日历。

    Returns
    -------
    pd.DataFrame
        含 trading_date 和 market_daily_ret 两列的日频基准表。
    """
    # 原始 pct 为百分比（3=3%），市值加权时转小数
    benchmark_daily = (
        prepared.filter(
            pl.col("code_prefix").is_in(["6", "0", "3"])
            & pl.col("benchmark_weight").is_not_null()
            & pl.col("pct").is_not_null()
        )
        .group_by("trading_date")
        .agg(
            ((pl.col("pct") / 100.0) * pl.col("benchmark_weight"))
            .sum()
            .alias("weighted_return_sum"),
            pl.col("benchmark_weight").sum().alias("weight_sum"),
        )
        .with_columns(
            (pl.col("weighted_return_sum") / pl.col("weight_sum"))
            .alias("market_daily_ret")
        )
        .select("trading_date", "market_daily_ret")
    )

    # 合并到全市场日历，无数据日返回 None
    result = calendar.select("trading_date").to_pandas().merge(
        benchmark_daily.to_pandas(), on="trading_date", how="left"
    )
    result["trading_date"] = pd.to_datetime(result["trading_date"])
    return result.sort_values("trading_date").reset_index(drop=True)


# ============================================================
# compute_threshold：用扩展窗口分位数生成择时阈值（支持单/双阈值模式）
# ============================================================
def compute_threshold(
    data: pd.DataFrame,
    factor_columns: List[str],
    quantile: float = THRESHOLD_QUANTILE,
    lower_quantile: Optional[float] = None,
    upper_quantile: Optional[float] = None,
    min_history: int = MIN_HISTORY,
) -> pd.DataFrame:
    """
    用严格滞后一日的扩展窗口分位数生成择时阈值。

    支持两种模式：
    - 单阈值模式（默认参数，仅指定 quantile）：
      生成 threshold_{factor} 列
    - 双阈值模式（同时指定 lower_quantile 和 upper_quantile）：
      生成 lower_{factor} 和 upper_{factor} 列

    每个 t 日的阈值只使用 t-1 日及更早的历史数据，不包含任何当日信息。
    前 min_history 个交易日处于预热期，阈值设为 NaN。

    Parameters
    ----------
    data : pd.DataFrame
        含因子列的研究数据。
    factor_columns : list[str]
        因子列名列表。
    quantile : float
        单阈值模式分位数（默认 THRESHOLD_QUANTILE）。
    lower_quantile : float, optional
        双阈值模式下界分位数。与 upper_quantile 同时指定时进入双阈值模式。
    upper_quantile : float, optional
        双阈值模式上界分位数。与 lower_quantile 同时指定时进入双阈值模式。
    min_history : int
        最小预热天数（默认 MIN_HISTORY）。

    Returns
    -------
    pd.DataFrame
        原表基础上添加阈值列。
    """
    result = data.copy().sort_values("trading_date").reset_index(drop=True)
    use_band = (lower_quantile is not None) and (upper_quantile is not None)

    for factor in factor_columns:
        expanding = result[factor].shift(1).expanding(min_periods=min_history)
        if use_band:
            result[f"lower_{factor}"] = expanding.quantile(lower_quantile)
            result[f"upper_{factor}"] = expanding.quantile(upper_quantile)
        else:
            result[f"threshold_{factor}"] = expanding.quantile(quantile)

    return result


# ============================================================
# annualized_metrics：由日收益计算年化指标
# ============================================================
def annualized_metrics(daily_returns: pd.Series) -> Dict[str, float]:
    """
    由日收益序列计算年化收益、最大回撤和零无风险利率夏普比率。

    Parameters
    ----------
    daily_returns : pd.Series
        日收益率序列（小数形式，如 0.01 = 1%）。

    Returns
    -------
    dict[str, float]
        含 annual_return, max_drawdown, sharpe 三个指标。
    """
    values = pd.Series(daily_returns, dtype=float).dropna()
    nav = (1 + values).cumprod()
    volatility = values.std(ddof=1)
    return {
        "annual_return": nav.iloc[-1] ** (252 / len(values)) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "sharpe": values.mean() / volatility * np.sqrt(252) if volatility > 0 else np.nan,
    }


# ============================================================
# run_timing：基于信号列做非重叠择时回测
# ============================================================
def run_timing(
    data: pd.DataFrame,
    signal_column: str,
    horizon: int,
    anchor_date: Optional[pd.Timestamp] = None,
    require_complete_exit: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    信号驱动的连续持仓择时回测。

    信号触发即开仓/续期，持有 horizon 个交易日。
    预热期信号为 NaN 时不进入样本，不改变当前仓位状态。
    空仓日仓位为 0，但每日仍保留一行收益记录，保证净值曲线连续。

    采用两遍遍历：先逐日生成仓位序列，再从仓位序列提取连续持仓/空仓段。

    Parameters
    ----------
    data : pd.DataFrame
        含 trading_date、signal_column（数值型，1=持仓/0=空仓/NaN=无效）
        和 market_daily_ret 的日频表。
    signal_column : str
        信号列名。
    horizon : int
        持有周期（交易日数），同时也是续期长度。
    anchor_date : pd.Timestamp, optional
        回测起始日。若为 None，从 signal_column 首个非空值所在行起算。
    require_complete_exit : bool
        若为 True，尾部不完整的持仓段不进入 blocks 统计。

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (daily, blocks)
        - daily: 逐日明细，含 trading_date, position, market_daily_ret,
                 strategy_daily_ret, strategy_nav, benchmark_nav
        - blocks: 连续持仓/空仓段明细，每行一个完整的持仓段或空仓段。
    """
    ordered = data.copy().sort_values("trading_date").reset_index(drop=True)
    ordered["trading_date"] = pd.to_datetime(ordered["trading_date"])

    # 确定锚点行索引
    if anchor_date is not None:
        anchor_index = int(
            ordered.index[ordered["trading_date"] == pd.Timestamp(anchor_date)][0]
        )
    else:
        first_valid = ordered[signal_column].first_valid_index()
        if first_valid is None:
            return pd.DataFrame(), pd.DataFrame()
        anchor_index = int(first_valid)

    n = len(ordered)

    # ===== 第一遍：生成逐日仓位序列 =====
    # positions[i] 对应第 anchor_index + i 个决策日的次日仓位
    positions = np.zeros(n - 1 - anchor_index)
    holding_until_idx = -1

    for offset, t_idx in enumerate(range(anchor_index, n - 1)):
        signal = ordered.iloc[t_idx][signal_column]
        # 预热期 NaN 不触发，也不改变当前仓位
        if not pd.isna(signal) and signal == 1:
            new_until = t_idx + horizon
            if new_until > holding_until_idx:
                holding_until_idx = new_until

        positions[offset] = 1.0 if t_idx + 1 <= holding_until_idx else 0.0

    # ===== 第二遍：构建逐日明细 =====
    daily_rows = []
    for offset, t_idx in enumerate(range(anchor_index, n - 1)):
        daily_rows.append({
            "trading_date": ordered.iloc[t_idx + 1]["trading_date"],
            "position": positions[offset],
            "market_daily_ret": ordered.iloc[t_idx + 1]["market_daily_ret"],
            "strategy_daily_ret": positions[offset] * ordered.iloc[t_idx + 1]["market_daily_ret"],
        })
    daily = pd.DataFrame(daily_rows)

    # ===== 第三遍：从仓位序列提取连续持仓/空仓段 =====
    block_rows = []
    seg_start = 0
    seg_position = positions[0]

    for i in range(1, len(positions)):
        if positions[i] != seg_position:
            # 仓位变化：结束当前段
            start_tidx = anchor_index + seg_start
            end_tidx = anchor_index + i - 1
            is_holding = seg_position > 0.5
            # 尾部持仓段不完整时跳过
            incomplete = is_holding and require_complete_exit and (start_tidx + horizon >= n)
            if not incomplete:
                block_slice = ordered.iloc[start_tidx + 1 : end_tidx + 1]
                if len(block_slice) > 0:
                    bench_ret = (1 + block_slice["market_daily_ret"]).prod() - 1
                    block_rows.append({
                        "block_id": start_tidx,
                        "position": seg_position,
                        "decision_date": ordered.iloc[start_tidx]["trading_date"],
                        "block_start_date": ordered.iloc[start_tidx + 1]["trading_date"],
                        "block_end_date": ordered.iloc[end_tidx]["trading_date"],
                        "block_duration": end_tidx - start_tidx,
                        "benchmark_block_return": bench_ret,
                        "strategy_block_return": seg_position * bench_ret,
                    })
            seg_start = i
            seg_position = positions[i]

    # 最后一个段
    if seg_start < len(positions):
        start_tidx = anchor_index + seg_start
        end_tidx = anchor_index + len(positions) - 1
        is_holding = seg_position > 0.5
        incomplete = is_holding and require_complete_exit and (start_tidx + horizon >= n)
        if not incomplete:
            block_slice = ordered.iloc[start_tidx + 1 : end_tidx + 1]
            if len(block_slice) > 0:
                bench_ret = (1 + block_slice["market_daily_ret"]).prod() - 1
                block_rows.append({
                    "block_id": start_tidx,
                    "position": seg_position,
                    "decision_date": ordered.iloc[start_tidx]["trading_date"],
                    "block_start_date": ordered.iloc[start_tidx + 1]["trading_date"],
                    "block_end_date": ordered.iloc[end_tidx]["trading_date"],
                    "block_duration": end_tidx - start_tidx,
                    "benchmark_block_return": bench_ret,
                    "strategy_block_return": seg_position * bench_ret,
                })

    blocks = pd.DataFrame(block_rows)

    if len(daily) > 0:
        daily["benchmark_nav"] = (1 + daily["market_daily_ret"]).cumprod()
        daily["strategy_nav"] = (1 + daily["strategy_daily_ret"]).cumprod()

    return daily, blocks


# ============================================================
# summarize_timing：汇总单因子择时绩效
# ============================================================
def summarize_timing(
    daily: pd.DataFrame,
    blocks: pd.DataFrame,
    factor: str,
    horizon: int,
    factor_labels: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """
    汇总单因子择时的持仓胜率、择时命中率和风险收益指标。

    统计基于 blocks（连续持仓/空仓段），而非固定 n 天非重叠 block。
    - holding_win_rate: 持仓段中 benchmark_block_return > 0 的比例
    - timing_hit_rate: 择时猜对方向的比例（持仓且基准涨 + 空仓且基准跌）

    Parameters
    ----------
    daily : pd.DataFrame
        run_timing 返回的逐日明细。
    blocks : pd.DataFrame
        run_timing 返回的连续持仓/空仓段明细。
    factor : str
        因子列名。
    horizon : int
        持有周期。
    factor_labels : dict[str, str], optional
        因子显示标签映射。若为 None，使用空字典，因子列名直接作为显示标签。

    Returns
    -------
    dict[str, object]
        含 factor, horizon, rebalance_count, holding_ratio, holding_win_rate,
        timing_hit_rate, annual_return, benchmark_annual_return, max_drawdown,
        sharpe, final_nav, benchmark_final_nav, relative_final_nav 等指标。
    """
    if factor_labels is None:
        factor_labels = {}

    if len(blocks) > 0:
        held_blocks = blocks.loc[blocks["position"] > 0.5]
        holding_win_rate = (
            (held_blocks["benchmark_block_return"] > 0).mean()
            if len(held_blocks)
            else np.nan
        )
        timing_hit_rate = (
            ((blocks["position"] > 0.5) & (blocks["benchmark_block_return"] > 0))
            | ((blocks["position"] <= 0.5) & (blocks["benchmark_block_return"] <= 0))
        ).mean()
    else:
        holding_win_rate = np.nan
        timing_hit_rate = np.nan

    strategy_metrics = annualized_metrics(daily["strategy_daily_ret"])
    benchmark_metrics = annualized_metrics(daily["market_daily_ret"])

    return {
        "factor": factor,
        "factor_label": factor_labels.get(factor, factor),
        "horizon": horizon,
        "rebalance_count": len(blocks.loc[blocks["position"] > 0.5]) if len(blocks) > 0 else 0,
        "holding_ratio": daily["position"].mean() if len(daily) > 0 else np.nan,
        "holding_win_rate": holding_win_rate,
        "timing_hit_rate": timing_hit_rate,
        "annual_return": strategy_metrics["annual_return"],
        "benchmark_annual_return": benchmark_metrics["annual_return"],
        "max_drawdown": strategy_metrics["max_drawdown"],
        "sharpe": strategy_metrics["sharpe"],
        "final_nav": daily["strategy_nav"].iloc[-1] if len(daily) > 0 else np.nan,
        "benchmark_final_nav": daily["benchmark_nav"].iloc[-1] if len(daily) > 0 else np.nan,
        "relative_final_nav": (
            daily["strategy_nav"].iloc[-1] / daily["benchmark_nav"].iloc[-1] - 1
        ) if len(daily) > 0 else np.nan,
    }


# ============================================================
# plot_timing_nav_comparison：逐因子对比择时净值与基准净值
# ============================================================
def plot_timing_nav_comparison(
    daily_results: Dict[Tuple[str, int], pd.DataFrame],
    factor_columns: List[str],
    horizons: Tuple[int, ...] = HORIZONS_DEFAULT,
    start_date=None,
    factor_labels: Optional[Dict[str, str]] = None,
    benchmark_label: str = "全A市值加权基准",
) -> None:
    """
    逐因子对比四个调仓周期的择时净值与同期满仓基准。

    daily_results 的键为 (factor, horizon)，值来自
    run_timing 的逐日明细。两条净值曲线共享相同交易日，
    因此图中的差异只来自仓位信号，不受样本区间错位影响。

    Parameters
    ----------
    daily_results : dict[tuple[str, int], pd.DataFrame]
        键为 (factor, horizon)，值为含 strategy_nav/benchmark_nav 的日净值表。
    factor_columns : list[str]
        因子列名列表。
    horizons : tuple[int, ...]
        未来收益周期列表。
    start_date : str or pd.Timestamp, optional
        绘图起始日期。若指定，所有子图只显示该日期之后的数据，并将两条
        净值曲线重归一化到起始日 = 1.0。默认从数据起点开始。
    factor_labels : dict[str, str], optional
        因子显示标签映射。若为 None，使用因子列名本身作为显示标签。
    benchmark_label : str
        基准图例名称（默认 "全A市值加权基准"）。在多基准对比时，
        替换为该基准的描述。
    """
    if factor_labels is None:
        factor_labels = {f: f for f in factor_columns}

    for factor in factor_columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        for axis, horizon in zip(axes.ravel(), horizons):
            detail = daily_results[(factor, horizon)].copy()

            # 若指定起始日期：过滤无效预热期、重归一化起点
            if start_date is not None:
                start = pd.Timestamp(start_date)
                detail = detail[detail["trading_date"] >= start].copy()
                if len(detail) > 0:
                    detail["benchmark_nav"] /= detail["benchmark_nav"].iloc[0]
                    detail["strategy_nav"] /= detail["strategy_nav"].iloc[0]

            # 基准净值：黑色虚线
            axis.plot(
                detail["trading_date"],
                detail["benchmark_nav"],
                label=benchmark_label,
                color="black",
                linestyle="--",
                linewidth=1.4,
            )
            # 策略净值：红色实线
            axis.plot(
                detail["trading_date"],
                detail["strategy_nav"],
                label="单因子择时",
                color="#C44E52",
                linewidth=1.6,
            )
            axis.set_title(
                f"{horizon} 日调仓｜持仓比例 {detail['position'].mean():.1%}"
            )
            axis.grid(alpha=0.25)
            axis.legend()

        fig.suptitle(
            f"{factor_labels.get(factor, factor)}：单因子择时净值与基准",
            fontsize=15,
        )
        plt.show()
