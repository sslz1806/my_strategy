"""五日涨跌停情绪因子的公共研究函数。"""

import warnings
from datetime import date
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats

THRESHOLD_QUANTILE = 0.60
MIN_HISTORY = 252
HORIZONS = (1, 3, 5, 10)
ROLLING_IC_WINDOWS = (50, 100)
ROLLING_IC_MIN_VALID_RATIO = 0.95
IC_NEUTRAL_BAND = 0.02
PRICE_TOLERANCE = 1e-6
FACTOR_LABELS = {
    "limit_up_ratio": "涨停占比",
    "limit_down_ratio": "跌停占比",
    "net_limit_ratio": "净涨停占比（涨跌停强度）",
    "limit_up_down_ratio": "涨跌停比值",
    "limit_up_next_ret": "涨停次日收益",
    "limit_down_next_ret": "跌停次日收益",
}
FACTOR_DIRECTIONS = {
    "limit_up_ratio": 1,
    "limit_down_ratio": 1,
    "net_limit_ratio": 1,
    "limit_up_down_ratio": 1,
    "limit_up_next_ret": 1,
    "limit_down_next_ret": 1,
}
FACTOR_COLUMNS = list(FACTOR_LABELS)

def prepare_stock_daily(
    daily_raw: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    清洗逐股日线，对齐交易日，筛选因子股票池，标记涨停/跌停。

    内部完成沪深 A 股筛选（排除 ST、停牌、价格异常），返回已过滤的日线数据。
    下游函数接收的 prepared 已经是合格股票，无需再判断 eligible。

    Parameters
    ----------
    daily_raw : pl.DataFrame
        原始日线，含 code, trading_date, close, pre_close, limit_up,
        limit_down, is_st, is_suspended, total_mv。

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        (prepared, calendar)
        - prepared: 仅含因子合格股票的清洗对齐数据
        - calendar: 全市场交易日历
    """
    # 1. 选择需要的基础字段，按股票+日期排序
    columns = [
        "code", "trading_date", "close", "pre_close", "limit_up",
        "limit_down", "is_st", "is_suspended", "total_mv",
    ]
    data = (
        daily_raw.select(columns)
        .with_columns(
            pl.col("trading_date").cast(pl.Date),
            (pl.col("close") / pl.col("pre_close") - 1).alias("stock_daily_ret"),
        )
        .sort(["code", "trading_date"])
    )

    # 2. 生成全市场交易日历（仅保留 next_market_date 供次日收益校验用）
    calendar = (
        data.select("trading_date")
        .unique()
        .sort("trading_date")
        .with_columns(
            pl.col("trading_date").shift(-1).alias("next_market_date"),
        )
    )

    # 3. 按股票分组获得前日市值和次一收盘价
    data = data.join(calendar, on="trading_date", how="left").with_columns(
        pl.col("total_mv").shift(1).over("code").alias("previous_total_mv"),
        pl.col("trading_date").shift(-1).over("code").alias("next_stock_date"),
        pl.col("close").shift(-1).over("code").alias("next_close"),
        # 代码第6位（首位数字）：6=沪市，0=深市主板，3=创业板
        pl.col("code").str.slice(5, 1).alias("code_prefix"),
    )

    # 4. 因子股票池条件（内部使用，不输出为列）
    eligible = (
        pl.col("code_prefix").is_in(["6", "0", "3"])
        & ~pl.col("is_st").fill_null(False).cast(pl.Boolean)
        & ~pl.col("is_suspended").fill_null(False).cast(pl.Boolean)
        & (pl.col("close") > 0)
        & (pl.col("pre_close") > 0)
        & (pl.col("limit_up") > 0)
        & (pl.col("limit_down") > 0)
    )

    # 5. 标记涨停/跌停（只对合格股票标记，避免北交所等干扰）
    data = data.with_columns(
        (
            eligible
            & ((pl.col("close") - pl.col("limit_up")).abs() <= PRICE_TOLERANCE)
        ).alias("is_limit_up"),
        (
            eligible
            & ((pl.col("close") - pl.col("limit_down")).abs() <= PRICE_TOLERANCE)
        ).alias("is_limit_down"),
    )

    # 6. 事件次日收益：只在该股票下一日恰为市场下一交易日时才有效
    #    防止停牌造成的跨期收益被错误归到因子日期
    data = data.with_columns(
        pl.when(
            (pl.col("next_stock_date") == pl.col("next_market_date"))
            & (pl.col("next_close") > 0)
        )
        .then(pl.col("next_close") / pl.col("close") - 1)
        .otherwise(None)
        .alias("event_next_ret"),
    )

    # 7. 基准权重：前一交易日总市值
    #    注：停牌时前日市值仍可用于加权（总市值非隔夜跳变）
    data = data.with_columns(
        pl.when(pl.col("previous_total_mv") > 0)
        .then(pl.col("previous_total_mv"))
        .otherwise(None)
        .alias("benchmark_weight"),
    )

    # 8. 过滤：只返回合格股票（下游无需再筛选）
    data = data.filter(eligible)
    return data, calendar


def build_daily_sentiment_factors(
    prepared: pl.DataFrame,
    calendar: pl.DataFrame,
    window: int = 5,
) -> pl.DataFrame:
    """
    一次 group_by 构造六个涨跌停情绪因子。

    四个比率因子（滚动窗口内累计）：
      - limit_up_ratio: 涨停占比 = 5日涨停数 / 5日最大合格股票数
      - limit_down_ratio: 跌停占比
      - net_limit_ratio: 净涨停占比
      - limit_up_down_ratio: 涨跌停比值（跌停数为0时返回Null）

    两个收益因子（滚动窗口内事件加权平均）：
      - limit_up_next_ret: 涨停次日收益均值
      - limit_down_next_ret: 跌停次日收益均值

    收益列通过 shift(1) 对齐到兑现日（t 日涨停的次日在 t+1），
    替代原版按 next_market_date 分组再 join 的做法，数学等价。
    """
    # 1. 一次聚合：计数 + 次日收益总和
    daily = (
        prepared.group_by("trading_date")
        .agg(
            n_stock=pl.col("code").n_unique(),
            n_up=pl.col("is_limit_up").cast(pl.Int64).sum(),
            n_down=pl.col("is_limit_down").cast(pl.Int64).sum(),
            up_ret=pl.col("event_next_ret").filter(pl.col("is_limit_up")).sum(),
            down_ret=pl.col("event_next_ret").filter(pl.col("is_limit_down")).sum(),
        )
        .sort("trading_date")
    )

    # 2. 对齐到全市场日历，无事件日补 0
    result = (
        calendar.select("trading_date")
        .join(daily, on="trading_date", how="left")
        .sort("trading_date")
        .with_columns(pl.col("n_stock", "n_up", "n_down", "up_ret", "down_ret").fill_null(0))
    )

    # 3. 滚动窗口 → 六个因子（select 里直出，不保留中间列）
    denom = pl.when(pl.col("n_stock").rolling_max(window, min_periods=window) > 0) \
              .then(pl.col("n_stock").rolling_max(window, min_periods=window)).otherwise(None)

    return result.select(
        "trading_date",
        limit_up_ratio=pl.col("n_up").rolling_sum(window, min_periods=window) / denom,
        limit_down_ratio=pl.col("n_down").rolling_sum(window, min_periods=window) / denom,
        net_limit_ratio=(
            pl.col("n_up").rolling_sum(window, min_periods=window)
            - pl.col("n_down").rolling_sum(window, min_periods=window)
        ) / denom,
        limit_up_down_ratio=pl.when(pl.col("n_down").rolling_sum(window, min_periods=window) > 0)
            .then(pl.col("n_up").rolling_sum(window, min_periods=window)
                  / pl.col("n_down").rolling_sum(window, min_periods=window))
            .otherwise(None),
        # 收益 shift(1)：t 日涨停确认收益在 t+1，因子 t 只能看到 t-1 及以前的收益
        limit_up_next_ret=(
            pl.col("up_ret").shift(1).rolling_sum(window, min_periods=window)
            / pl.col("n_up").shift(1).rolling_sum(window, min_periods=window)
        ),
        limit_down_next_ret=(
            pl.col("down_ret").shift(1).rolling_sum(window, min_periods=window)
            / pl.col("n_down").shift(1).rolling_sum(window, min_periods=window)
        ),
    )


def build_value_weighted_benchmark(
    prepared: pl.DataFrame,
    calendar: pl.DataFrame,
) -> pd.DataFrame:
    """
    使用前一交易日总市值构造沪深 A 股市值加权日收益。

    Parameters
    ----------
    prepared : pl.DataFrame
        prepare_stock_daily 的输出，含 benchmark_weight 和清洗后的 stock_daily_ret 字段。
    calendar : pl.DataFrame
        全市场交易日历。

    Returns
    -------
    pd.DataFrame
        含 trading_date 和 market_daily_ret 两列的日频基准表。
    """
    # stock_daily_ret 已是小数收益率
    benchmark_daily = (
        prepared.filter(
            pl.col("code_prefix").is_in(["6", "0", "3"])
            & pl.col("benchmark_weight").is_not_null()
            & pl.col("stock_daily_ret").is_not_null()
        )
        .group_by("trading_date")
        .agg(
            (pl.col("stock_daily_ret") * pl.col("benchmark_weight"))
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


def compute_threshold(
    data: pd.DataFrame,
    factor_columns: list[str] = FACTOR_COLUMNS,
    quantile: float = THRESHOLD_QUANTILE,
    lower_quantile: float = None,
    upper_quantile: float = None,
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


def annualized_metrics(daily_returns: pd.Series) -> dict[str, float]:
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


def run_timing(
    data: pd.DataFrame,
    signal_column: str,
    horizon: int,
    anchor_date: pd.Timestamp = None,
    require_complete_exit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def summarize_timing(
    daily: pd.DataFrame,
    blocks: pd.DataFrame,
    factor: str,
    horizon: int,
) -> dict[str, object]:
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

    Returns
    -------
    dict[str, object]
        含 factor, horizon, rebalance_count, holding_ratio, holding_win_rate,
        timing_hit_rate, annual_return, benchmark_annual_return, max_drawdown,
        sharpe, final_nav, benchmark_final_nav, relative_final_nav 等指标。
    """
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
        "factor_label": FACTOR_LABELS.get(factor, factor),
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


def plot_timing_nav_comparison(
    daily_results: dict[tuple[str, int], pd.DataFrame],
    factor_columns: list[str] = FACTOR_COLUMNS,
    horizons: tuple[int, ...] = HORIZONS,
    start_date=None,
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
    """
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
                label="全 A 市值加权基准",
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
            f"{FACTOR_LABELS[factor]}：单因子择时净值与基准",
            fontsize=15,
        )
        plt.show()


def analyze_ic(
    data: pd.DataFrame,
    factor_columns: list[str] = FACTOR_COLUMNS,
    horizons: tuple[int, ...] = HORIZONS,
    factor_directions: dict[str, int] = FACTOR_DIRECTIONS,
) -> pd.DataFrame:
    """
    计算各因子与不同周期未来收益的 Pearson、Spearman 时序 IC。

    这里计算的是市场时序 IC（单列因子 vs 单列未来收益），而不是截面选股 IC。
    
    - Pearson IC: 线性相关系数，衡量因子与未来收益的线性关系
    - Spearman IC: 秩相关系数，衡量因子与未来收益的排序一致性
    - 方向调整 IC: Pearson IC × 因子方向（方便六个指标统一按"越大越看多"比较）

    Parameters
    ----------
    data : pd.DataFrame
        含因子列和 future_market_daily_ret_{n}d 列的研究数据。
    factor_columns : list[str]
        因子列名列表。
    horizons : tuple[int, ...]
        未来收益周期列表。
    factor_directions : dict[str, int]
        因子方向字典：+1=正向，-1=反向。

    Returns
    -------
    pd.DataFrame
        每行一个(因子, 周期)组合的 IC 指标。
    """
    rows = []
    for factor in factor_columns:
        # 因子方向：用于计算方向调整 IC
        direction = factor_directions.get(factor, 1)
        for horizon in horizons:
            target = f"future_market_daily_ret_{horizon}d"
            # 剔除 inf/NaN 后再计算相关性
            sample = (
                data[[factor, target]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            # Pearson 线性相关
            pearson_ic = sample[factor].corr(sample[target], method="pearson")
            # Spearman 秩相关（稳健性检验）
            spearman_ic = sample[factor].corr(sample[target], method="spearman")
            rows.append(
                {
                    "factor": factor,
                    "factor_label": FACTOR_LABELS.get(factor, factor),
                    "horizon": horizon,
                    "n_obs": len(sample),
                    "pearson_ic": pearson_ic,
                    "spearman_ic": spearman_ic,
                    "directional_pearson_ic": pearson_ic * direction,
                }
            )
    return pd.DataFrame(rows)


def report_ic_summary(
    summary: pd.DataFrame,
    factor_columns: list[str] = FACTOR_COLUMNS,
    horizons: tuple[int, ...] = HORIZONS,
) -> None:
    """
    展示 IC 汇总表格，并绘制原始及方向调整后的 Pearson IC 热力图。

    热力图左侧为原始 IC（反映真实相关方向），右侧为方向调整 IC（方便统一比较）。
    颜色深浅映射 IC 绝对值大小，红色=正，蓝色=负。

    Parameters
    ----------
    summary : pd.DataFrame
        analyze_ic 的输出 DataFrame。
    factor_columns : list[str]
        因子列名列表（决定显示顺序）。
    horizons : tuple[int, ...]
        未来收益周期列表（决定 x 轴顺序）。
    """
    # 1. 展示汇总数字表
    display_table = summary[
        [
            "factor_label", "horizon", "n_obs", "pearson_ic",
            "spearman_ic", "directional_pearson_ic",
        ]
    ].rename(
        columns={
            "factor_label": "因子",
            "horizon": "未来日数",
            "n_obs": "样本数",
            "pearson_ic": "Pearson IC",
            "spearman_ic": "Spearman IC",
            "directional_pearson_ic": "方向调整 IC",
        }
    )
    display(
        display_table.style.format(
            {
                "Pearson IC": "{:.4f}",
                "Spearman IC": "{:.4f}",
                "方向调整 IC": "{:.4f}",
            }
        )
    )

    # 2. 绘制热力图：原始 IC（左）和方向调整 IC（右）
    factor_order = [FACTOR_LABELS.get(factor, factor) for factor in factor_columns]
    heatmap_specs = (
        ("pearson_ic", "原始 Pearson IC"),
        ("directional_pearson_ic", "方向调整 Pearson IC"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, (metric, title) in zip(axes, heatmap_specs):
        # pivot 成 6因子 × 4周期 的矩阵
        matrix = (
            summary.pivot(index="factor_label", columns="horizon", values=metric)
            .reindex(index=factor_order, columns=horizons)
        )
        values = matrix.to_numpy(dtype=float)
        # 自动确定色阶范围，避免极端值冲淡中间色调
        finite_values = np.abs(values[np.isfinite(values)])
        limit = max(0.05, float(finite_values.max())) if finite_values.size else 0.05
        image = axis.imshow(
            values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit
        )
        # 标记坐标轴
        axis.set_xticks(
            range(len(horizons)), [f"{horizon}日" for horizon in horizons]
        )
        axis.set_yticks(range(len(factor_order)), factor_order)
        axis.set_title(title)
        # 在每个格子中标注 IC 数值
        for row_index, row in enumerate(values):
            for column_index, value in enumerate(row):
                label = f"{value:.3f}" if np.isfinite(value) else "—"
                axis.text(
                    column_index, row_index, label, ha="center", va="center"
                )
        fig.colorbar(image, ax=axis, shrink=0.82)
    plt.show()


def format_optional_date(value: object) -> str:
    """
    格式化可选日期；缺失日期在汇总表中显示为破折号。
    
    用于滚动 IC 汇总表的最新可用日列，若窗口尚未完整则显示 —。
    """
    if pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def compute_rolling_ic(
    data: pd.DataFrame,
    factor_columns: list[str] = FACTOR_COLUMNS,
    horizons: tuple[int, ...] = HORIZONS,
    windows: tuple[int, ...] = ROLLING_IC_WINDOWS,
    min_valid_ratio: float = ROLLING_IC_MIN_VALID_RATIO,
) -> pd.DataFrame:
    """
    计算滚动 Pearson IC，并同时记录窗口结束日和真实可用日。

    因子日期 t 对应的未来 n 日收益要到 t+n 收盘后才完全兑现，因此图表必须
    使用 available_date 作横轴；factor_window_end_date 仅用于追溯原窗口。

    窗口跨度固定为交易日数，完整预热后允许少量缺失；有效配对数必须达到
    ceil(window * min_valid_ratio) 才输出 IC，并随结果记录实际样本量。

    Parameters
    ----------
    data : pd.DataFrame
        含因子列和 future_market_daily_ret_{n}d 列的研究数据。
    factor_columns : list[str]
        因子列名列表。
    horizons : tuple[int, ...]
        未来收益周期列表。
    windows : tuple[int, ...]
        滚动窗口列表，每个窗口独立输出一条 IC 序列。
    min_valid_ratio : float
        窗口内最少有效配对比例（默认 0.95 = 95%）。

    Returns
    -------
    pd.DataFrame
        长格式 DataFrame，每行对应一个(因子, 周期, 窗口, 交易日)的滚动 IC 值。
    """
    # 参数校验
    if not 0 < min_valid_ratio <= 1:
        raise ValueError("min_valid_ratio 必须在 (0, 1] 范围内")

    ordered = data.copy().sort_values("trading_date").reset_index(drop=True)
    rows = []
    
    # 遍历因子 × 周期 × 窗口，生成每条 IC 序列
    for factor in factor_columns:
        for horizon in horizons:
            target = f"future_market_daily_ret_{horizon}d"
            # 剔除 inf 避免相关计算异常
            factor_values = ordered[factor].replace([np.inf, -np.inf], np.nan)
            target_values = ordered[target].replace([np.inf, -np.inf], np.nan)
            # 有效配对：因子值和目标收益均非空
            pair_is_valid = factor_values.notna() & target_values.notna()
            
            for window in windows:
                # 最少需要的有效配对数
                min_required_obs = int(np.ceil(window * min_valid_ratio))
                # 滚动窗口内的有效配对计数
                rolling_n_obs = (
                    pair_is_valid.rolling(window, min_periods=1).sum().astype(int)
                )
                # 滚动 Pearson 相关系数
                rolling_ic = factor_values.rolling(
                    window, min_periods=min_required_obs
                ).corr(target_values)
                
                # 前 window-1 天属于预热期，不输出 IC
                # 同时要求有效配对数 >= 最低要求
                full_window = pd.Series(
                    np.arange(len(ordered)) >= window - 1,
                    index=ordered.index,
                )
                rolling_ic = rolling_ic.where(
                    full_window & (rolling_n_obs >= min_required_obs)
                )
                
                rows.append(
                    pd.DataFrame(
                        {
                            "factor": factor,
                            "factor_label": FACTOR_LABELS.get(factor, factor),
                            "horizon": horizon,
                            "window": window,
                            "factor_window_end_date": ordered["trading_date"],
                            "available_date": ordered["trading_date"].shift(-horizon),
                            "rolling_n_obs": rolling_n_obs,
                            "min_required_obs": min_required_obs,
                            "rolling_ic": rolling_ic,
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def build_non_overlapping_cumulative_ic(
    series: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    从滚动 IC 序列抽取互不重叠窗口，并计算区块 IC 累计和。

    第一个区块覆盖第 1～window 个交易日，之后每隔 window 个交易日
    取一次滚动 IC；因此相邻累计点使用的因子窗口没有交集。
    这避免了滚动 IC 中相邻窗口高度重叠带来的视觉平滑假象。

    Parameters
    ----------
    series : pd.DataFrame
        单条滚动 IC 序列（已按 factor_window_end_date 排序）。
    window : int
        抽取步长（与滚动窗口大小一致）。

    Returns
    -------
    pd.DataFrame
        含 block_ic（非重叠区块 IC）和 cumulative_ic（累计和）的抽样结果。
    """
    if window <= 0:
        raise ValueError("window 必须是正整数")

    ordered = series.sort_values("factor_window_end_date").reset_index(drop=True)
    # 每隔 window 个交易日取一个点
    endpoints = ordered.iloc[window - 1 :: window].copy()
    endpoints = endpoints.dropna(subset=["available_date"]).rename(
        columns={"rolling_ic": "block_ic"}
    )
    endpoints["cumulative_ic"] = endpoints["block_ic"].cumsum()
    return endpoints


def plot_rolling_ic_history(
    rolling_detail: pd.DataFrame,
    full_ic_summary: pd.DataFrame,
    factor_columns: list[str] = FACTOR_COLUMNS,
    horizons: tuple[int, ...] = HORIZONS,
    windows: tuple[int, ...] = ROLLING_IC_WINDOWS,
    neutral_band: float = IC_NEUTRAL_BAND,
) -> None:
    """
    逐因子绘制四个未来周期的滚动 IC 历史走势。

    每张图包含四个周期子图（1/3/5/10 日）；左轴展示当前配置窗口的滚动 IC、
    零轴和全样本 IC 参考线以及 ±0.02 中性区间，右轴展示非重叠区块 IC 累计和。
    横轴固定使用 available_date，确保图中每个点只在未来收益已经完整兑现后出现。

    Parameters
    ----------
    rolling_detail : pd.DataFrame
        compute_rolling_ic 的输出，长格式 IC 序列。
    full_ic_summary : pd.DataFrame
        analyze_ic 的输出，用于在全样本 IC 参考线。
    factor_columns : list[str]
        因子列名列表。
    horizons : tuple[int, ...]
        未来收益周期列表。
    windows : tuple[int, ...]
        滚动窗口列表。
    neutral_band : float
        IC 中性区间半宽（默认 ±0.02）。
    """
    # 构建全样本 IC 查询表：(因子, 周期) → Pearson IC
    full_ic_lookup = full_ic_summary.set_index(["factor", "horizon"])["pearson_ic"]
    # 给每个窗口分配一个颜色
    colors = {window: f"C{index}" for index, window in enumerate(windows)}

    for factor in factor_columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        for axis, horizon in zip(axes.ravel(), horizons):
            # 创建右轴（非重叠累计 IC）
            cumulative_axis = axis.twinx()
            
            for window in windows:
                color = colors[window]
                # 提取当前(因子, 周期, 窗口)的 IC 序列
                series = rolling_detail[
                    (rolling_detail["factor"] == factor)
                    & (rolling_detail["horizon"] == horizon)
                    & (rolling_detail["window"] == window)
                ].sort_values("factor_window_end_date")
                # 过滤尚未兑现的日期（available_date 为空），保留 IC 的 NaN
                rolling_series = series.dropna(subset=["available_date"])
                
                # 左轴：滚动 IC 曲线
                axis.plot(
                    rolling_series["available_date"],
                    rolling_series["rolling_ic"],
                    label=f"{window} 日滚动 IC",
                    color=color,
                    linewidth=1.3 if window == min(windows) else 1.8,
                )
                
                # 右轴：非重叠区块累计 IC（点线图）
                cumulative = build_non_overlapping_cumulative_ic(series, window)
                cumulative_axis.plot(
                    cumulative["available_date"],
                    cumulative["cumulative_ic"],
                    label=f"{window} 日非重叠累计 IC",
                    color=color,
                    linestyle="--",
                    marker="o",
                    markersize=3.5,
                    linewidth=1.1,
                    alpha=0.85,
                )

            # 参考线：零轴、中性区间、全样本 IC
            axis.axhline(0, color="black", linewidth=0.9)
            axis.axhspan(-neutral_band, neutral_band, color="grey", alpha=0.12)
            axis.axhline(
                full_ic_lookup.loc[(factor, horizon)],
                color="#55A868",
                linestyle="--",
                linewidth=1.0,
                label="全样本 IC",
            )
            
            axis.set_title(f"未来 {horizon} 日")
            axis.set_ylabel("Pearson IC")
            cumulative_axis.set_ylabel("非重叠累计 IC")
            cumulative_axis.axhline(
                0, color="grey", linestyle=":", linewidth=0.7, alpha=0.7
            )
            cumulative_axis.tick_params(axis="y", labelsize=8)
            axis.grid(alpha=0.25)
            
            # 合并左右轴的图例
            left_handles, left_labels = axis.get_legend_handles_labels()
            right_handles, right_labels = cumulative_axis.get_legend_handles_labels()
            axis.legend(
                left_handles + right_handles,
                left_labels + right_labels,
                fontsize=8,
            )

        fig.suptitle(
            f"{FACTOR_LABELS[factor]}：滚动时序 IC（按真实可用日期）",
            fontsize=15,
        )
        plt.show()


def run_multi_benchmark_timing(
    factor_daily: pl.DataFrame,
    benchmarks: List[str],
    prepared_daily: pl.DataFrame,
    calendar: pl.DataFrame,
    start_date: date,
    end_date: date,
    horizons: Tuple[int, ...] = (1, 3, 5, 10),
    lower_quantile: float = 0.65,
    upper_quantile: float = 1.0,
    min_history: int = 252,
    factor_columns: List[str] = None,
    factor_labels: Dict[str, str] = None,
    benchmark_data_source: str = "auto",
) -> dict:
    """
    对多个基准运行同一套情绪因子的择时回测。

    返回:
        summary: pd.DataFrame — 每行一个(基准, 因子, 持有期)的绩效汇总
        daily: dict — {(基准, 因子, 持有期): 逐日明细}
    """
    from 因子回测.涨跌停情绪因子.benchmark_loader import load_benchmark
    from 因子回测.alpha import add_future_return

    # 空基准列表处理
    if not benchmarks:
        return {"summary": pd.DataFrame(), "daily": {}}

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

def plot_multi_benchmark_summary(
    multi_results: dict,
    factor_columns: List[str],
    factor_labels: Dict[str, str],
    horizons: Tuple[int, ...] = (1, 3, 5, 10),
    benchmarks: List[str] = None,
    benchmark_labels: Dict[str, str] = None,
) -> None:
    """
    绘制多基准择时对比。

    1. 每个 (因子, 持有期) 画一张多折线净值对比图
    2. 一张热力图：row=基准, column=因子(按持有期分面)，颜色=年化超额收益

    Parameters
    ----------
    multi_results : dict
        run_multi_benchmark_timing 返回的结果字典，含 daily 和 summary。
    factor_columns : list[str]
        因子列名列表。
    factor_labels : dict[str, str]
        因子显示标签映射。
    horizons : tuple[int, ...]
        持有周期列表。
    benchmarks : list[str], optional
        基准名称列表。用于确定热力图的显示顺序。
        若为 None，从 daily 键中推断。
    benchmark_labels : dict[str, str], optional
        基准显示标签映射。若为 None，使用基准名称本身作为显示标签。
    """
    daily = multi_results["daily"]
    summary = multi_results["summary"]

    # 从 daily 键中推断 benchmarks（若未提供）
    if benchmarks is None:
        benchmarks = sorted(set(k[0] for k in daily.keys()))
    if benchmark_labels is None:
        benchmark_labels = {b: b for b in benchmarks} if benchmarks else {}

    colors = ["#C44E52", "#1f77b4", "#2ca02c", "#9467bd"]

    # ===== 1. 每个 (因子, 持有期) 的净值对比图 =====
    for factor in factor_columns:
        n_horizons_plot = len(horizons)
        n_cols = min(2, n_horizons_plot)
        n_rows = (n_horizons_plot + 1) // 2  # ceiling division
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True
        )
        # 统一展平处理，兼容 1x1 场景
        if n_rows == 1 and n_cols == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.ravel()

        for ax_idx, horizon in enumerate(horizons):
            if ax_idx >= len(axes_flat):
                continue
            ax = axes_flat[ax_idx]
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

        # 隐藏多余的空白子图
        for ax_idx in range(len(horizons), len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        fig.suptitle(
            f"{factor_labels.get(factor, factor)}：多基准择时净值对比",
            fontsize=15,
        )
        plt.show()

    # ===== 2. 热力图：基准 x 因子（按持有期分面） =====
    if len(summary) == 0:
        print("无可用汇总数据，跳过热力图")
        return

    n_horizons = len(horizons)
    heatmap_height = max(5, 0.5 * len(benchmarks)) if benchmarks else 5
    fig, axes = plt.subplots(
        1, n_horizons,
        figsize=(5 * n_horizons, heatmap_height),
        constrained_layout=True,
    )
    if n_horizons == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        sub = summary[summary["horizon"] == horizon]
        if len(sub) == 0:
            ax.set_title(f"{horizon}日持有期 | 无数据")
            ax.grid(alpha=0.25)
            continue

        pivot = sub.pivot(
            index="benchmark", columns="factor", values="annual_excess_return"
        )
        if benchmarks:
            pivot = pivot.reindex(
                index=[b for b in benchmarks if b in pivot.index]
            )
        if factor_columns:
            pivot = pivot.reindex(columns=factor_columns)

        vals = pivot.to_numpy(dtype=float)
        if vals.size > 0:
            max_abs = float(np.nanmax(np.abs(vals)))
            limit = max(0.01, max_abs) if not np.isnan(max_abs) else 0.05
        else:
            limit = 0.05

        im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
        ax.set_xticks(
            range(len(pivot.columns)),
            [factor_labels.get(c, c) for c in pivot.columns],
            fontsize=8,
        )
        ax.set_yticks(
            range(len(pivot.index)),
            [benchmark_labels.get(b, b) for b in pivot.index],
        )
        ax.set_title(f"{horizon}日持有期 | 年化超额收益")
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.show()
