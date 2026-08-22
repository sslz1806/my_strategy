"""五日涨跌停情绪因子的公共研究函数。"""

import numpy as np
import pandas as pd
import polars as pl

THRESHOLD_QUANTILE = 0.60
MIN_HISTORY = 252
HORIZONS = (1, 3, 5, 10)
ROLLING_IC_WINDOWS = (50, 100)
ROLLING_IC_MIN_VALID_RATIO = 0.95
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
        # 分母为 0 时返回 None（防浮点噪声导致 inf）
        limit_up_next_ret=pl.when(
            pl.col("n_up").shift(1).rolling_sum(window, min_periods=window) > 0
        ).then(
            pl.col("up_ret").shift(1).rolling_sum(window, min_periods=window)
            / pl.col("n_up").shift(1).rolling_sum(window, min_periods=window)
        ).otherwise(None),
        limit_down_next_ret=pl.when(
            pl.col("n_down").shift(1).rolling_sum(window, min_periods=window) > 0
        ).then(
            pl.col("down_ret").shift(1).rolling_sum(window, min_periods=window)
            / pl.col("n_down").shift(1).rolling_sum(window, min_periods=window)
        ).otherwise(None),
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


def run_time_backtest(
    data: pd.DataFrame,
    signal_column: str,
    ret_column: str,
    horizon: int,
    date_column: str = "trading_date",
    anchor_date: pd.Timestamp = None,
    require_complete_exit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对单个指数收益序列运行信号驱动的连续持仓择时回测。

    信号触发即开仓/续期，持有 horizon 个交易日。
    预热期信号为 NaN 时不进入样本，不改变当前仓位状态。
    空仓日仓位为 0，但每日仍保留一行收益记录，保证净值曲线连续。

    采用两遍遍历：先逐日生成仓位序列，再从仓位序列提取连续持仓/空仓段。

    Parameters
    ----------
    data : pd.DataFrame
        已完成外部清洗的日频表，包含日期、信号和单个指数收益列。
    signal_column : str
        二值信号列名，1=开仓或续期，0=不触发，NaN=预热期无效值。
    ret_column : str
        指数日收益列名，收益使用小数形式（0.01 = 1%）。
    horizon : int
        持有周期（交易日数），同时也是续期长度。
    date_column : str
        日期列名，默认 trading_date。
    anchor_date : pd.Timestamp, optional
        回测起始日。若为 None，从 signal_column 首个非空值所在行起算。
    require_complete_exit : bool
        若为 True，尾部不完整的持仓段不进入 blocks 统计。

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (daily, blocks)
        - daily: 逐日明细，含日期、position、ret_column、
                 strategy_daily_ret, strategy_nav, benchmark_nav
        - blocks: 连续持仓/空仓段明细，每行一个完整的持仓段或空仓段。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon 必须是正整数")

    required_columns = {date_column, signal_column, ret_column}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"data 缺少必要字段：{missing_columns}")

    ordered = data.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column], errors="coerce")
    if ordered[date_column].isna().any():
        raise ValueError(f"{date_column} 包含无法解析的日期")
    if ordered[date_column].duplicated().any():
        raise ValueError(f"{date_column} 存在重复日期")

    try:
        ordered[ret_column] = pd.to_numeric(ordered[ret_column], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ret_column} 必须是数值型日收益") from exc
    if not np.isfinite(ordered[ret_column].to_numpy()).all():
        raise ValueError(f"{ret_column} 必须全部是有限数值，不允许 NaN 或 inf")

    valid_signals = ordered[signal_column].dropna()
    if not valid_signals.isin([0, 1]).all():
        raise ValueError(f"{signal_column} 只能包含 0、1 或 NaN")

    ordered = ordered.sort_values(date_column).reset_index(drop=True)
    daily_columns = [
        date_column,
        "position",
        ret_column,
        "strategy_daily_ret",
        "benchmark_nav",
        "strategy_nav",
    ]
    block_columns = [
        "block_id",
        "position",
        "decision_date",
        "block_start_date",
        "block_end_date",
        "block_duration",
        "benchmark_block_return",
        "strategy_block_return",
    ]

    # 确定锚点行索引
    if anchor_date is not None:
        anchor_timestamp = pd.to_datetime(anchor_date, errors="coerce")
        if pd.isna(anchor_timestamp):
            raise ValueError("anchor_date 无法解析为日期")
        anchor_matches = ordered.index[ordered[date_column] == anchor_timestamp]
        if len(anchor_matches) == 0:
            raise ValueError("anchor_date 不在 data 的日期列中")
        anchor_index = int(anchor_matches[0])
    else:
        first_valid = ordered[signal_column].first_valid_index()
        if first_valid is None:
            return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=block_columns)
        anchor_index = int(first_valid)

    n = len(ordered)
    if anchor_index >= n - 1:
        return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=block_columns)

    # ===== 第一遍：生成逐日仓位序列 =====
    # positions[i] 对应第 anchor_index + i 个决策日的次日仓位
    positions = np.zeros(n - 1 - anchor_index)
    holding_until_idx = -1
    last_signal_idx = -1

    for offset, t_idx in enumerate(range(anchor_index, n - 1)):
        signal = ordered.iloc[t_idx][signal_column]
        # 预热期 NaN 不触发，也不改变当前仓位
        if not pd.isna(signal) and signal == 1:
            new_until = t_idx + horizon
            if new_until > holding_until_idx:
                holding_until_idx = new_until
            last_signal_idx = t_idx

        positions[offset] = 1.0 if t_idx + 1 <= holding_until_idx else 0.0

    # ===== 第二遍：构建逐日明细 =====
    daily_rows = []
    for offset, t_idx in enumerate(range(anchor_index, n - 1)):
        daily_rows.append({
            date_column: ordered.iloc[t_idx + 1][date_column],
            "position": positions[offset],
            ret_column: ordered.iloc[t_idx + 1][ret_column],
            "strategy_daily_ret": positions[offset] * ordered.iloc[t_idx + 1][ret_column],
        })
    daily = pd.DataFrame(
        daily_rows,
        columns=[date_column, "position", ret_column, "strategy_daily_ret"],
    )

    # ===== 第三遍：从仓位序列提取连续持仓/空仓段 =====
    block_rows = []
    seg_start = 0
    seg_position = positions[0]

    def append_segment(seg_end: int, is_last: bool) -> None:
        """把 positions 的闭区间 [seg_start, seg_end] 转成一条区段记录。"""
        if (
            is_last
            and seg_position > 0.5
            and require_complete_exit
            and last_signal_idx + horizon >= n
        ):
            return

        decision_idx = anchor_index + seg_start
        return_start_idx = decision_idx + 1
        return_end_idx = anchor_index + seg_end + 1
        block_slice = ordered.iloc[return_start_idx : return_end_idx + 1]
        benchmark_return = (1 + block_slice[ret_column]).prod() - 1
        block_rows.append(
            {
                "block_id": decision_idx,
                "position": seg_position,
                "decision_date": ordered.iloc[decision_idx][date_column],
                "block_start_date": ordered.iloc[return_start_idx][date_column],
                "block_end_date": ordered.iloc[return_end_idx][date_column],
                "block_duration": len(block_slice),
                "benchmark_block_return": benchmark_return,
                "strategy_block_return": seg_position * benchmark_return,
            }
        )

    for i in range(1, len(positions)):
        if positions[i] != seg_position:
            append_segment(i - 1, is_last=False)
            seg_start = i
            seg_position = positions[i]

    # 最后一个段
    append_segment(len(positions) - 1, is_last=True)

    blocks = pd.DataFrame(block_rows, columns=block_columns)

    daily["benchmark_nav"] = (1 + daily[ret_column]).cumprod()
    daily["strategy_nav"] = (1 + daily["strategy_daily_ret"]).cumprod()

    return daily, blocks


def run_timing(
    data: pd.DataFrame,
    signal_column: str,
    horizon: int,
    anchor_date: pd.Timestamp = None,
    require_complete_exit: bool = True,
    ret_column: str = "market_daily_ret",
    date_column: str = "trading_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """兼容旧调用的别名；实际逻辑统一由 run_time_backtest 实现。"""
    return run_time_backtest(
        data=data,
        signal_column=signal_column,
        ret_column=ret_column,
        horizon=horizon,
        date_column=date_column,
        anchor_date=anchor_date,
        require_complete_exit=require_complete_exit,
    )


def summarize_timing(
    daily: pd.DataFrame,
    blocks: pd.DataFrame,
    factor: str,
    horizon: int,
    ret_column: str = "market_daily_ret",
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

    strategy_returns = pd.Series(daily["strategy_daily_ret"], dtype=float).dropna()
    benchmark_returns = pd.Series(daily[ret_column], dtype=float).dropna()

    if strategy_returns.empty:
        annual_return = max_drawdown = sharpe = np.nan
    else:
        strategy_nav = (1 + strategy_returns).cumprod()
        strategy_volatility = strategy_returns.std(ddof=1)
        annual_return = strategy_nav.iloc[-1] ** (252 / len(strategy_returns)) - 1
        max_drawdown = (strategy_nav / strategy_nav.cummax() - 1).min()
        sharpe = (
            strategy_returns.mean() / strategy_volatility * np.sqrt(252)
            if strategy_volatility > 0
            else np.nan
        )

    if benchmark_returns.empty:
        benchmark_annual_return = np.nan
    else:
        benchmark_nav = (1 + benchmark_returns).cumprod()
        benchmark_annual_return = (
            benchmark_nav.iloc[-1] ** (252 / len(benchmark_returns)) - 1
        )

    return {
        "factor": factor,
        "factor_label": FACTOR_LABELS.get(factor, factor),
        "horizon": horizon,
        "rebalance_count": len(blocks.loc[blocks["position"] > 0.5]) if len(blocks) > 0 else 0,
        "holding_ratio": daily["position"].mean() if len(daily) > 0 else np.nan,
        "holding_win_rate": holding_win_rate,
        "timing_hit_rate": timing_hit_rate,
        "annual_return": annual_return,
        "benchmark_annual_return": benchmark_annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "final_nav": daily["strategy_nav"].iloc[-1] if len(daily) > 0 else np.nan,
        "benchmark_final_nav": daily["benchmark_nav"].iloc[-1] if len(daily) > 0 else np.nan,
        "relative_final_nav": (
            daily["strategy_nav"].iloc[-1] / daily["benchmark_nav"].iloc[-1] - 1
        ) if len(daily) > 0 else np.nan,
    }


def analyze_ic(
    data: pd.DataFrame,
    factor_columns: list[str] = FACTOR_COLUMNS,
    horizons: tuple[int, ...] = HORIZONS,
    factor_directions: dict[str, int] = FACTOR_DIRECTIONS,
    ret_column: str = "market_daily_ret",
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
        含因子列和 future_{ret_column}_{n}d 列的研究数据。
    factor_columns : list[str]
        因子列名列表。
    horizons : tuple[int, ...]
        未来收益周期列表。
    factor_directions : dict[str, int]
        因子方向字典：+1=正向，-1=反向。
    ret_column : str
        add_future_return 使用的原始日收益列名；目标列按
        future_{ret_column}_{horizon}d 推导。

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
            target = f"future_{ret_column}_{horizon}d"
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


def analyze_rolling_ic(
    data: pd.DataFrame,
    factor_columns: list[str] = FACTOR_COLUMNS,
    horizons: tuple[int, ...] = HORIZONS,
    windows: tuple[int, ...] = ROLLING_IC_WINDOWS,
    min_valid_ratio: float = ROLLING_IC_MIN_VALID_RATIO,
    ret_column: str = "market_daily_ret",
    date_column: str = "trading_date",
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
        含因子列和 future_{ret_column}_{n}d 列的研究数据。
    factor_columns : list[str]
        因子列名列表。
    horizons : tuple[int, ...]
        未来收益周期列表。
    windows : tuple[int, ...]
        滚动窗口列表，每个窗口独立输出一条 IC 序列。
    min_valid_ratio : float
        窗口内最少有效配对比例（默认 0.95 = 95%）。
    ret_column : str
        生成未来收益时使用的原始收益列名。
    date_column : str
        因子观测日期列名。

    Returns
    -------
    pd.DataFrame
        长格式 DataFrame，每行对应一个(因子, 周期, 窗口, 交易日)的滚动 IC 值。
    """
    # 参数校验
    if not 0 < min_valid_ratio <= 1:
        raise ValueError("min_valid_ratio 必须在 (0, 1] 范围内")

    ordered = data.copy().sort_values(date_column).reset_index(drop=True)
    rows = []
    
    # 遍历因子 × 周期 × 窗口，生成每条 IC 序列
    for factor in factor_columns:
        for horizon in horizons:
            target = f"future_{ret_column}_{horizon}d"
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
                            "factor_window_end_date": ordered[date_column],
                            "available_date": ordered[date_column].shift(-horizon),
                            "rolling_n_obs": rolling_n_obs,
                            "min_required_obs": min_required_obs,
                            "rolling_ic": rolling_ic,
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)
