"""
高人气补涨策略 v4：Tushare 数据源与掘金数据源对比回测。

脚本目的：
    1. 保留原 Notebook 的 Tushare 口径，作为 baseline。
    2. 新增 gm_stock_all_data 口径，并把字段转换成策略原先使用的字段名和单位。
    3. 同时跑两套信号与回测，输出信号差异、交易结果和摘要报表。

运行示例：
    E:\\working\\anaconda3\\envs\\quant\\python.exe hot_rebound_data_source_backtest.py

注意：
    本脚本只做数据源对比，不修改原 Notebook。
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from my_utils.fun import (
    add_pre_n_ratio,
    add_sma,
    cal_limit_avg_turnover,
    get_logger,
    mark_last_limit_desc,
    mark_limit_desc,
    mark_limit_status,
    read_day_data,
    read_min_data,
)
from my_utils.trade_fun import cal_trade_info


PARAMS_DICT: dict[str, Any] = {
    "popularity_window": 15,
    "hot_lookback": 7,
    "pullback_window": 20,
    "min_limit_up_count": 2,
    "max_yzb_count": 2,
    "popularity_score_threshold": 0.72,
    "amount_rank_threshold": 0.90,
    "turnover_rank_threshold": 0.88,
    "pullback_low": -12,
    "pullback_high": -4,
    "open_pct_low": -3.5,
    "open_pct_high": 2.5,
    "mv_min": 30,
    "mv_max": 800,
    "avg_limit_turnover_min": 3,
    "avg_limit_turnover_max": 35,
    "min_turn_over": 2,
    "min_volume_ratio_5": 0.8,
    "take_profit_pct": 0.28,
    "stop_loss_pct": 0.08,
    "max_holding_days": 3,
    "fee_rate": 0.004,
    "position_weight": 0.45,
    "opt_pullback_high": -5,
    "opt_avg_limit_turnover_min": 6.3,
    "opt_avg_limit_turnover_max": 8.5,
}


def rank_pct_desc(expr: pl.Expr, by: str) -> pl.Expr:
    """按交易日做降序百分位排名；数值越大，返回值越接近 1。"""
    group_size = pl.len().over(by)
    rank_desc = expr.rank(descending=True, method="average").over(by)
    return (
        pl.when(group_size <= 1)
        .then(0.0)
        .otherwise((group_size - rank_desc) / (group_size - 1))
        .cast(pl.Float64)
    )


def calc_days_since_hot(group: pl.DataFrame) -> pl.DataFrame:
    """计算每只股票距离最近一次热股日过去了多少个交易日。"""
    hot_list = group["is_hot_today"].to_list()
    result = []
    last_hot_idx = None

    for idx, is_hot in enumerate(hot_list):
        if is_hot:
            last_hot_idx = idx
            result.append(0)
        elif last_hot_idx is None:
            result.append(None)
        else:
            result.append(idx - last_hot_idx)

    return group.with_columns(pl.Series("days_since_hot", result, dtype=pl.Int64))


def normalize_gm_stock_data(stock_data: pl.DataFrame) -> pl.DataFrame:
    """把掘金日线字段转换成高人气策略原先使用的字段口径。

    关键口径：
        - gm volume 是“股”，原 Tushare volume 近似是“手”，所以除以 100。
        - gm amount 是“元”，原 Tushare amount 近似是“万元”，所以除以 10000。
        - gm mv_A_free_float/total_mv 是“元”，策略阈值使用“亿”，所以除以 1e8。
        - gm turnover_rate 对应原策略的 turn_over。
        - gm is_st 映射成 type='ST'，以复用原有 ST 过滤条件。
        - gm 新股/特殊交易日可能给 limit_up/limit_down=0，这里转 null，后续统一 drop。
    """
    extra_columns = []
    if "industry" not in stock_data.columns:
        extra_columns.append(pl.lit(None, dtype=pl.String).alias("industry"))
    if "area" not in stock_data.columns:
        extra_columns.append(pl.lit(None, dtype=pl.String).alias("area"))

    return stock_data.with_columns(
        [
            (pl.col("volume") / 100).alias("volume"),
            (pl.col("amount") / 10000).alias("amount"),
            pl.col("turnover_rate").alias("turn_over"),
            (pl.col("mv_A_free_float") / 1e8).alias("free_float_mv"),
            (pl.col("mv_A_free_float") / 1e8).alias("float_mv"),
            (pl.col("total_mv") / 1e8).alias("total_mv"),
            pl.when(pl.col("is_st")).then(pl.lit("ST")).otherwise(None).alias("type"),
            pl.when(pl.col("limit_up") > 0).then(pl.col("limit_up")).otherwise(None).alias("limit_up"),
            pl.when(pl.col("limit_down") > 0).then(pl.col("limit_down")).otherwise(None).alias("limit_down"),
            *extra_columns,
        ]
    )


def load_ts_strategy_data(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    """读取原高人气策略使用的 Tushare 口径数据。"""
    stock_data = read_day_data(start_date=start_date, end_date=end_date, file_path="ts_stock_all_data")
    stock_data = stock_data.drop_nulls(subset=["open", "high", "low", "close", "pre_close", "limit_up", "limit_down"])

    market_value = read_day_data(start_date=start_date, end_date=end_date, file_path="ts_daily_basic")
    market_value = market_value.with_columns(
        # 与原 Notebook 保持一致：自由流通股本 * 收盘价 / 1e4，得到“亿”级市值。
        (pl.col("free_share") * pl.col("close") / 1e4).alias("free_float_mv")
    )
    stock_data = stock_data.join(
        market_value.select(["code", "trading_date", "free_float_mv", "turnover_rate"]),
        on=["code", "trading_date"],
        how="left",
    )

    drop_cols = [
        "change",
        "total_share",
        "attack",
        "activity",
        "pe",
        "float_share",
        "buying",
        "selling",
        "swing",
        "strength",
        "avg_turnover",
    ]
    drop_cols = [col for col in drop_cols if col in stock_data.columns]
    if drop_cols:
        stock_data = stock_data.drop(drop_cols)
    return stock_data.sort(["code", "trading_date"])


def load_gm_strategy_data(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    """读取并标准化掘金口径数据，使其能进入同一套高人气策略逻辑。"""
    stock_data = read_day_data(start_date=start_date, end_date=end_date, file_path="gm_stock_all_data")
    stock_data = normalize_gm_stock_data(stock_data)
    stock_data = stock_data.drop_nulls(subset=["open", "high", "low", "close", "pre_close", "limit_up", "limit_down"])
    return stock_data.sort(["code", "trading_date"])


def build_hot_rebound_signals(stock_data: pl.DataFrame, params_dict: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """按高人气补涨 v4 逻辑生成基础信号和优化后信号。"""
    stock_data = mark_limit_status(stock_data)
    stock_data = mark_limit_desc(stock_data)
    stock_data = mark_last_limit_desc(stock_data)
    stock_data = cal_limit_avg_turnover(stock_data, window=10)
    stock_data = add_sma(stock_data, window=5)
    stock_data = add_sma(stock_data, window=10)
    stock_data = add_sma(stock_data, window=20)
    stock_data = add_pre_n_ratio(stock_data, field="volume", n=5)

    stock_data = stock_data.with_columns(
        [
            ((pl.col("open") - pl.col("pre_close")) / pl.col("pre_close") * 100).alias("open_pct"),
            ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).alias("body_pct"),
            ((pl.col("amount") * 100 / pl.col("volume")).fill_nan(0.0).fill_null(0.0)).alias("vwap"),
            ((pl.col("open") >= pl.col("limit_up") * 0.999) & (pl.col("close") >= pl.col("limit_up") * 0.999)).alias(
                "is_yzb"
            ),
        ]
    )

    popularity_window = params_dict["popularity_window"]
    pullback_window = params_dict["pullback_window"]
    stock_data = stock_data.with_columns(
        [
            pl.col("is_limit_up")
            .cast(pl.Int64)
            .rolling_sum(window_size=popularity_window, min_samples=1)
            .over("code")
            .shift(1)
            .alias(f"limit_up_count_{popularity_window}"),
            pl.col("is_yzb")
            .cast(pl.Int64)
            .rolling_sum(window_size=popularity_window, min_samples=1)
            .over("code")
            .shift(1)
            .alias(f"yzb_count_{popularity_window}"),
            pl.col("high")
            .rolling_max(window_size=pullback_window, min_samples=1)
            .over("code")
            .shift(1)
            .alias(f"recent_high_{pullback_window}"),
        ]
    )

    stock_data = stock_data.with_columns(
        [
            (
                (pl.col("close") - pl.col(f"recent_high_{pullback_window}"))
                / pl.col(f"recent_high_{pullback_window}")
                * 100
            )
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("pullback_pct"),
            rank_pct_desc(pl.col("amount"), "trading_date").alias("amount_rank_pct"),
            rank_pct_desc(pl.col("turn_over"), "trading_date").alias("turnover_rank_pct"),
        ]
    )

    stock_data = stock_data.with_columns(
        [
            (pl.col("volume_ratio_5").clip(0.0, 3.0) / 3.0).alias("volume_ratio_score"),
            (pl.col(f"limit_up_count_{popularity_window}").clip(0.0, 4.0) / 4.0).alias("board_score"),
        ]
    )
    stock_data = stock_data.with_columns(
        (
            pl.col("amount_rank_pct") * 0.35
            + pl.col("turnover_rank_pct") * 0.25
            + pl.col("volume_ratio_score") * 0.20
            + pl.col("board_score") * 0.20
        ).alias("popularity_score")
    )
    stock_data = stock_data.with_columns(
        (
            (pl.col("popularity_score") >= params_dict["popularity_score_threshold"])
            | (
                (pl.col("amount_rank_pct") >= params_dict["amount_rank_threshold"])
                & (pl.col("turnover_rank_pct") >= params_dict["turnover_rank_threshold"])
                & (pl.col(f"limit_up_count_{popularity_window}") >= params_dict["min_limit_up_count"])
            )
        ).alias("is_hot_today")
    )

    stock_data = stock_data.group_by("code").map_groups(calc_days_since_hot)
    stock_data = stock_data.with_columns(pl.col("trading_date").shift(-1).over("code").alias("next_trading_date"))

    code_part = pl.col("code").str.split(".").list[1]
    non_st_filter = ~(pl.col("type").is_not_null() & (pl.col("type") == "ST"))
    non_gem_kc_filter = ~(code_part.str.starts_with("30") | code_part.str.starts_with("688") | code_part.str.starts_with("8"))
    signal_condition = (
        non_st_filter
        & non_gem_kc_filter
        & pl.col("next_trading_date").is_not_null()
        & (pl.col("days_since_hot") >= 1)
        & (pl.col("days_since_hot") <= params_dict["hot_lookback"])
        & (pl.col(f"limit_up_count_{params_dict['popularity_window']}") >= params_dict["min_limit_up_count"])
        & (pl.col(f"yzb_count_{params_dict['popularity_window']}") <= params_dict["max_yzb_count"])
        & (pl.col("pullback_pct") >= params_dict["pullback_low"])
        & (pl.col("pullback_pct") <= params_dict["pullback_high"])
        & (pl.col("free_float_mv") >= params_dict["mv_min"])
        & (pl.col("free_float_mv") <= params_dict["mv_max"])
        & (pl.col("turn_over") >= params_dict["min_turn_over"])
        & (pl.col("avg_limit_turnover_10") >= params_dict["avg_limit_turnover_min"])
        & (pl.col("avg_limit_turnover_10") <= params_dict["avg_limit_turnover_max"])
        & (pl.col("volume_ratio_5") >= params_dict["min_volume_ratio_5"])
        & (pl.col("open_pct") >= params_dict["open_pct_low"])
        & (pl.col("open_pct") <= params_dict["open_pct_high"])
        & (pl.col("body_pct") > 0)
        & (pl.col("close") > pl.col("sma_5"))
        & (~pl.col("is_limit_up"))
        & (pl.col("close") > pl.col("limit_down") * 1.01)
    )

    stock_data = stock_data.with_columns(pl.when(signal_condition).then(1).otherwise(0).alias("signal"))
    base_signals_df = (
        stock_data.filter(pl.col("signal") == 1)
        .select(
            [
                "code",
                "trading_date",
                "next_trading_date",
                "name",
                "industry",
                "close",
                "open_pct",
                "body_pct",
                "pullback_pct",
                "free_float_mv",
                "turn_over",
                "amount_rank_pct",
                "turnover_rank_pct",
                "volume_ratio_5",
                "avg_limit_turnover_10",
                "popularity_score",
                "days_since_hot",
                f"limit_up_count_{params_dict['popularity_window']}",
                f"yzb_count_{params_dict['popularity_window']}",
                "last_limit_desc",
                "signal",
            ]
        )
        .rename({"trading_date": "signal_date", "next_trading_date": "trading_date"})
        .sort(["trading_date", "code"])
    )
    signals_df = base_signals_df.filter(
        (pl.col("pullback_pct") <= params_dict["opt_pullback_high"])
        & (pl.col("avg_limit_turnover_10") >= params_dict["opt_avg_limit_turnover_min"])
        & (pl.col("avg_limit_turnover_10") <= params_dict["opt_avg_limit_turnover_max"])
    )
    return base_signals_df, signals_df


def buy_hot_rebound_trade(
    code_list: list[str],
    trade_date: dt.date,
    fee_rate: float = 0.004,
    take_profit_pct: float = 0.28,
    stop_loss_pct: float = 0.08,
    max_holding_days: int = 3,
    day_data_file_path: str = "ts_stock_all_data",
) -> list[dict[str, Any]] | None:
    """模拟单个交易日、单组股票的分钟级买卖过程。"""
    start_date = trade_date
    end_date = start_date + dt.timedelta(days=max_holding_days + 10)

    try:
        day_data = read_day_data(start_date, end_date, code_list, file_path=day_data_file_path)
        if day_data_file_path == "gm_stock_all_data":
            day_data = day_data.with_columns(
                [
                    pl.when(pl.col("limit_up") > 0).then(pl.col("limit_up")).otherwise(None).alias("limit_up"),
                    pl.when(pl.col("limit_down") > 0).then(pl.col("limit_down")).otherwise(None).alias("limit_down"),
                ]
            )
        mins_data = read_min_data(start_date, end_date, code_list)
        mins_data = mins_data.join(
            day_data.select(["code", "trading_date", "limit_up", "limit_down"]),
            on=["code", "trading_date"],
            how="left",
        )
        mins_data = mins_data.drop_nulls(subset=["open", "high", "low", "close", "limit_up", "limit_down"])
    except Exception:
        return None

    result = []
    for code in code_list:
        code_mins = mins_data.filter(pl.col("code") == code).sort("datetime")
        if code_mins.height == 0:
            continue

        trade_info = {
            "code": code,
            "buy_time": None,
            "buy_price": None,
            "sell_time": None,
            "sell_price": None,
            "profit": None,
            "holding_days": None,
            "sell_reason": None,
        }
        buy_bar = code_mins.filter(
            (pl.col("trading_date") == trade_date)
            & (pl.col("datetime").dt.hour() == 9)
            & (pl.col("datetime").dt.minute() == 30)
        )
        if buy_bar.height == 0:
            continue

        buy_price = buy_bar["open"].to_list()[0]
        trade_info["buy_time"] = dt.datetime.combine(trade_date, dt.time(9, 30))
        trade_info["buy_price"] = float(buy_price)
        take_profit_price = buy_price * (1 + take_profit_pct)
        stop_loss_price = buy_price * (1 - stop_loss_pct)

        trading_days = sorted(code_mins["trading_date"].unique().to_list())
        if trade_date not in trading_days:
            continue

        hold_days = trading_days[trading_days.index(trade_date) : trading_days.index(trade_date) + max_holding_days]
        sold = False
        for offset, single_date in enumerate(hold_days):
            day_mins = code_mins.filter(pl.col("trading_date") == single_date).sort("datetime")
            if day_mins.height == 0:
                continue

            for row in day_mins.iter_rows(named=True):
                if row["low"] <= stop_loss_price:
                    trade_info["sell_time"] = row["datetime"]
                    trade_info["sell_price"] = float(stop_loss_price)
                    trade_info["sell_reason"] = "stop_loss_sell"
                    sold = True
                    break
                if row["high"] >= take_profit_price:
                    trade_info["sell_time"] = row["datetime"]
                    trade_info["sell_price"] = float(take_profit_price)
                    trade_info["sell_reason"] = "take_profit_sell"
                    sold = True
                    break
            if sold:
                break

            if offset == len(hold_days) - 1:
                exit_bar = day_mins.filter((pl.col("datetime").dt.hour() == 14) & (pl.col("datetime").dt.minute() >= 55))
                if exit_bar.height == 0:
                    exit_bar = day_mins.tail(1)
                row = exit_bar.tail(1).to_dicts()[0]
                trade_info["sell_time"] = row["datetime"]
                trade_info["sell_price"] = float(row["close"])
                trade_info["sell_reason"] = "holding_period_exit"
                sold = True
                break

        if not sold:
            last_row = code_mins.tail(1)
            trade_info["sell_time"] = last_row["datetime"].to_list()[0]
            trade_info["sell_price"] = float(last_row["close"].to_list()[0])
            trade_info["sell_reason"] = "data_end_sell"

        trade_info["holding_days"] = (trade_info["sell_time"].date() - trade_info["buy_time"].date()).days + 1
        trade_info["profit"] = ((trade_info["sell_price"] / trade_info["buy_price"] - 1) - fee_rate) * 100
        result.append(trade_info)

    return result


def _to_pandas(df: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy()


def summarize_backtest(source: str, base_signals: pl.DataFrame, signals: pl.DataFrame, merged_df: pl.DataFrame | pd.DataFrame) -> dict[str, Any]:
    """生成信号与交易收益摘要。"""
    merged_pd = _to_pandas(merged_df)
    valid = merged_pd[merged_pd["profit"].notna()].copy() if "profit" in merged_pd.columns else pd.DataFrame()
    if valid.empty:
        return {
            "source": source,
            "base_signal_count": base_signals.height,
            "optimized_signal_count": signals.height,
            "trade_count": 0,
            "avg_profit": None,
            "median_profit": None,
            "win_rate": None,
            "total_weight_profit": None,
        }
    valid["weight_profit"] = valid["profit"] * PARAMS_DICT["position_weight"]
    return {
        "source": source,
        "base_signal_count": base_signals.height,
        "optimized_signal_count": signals.height,
        "trade_count": int(valid.shape[0]),
        "avg_profit": float(valid["profit"].mean()),
        "median_profit": float(valid["profit"].median()),
        "win_rate": float((valid["profit"] > 0).mean()),
        "total_weight_profit": float(valid["weight_profit"].sum()),
    }


def run_source_backtest(source: str, start_date: dt.date, end_date: dt.date, max_workers: int | None) -> dict[str, Any]:
    """运行单个数据源的信号生成与回测。"""
    if source == "ts":
        stock_data = load_ts_strategy_data(start_date, end_date)
        day_data_file_path = "ts_stock_all_data"
    elif source == "gm":
        stock_data = load_gm_strategy_data(start_date, end_date)
        day_data_file_path = "gm_stock_all_data"
    else:
        raise ValueError(f"unsupported source: {source}")

    base_signals, signals = build_hot_rebound_signals(stock_data, PARAMS_DICT)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    result_df, merged_df = cal_trade_info(
        signals,
        trade_fun=buy_hot_rebound_trade,
        trade_kwargs={
            "fee_rate": PARAMS_DICT["fee_rate"],
            "take_profit_pct": PARAMS_DICT["take_profit_pct"],
            "stop_loss_pct": PARAMS_DICT["stop_loss_pct"],
            "max_holding_days": PARAMS_DICT["max_holding_days"],
            "day_data_file_path": day_data_file_path,
        },
        start_date=start_date_str,
        end_date=end_date_str,
        max_workers=max_workers,
    )
    return {
        "source": source,
        "base_signals": base_signals,
        "signals": signals,
        "result_df": result_df,
        "merged_df": merged_df,
        "summary": summarize_backtest(source, base_signals, signals, merged_df),
    }


def build_signal_diff(ts_signals: pl.DataFrame, gm_signals: pl.DataFrame) -> pl.DataFrame:
    """对比两套数据源最终优化信号的重合和差异。"""
    key_cols = ["code", "trading_date"]
    ts_keys = ts_signals.select(key_cols).with_columns(pl.lit(True).alias("in_ts"))
    gm_keys = gm_signals.select(key_cols).with_columns(pl.lit(True).alias("in_gm"))
    return (
        ts_keys.join(gm_keys, on=key_cols, how="full", coalesce=True)
        .with_columns(
            [
                pl.col("in_ts").fill_null(False),
                pl.col("in_gm").fill_null(False),
            ]
        )
        .with_columns(
            pl.when(pl.col("in_ts") & pl.col("in_gm"))
            .then(pl.lit("both"))
            .when(pl.col("in_ts"))
            .then(pl.lit("ts_only"))
            .otherwise(pl.lit("gm_only"))
            .alias("signal_source")
        )
        .sort(["trading_date", "code"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="高人气补涨策略 v4 数据源对比回测")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default=dt.date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", default="signals_v2/hot_rebound_v4_data_source_backtest")
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging = get_logger(log_file="log/hot_rebound_data_source_backtest.log", inherit=False)
    logging.info(f"run hot rebound data source compare: {start_date} to {end_date}")

    results = [run_source_backtest(source, start_date, end_date, args.max_workers) for source in ["ts", "gm"]]
    summary_df = pd.DataFrame([item["summary"] for item in results])
    signal_diff = build_signal_diff(results[0]["signals"], results[1]["signals"])
    signal_diff_summary = signal_diff.group_by("signal_source").len().sort("signal_source")

    for item in results:
        source = item["source"]
        item["base_signals"].write_csv(output_dir / f"{source}_base_signals.csv", include_bom=True)
        item["signals"].write_csv(output_dir / f"{source}_optimized_signals.csv", include_bom=True)
        _to_pandas(item["merged_df"]).to_csv(output_dir / f"{source}_merged_backtest.csv", index=False, encoding="utf-8-sig")

    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    signal_diff.write_csv(output_dir / "signal_diff.csv", include_bom=True)
    signal_diff_summary.write_csv(output_dir / "signal_diff_summary.csv", include_bom=True)

    print("\n=== 数据源回测摘要 ===")
    print(summary_df.to_string(index=False))
    print("\n=== 信号交集差异 ===")
    print(signal_diff_summary.to_pandas().to_string(index=False))
    print(f"\n输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
