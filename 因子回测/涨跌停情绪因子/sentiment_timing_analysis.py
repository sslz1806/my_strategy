"""涨跌停情绪择时的可复用、可测试计算函数。

本模块只处理确定性的日线因子、市场前瞻收益和时序有效性统计；15 分钟
打板成交近似仍留在研究 Notebook 中，以免数据读取和统计逻辑耦合在一起。
所有收益率都使用小数表示，例如 1% 表示为 ``0.01``。
"""

from __future__ import annotations

from math import ceil
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm


_A_SHARE_PREFIXES = ("6", "0", "3")
_BASIC_REQUIRED_COLUMNS = {
    "code",
    "trading_date",
    "close",
    "pre_close",
    "pct",
    "limit_up",
    "limit_down",
    "is_st",
    "is_suspended",
}
_MARKET_REQUIRED_COLUMNS = {"code", "trading_date", "pct", "total_mv"}


def _check_required_columns(data: pl.DataFrame, required: set[str]) -> None:
    """在计算前给出明确缺列报错，避免 Polars 表达式报错难以定位。"""
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"输入数据缺少必要字段: {', '.join(missing)}")


def _add_code_prefix(data: pl.DataFrame) -> pl.DataFrame:
    """提取 Goldminer/RQ 代码中证券数字部分的首位。"""
    return data.with_columns(pl.col("code").str.slice(5, 1).alias("_code_prefix"))


def build_weekly_basic_factors(daily_raw: pl.DataFrame) -> pl.DataFrame:
    """从股票日线构造五个不含打板的周度情绪因子。

    周度涨停/跌停占比的分母采用该周单日最大有效股票数，与研报阈值的量级
    保持一致。涨跌停次日收益用所有有效事件的收益直接平均，不能先做日均值
    再做周均值；后者会错误地给少事件日相同权重。

    ``next_trading_date`` 由全市场交易日历确定，并只在该股票确有该日记录时
    取收益。只有该日期仍处于当前 ISO 周、且不晚于当前周实际最后交易日时，收益
    才可用于本周因子，因此能自动处理节假日短周、停牌缺行并避免把下周收益泄漏
    给本周。
    """
    _check_required_columns(daily_raw, _BASIC_REQUIRED_COLUMNS)

    # 先由全市场日期构造下一交易日，再与个股行连接。不能在单股内直接
    # ``shift(-1)``，否则停牌缺行会把两日后的记录错误地当作次日收益。
    market_calendar = (
        daily_raw.select("trading_date")
        .unique()
        .sort("trading_date")
        .with_columns(pl.col("trading_date").shift(-1).alias("_next_market_date"))
    )
    next_market_pct = daily_raw.select(["code", "trading_date", "pct"]).rename(
        {"trading_date": "_next_market_date", "pct": "_next_market_pct"}
    )

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
        .join(market_calendar, on="trading_date", how="left")
        .join(next_market_pct, on=["code", "_next_market_date"], how="left")
        .with_columns(
            [
                pl.col("trading_date").dt.strftime("%G-%V").alias("week_id"),
                (
                    (pl.col("close") >= pl.col("limit_up") - 0.01)
                    & (pl.col("close") > pl.col("pre_close"))
                ).alias("_is_limit_up"),
                (
                    (pl.col("close") <= pl.col("limit_down") + 0.01)
                    & (pl.col("close") < pl.col("pre_close"))
                ).alias("_is_limit_down"),
                pl.col("_next_market_date").alias("_next_trading_date"),
                (pl.col("_next_market_pct") / 100.0).alias("_next_day_return"),
            ]
        )
    )

    if eligible.is_empty():
        raise ValueError("没有满足股票池条件的有效日线记录")

    week_info = eligible.group_by("week_id").agg(
        [
            pl.col("trading_date").max().alias("week_end_date"),
            pl.col("trading_date").n_unique().alias("trading_days_in_week"),
        ]
    )

    eligible = eligible.join(week_info, on="week_id", how="left").with_columns(
        (
            pl.col("_next_trading_date").is_not_null()
            & (pl.col("_next_trading_date") <= pl.col("week_end_date"))
            & (pl.col("_next_trading_date").dt.strftime("%G-%V") == pl.col("week_id"))
            & pl.col("_next_day_return").is_finite()
        )
        .fill_null(False)
        .alias("_next_return_available")
    )

    daily_counts = eligible.group_by(["week_id", "trading_date"]).agg(
        [
            pl.len().alias("eligible_stock_count"),
            pl.col("_is_limit_up").cast(pl.Int64).sum().alias("limit_up_count"),
            pl.col("_is_limit_down").cast(pl.Int64).sum().alias("limit_down_count"),
        ]
    )

    weekly_counts = daily_counts.group_by("week_id").agg(
        [
            pl.col("eligible_stock_count").max().alias("max_daily_stock_count"),
            pl.col("limit_up_count").cast(pl.Int64).sum().alias("limit_up_count"),
            pl.col("limit_down_count").cast(pl.Int64).sum().alias("limit_down_count"),
        ]
    )

    weekly_events = eligible.group_by("week_id").agg(
        [
            pl.col("_next_day_return")
            .filter(pl.col("_is_limit_up") & pl.col("_next_return_available"))
            .mean()
            .alias("limit_up_next_ret"),
            pl.col("_next_day_return")
            .filter(pl.col("_is_limit_down") & pl.col("_next_return_available"))
            .mean()
            .alias("limit_down_next_ret"),
            (
                pl.col("_is_limit_up") & pl.col("_next_return_available")
            )
            .cast(pl.Int64)
            .sum()
            .alias("limit_up_event_count"),
            (
                pl.col("_is_limit_down") & pl.col("_next_return_available")
            )
            .cast(pl.Int64)
            .sum()
            .alias("limit_down_event_count"),
        ]
    )

    return (
        week_info.join(weekly_counts, on="week_id", how="left")
        .join(weekly_events, on="week_id", how="left")
        .with_columns(
            [
                (
                    pl.col("limit_up_count").cast(pl.Float64)
                    / pl.col("max_daily_stock_count")
                ).alias("limit_up_ratio"),
                (
                    pl.col("limit_down_count").cast(pl.Float64)
                    / pl.col("max_daily_stock_count")
                ).alias("limit_down_ratio"),
                (
                    (
                        pl.col("limit_up_count").cast(pl.Int64)
                        - pl.col("limit_down_count").cast(pl.Int64)
                    ).cast(pl.Float64)
                    / pl.col("max_daily_stock_count")
                ).alias("net_limit_ratio"),
            ]
        )
        .sort("week_end_date")
    )


def build_market_forward_returns(
    daily_raw: pl.DataFrame,
    horizons: Sequence[int] = (1, 3, 5, 10),
) -> pl.DataFrame:
    """构造本地全 A 市值加权市场收益及未来 h 个交易日收益。

    市场基准只把正且有限的 ``total_mv`` 作为权重，因此混入的指数记录或
    零市值记录不会影响收益。前瞻收益由按交易日排序的净值行 ``shift(-h)``
    计算，不把周末和节假日误作交易日。
    """
    _check_required_columns(daily_raw, _MARKET_REQUIRED_COLUMNS)
    normalized_horizons = tuple(dict.fromkeys(int(h) for h in horizons))
    if not normalized_horizons or any(h <= 0 for h in normalized_horizons):
        raise ValueError("horizons 必须是正整数序列")

    with_prefix = _add_code_prefix(daily_raw).filter(
        pl.col("_code_prefix").is_in(_A_SHARE_PREFIXES)
        & pl.col("pct").is_finite()
    )
    all_market_dates = with_prefix.select("trading_date").unique()
    weighted = with_prefix.filter(
        pl.col("total_mv").is_finite() & (pl.col("total_mv") > 0)
    )
    if weighted.is_empty():
        raise ValueError("没有正且有限的 total_mv，可用于市场收益的权重为空")

    daily = weighted.group_by("trading_date").agg(
        [
            pl.col("total_mv").sum().alias("_weight_sum"),
            (
                (pl.col("pct") / 100.0 * pl.col("total_mv")).sum()
                / pl.col("total_mv").sum()
            ).alias("market_daily_ret"),
        ]
    )
    missing_dates = all_market_dates.join(
        daily.select("trading_date"), on="trading_date", how="anti"
    )
    if not missing_dates.is_empty():
        sample_dates = ", ".join(str(value) for value in missing_dates["trading_date"].head(5))
        raise ValueError(f"以下交易日不存在有效市值权重: {sample_dates}")

    daily = daily.sort("trading_date")
    if daily.filter(~pl.col("market_daily_ret").is_finite()).height:
        raise ValueError("市场日收益出现非有限值，请检查 total_mv 与 pct")

    result = daily.with_columns(
        (1.0 + pl.col("market_daily_ret")).cum_prod().alias("market_nav")
    )
    result = result.with_columns(
        [
            (pl.col("market_nav").shift(-h) / pl.col("market_nav") - 1.0).alias(
                f"future_return_{h}d"
            )
            for h in normalized_horizons
        ]
    )
    return result.drop("_weight_sum")


def analyze_factor_effectiveness(
    weekly_factors: pl.DataFrame,
    factor_columns: Iterable[str],
    horizons: Sequence[int] = (1, 3, 5, 10),
    factor_directions: Mapping[str, float] | None = None,
    n_groups: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """评估市场级周频因子预测未来交易日收益的能力。

    这是时间序列检验，不是股票横截面 IC。因子与未来收益按周观测，对每个
    期限同时给出 Pearson、Spearman 与 HAC 回归统计。分组先乘以因子方向，
    所以 G5 始终代表“更强看多”的一端；例如跌停占比应传入 ``-1``。
    """
    if n_groups < 2:
        raise ValueError("n_groups 至少为 2")

    factor_list = list(factor_columns)
    horizons = tuple(dict.fromkeys(int(h) for h in horizons))
    required = set(factor_list) | {f"future_return_{h}d" for h in horizons}
    _check_required_columns(weekly_factors, required)
    directions = dict(factor_directions or {})
    data = weekly_factors.select(sorted(required)).to_pandas()

    summary_rows: list[dict[str, float | int | str]] = []
    group_rows: list[dict[str, float | int | str]] = []
    for factor in factor_list:
        direction = float(directions.get(factor, 1.0))
        if direction == 0:
            raise ValueError(f"{factor} 的方向不能为 0")

        for horizon in horizons:
            target = f"future_return_{horizon}d"
            valid = data[[factor, target]].rename(columns={factor: "x", target: "y"})
            valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
            n_obs = len(valid)
            base_row: dict[str, float | int | str] = {
                "factor": factor,
                "horizon": horizon,
                "n_obs": n_obs,
                "pearson_ic": np.nan,
                "spearman_ic": np.nan,
                "hac_beta": np.nan,
                "hac_t": np.nan,
                "hac_pvalue": np.nan,
                "directional_ic": np.nan,
                "q5_minus_q1": np.nan,
            }
            if n_obs < max(n_groups, 3) or valid["x"].nunique() < 2:
                summary_rows.append(base_row)
                continue

            pearson_ic = valid["x"].corr(valid["y"], method="pearson")
            spearman_ic = valid["x"].corr(valid["y"], method="spearman")
            maxlags = max(0, ceil(horizon / 5) - 1)
            fit = sm.OLS(valid["y"], sm.add_constant(valid["x"], has_constant="add")).fit(
                cov_type="HAC", cov_kwds={"maxlags": maxlags}
            )
            base_row.update(
                {
                    "pearson_ic": float(pearson_ic),
                    "spearman_ic": float(spearman_ic),
                    "hac_beta": float(fit.params["x"]),
                    "hac_t": float(fit.tvalues["x"]),
                    "hac_pvalue": float(fit.pvalues["x"]),
                    "directional_ic": float(direction * pearson_ic),
                }
            )

            # ``rank(method='first')`` 可在因子值有重复时稳定地切出完整分组。
            valid = valid.assign(_bullish_score=valid["x"] * direction)
            labels = [f"G{i}" for i in range(1, n_groups + 1)]
            valid = valid.assign(
                group=pd.qcut(
                    valid["_bullish_score"].rank(method="first"),
                    q=n_groups,
                    labels=labels,
                )
            )
            grouped = (
                valid.groupby("group", observed=False)["y"]
                .agg(mean_return="mean", win_rate=lambda values: (values > 0).mean(), n_obs="size")
                .reset_index()
            )
            grouped["group"] = grouped["group"].astype(str)
            grouped["factor"] = factor
            grouped["horizon"] = horizon
            group_rows.extend(grouped[["factor", "horizon", "group", "mean_return", "win_rate", "n_obs"]].to_dict("records"))
            mean_by_group = grouped.set_index("group")["mean_return"]
            base_row["q5_minus_q1"] = float(
                mean_by_group.loc[f"G{n_groups}"] - mean_by_group.loc["G1"]
            )
            summary_rows.append(base_row)

    summary_columns = [
        "factor",
        "horizon",
        "n_obs",
        "pearson_ic",
        "spearman_ic",
        "hac_beta",
        "hac_t",
        "hac_pvalue",
        "directional_ic",
        "q5_minus_q1",
    ]
    group_columns = ["factor", "horizon", "group", "mean_return", "win_rate", "n_obs"]
    return pd.DataFrame(summary_rows, columns=summary_columns), pd.DataFrame(
        group_rows, columns=group_columns
    )


def analyze_threshold_effectiveness(
    weekly_factors: pl.DataFrame,
    thresholds: Mapping[str, tuple[str, float]],
    horizons: Sequence[int] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """按研报阈值比较信号触发与未触发时的未来市场收益。

    阈值只用于描述既定研报规则的条件效果，不能把全样本调参结果解释为样本
    外预测能力。二元信号回归采用与连续因子相同的 HAC 协方差，便于处理 10 日
    未来收益与相邻周观测的重叠。
    """
    if not thresholds:
        raise ValueError("thresholds 不能为空")
    normalized_horizons = tuple(dict.fromkeys(int(h) for h in horizons))
    if not normalized_horizons or any(h <= 0 for h in normalized_horizons):
        raise ValueError("horizons 必须是正整数序列")

    factor_columns = set(thresholds)
    required = factor_columns | {f"future_return_{h}d" for h in normalized_horizons}
    _check_required_columns(weekly_factors, required)
    data = weekly_factors.select(sorted(required)).to_pandas()

    result_rows: list[dict[str, float | int | str]] = []
    for factor, (operator, threshold) in thresholds.items():
        if operator not in {"gt", "lt"}:
            raise ValueError(f"{factor} 的比较符只能是 'gt' 或 'lt'")
        for horizon in normalized_horizons:
            target = f"future_return_{horizon}d"
            valid = data[[factor, target]].rename(columns={factor: "x", target: "y"})
            valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
            trigger = valid["x"] > threshold if operator == "gt" else valid["x"] < threshold
            valid = valid.assign(_trigger=trigger.astype(int))
            triggered = valid.loc[valid["_trigger"] == 1, "y"]
            untriggered = valid.loc[valid["_trigger"] == 0, "y"]
            n_obs = len(valid)
            row: dict[str, float | int | str] = {
                "factor": factor,
                "operator": operator,
                "threshold": float(threshold),
                "horizon": horizon,
                "n_obs": n_obs,
                "trigger_count": int(trigger.sum()),
                "trigger_rate": float(trigger.mean()) if n_obs else np.nan,
                "trigger_mean_return": float(triggered.mean()) if len(triggered) else np.nan,
                "not_trigger_mean_return": float(untriggered.mean()) if len(untriggered) else np.nan,
                "mean_diff": np.nan,
                "trigger_win_rate": float((triggered > 0).mean()) if len(triggered) else np.nan,
                "hac_beta": np.nan,
                "hac_t": np.nan,
                "hac_pvalue": np.nan,
            }
            if len(triggered) and len(untriggered):
                row["mean_diff"] = float(triggered.mean() - untriggered.mean())
                maxlags = max(0, ceil(horizon / 5) - 1)
                fit = sm.OLS(
                    valid["y"], sm.add_constant(valid["_trigger"], has_constant="add")
                ).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
                row.update(
                    {
                        "hac_beta": float(fit.params["_trigger"]),
                        "hac_t": float(fit.tvalues["_trigger"]),
                        "hac_pvalue": float(fit.pvalues["_trigger"]),
                    }
                )
            result_rows.append(row)

    columns = [
        "factor",
        "operator",
        "threshold",
        "horizon",
        "n_obs",
        "trigger_count",
        "trigger_rate",
        "trigger_mean_return",
        "not_trigger_mean_return",
        "mean_diff",
        "trigger_win_rate",
        "hac_beta",
        "hac_t",
        "hac_pvalue",
    ]
    return pd.DataFrame(result_rows, columns=columns)
