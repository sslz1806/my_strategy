"""
对比 Tushare 本地数据源与掘金本地数据源的关键字段差异。

用途：
    1. 服务于“高人气补涨策略_v4_均衡优化版”迁移到 gm_stock_all_data 前的口径核查。
    2. 只比较会影响该策略信号的关键字段，不直接修改策略 Notebook。
    3. 输出 CSV 明细，方便后续复查和人工确认。

运行示例：
    E:\\working\\anaconda3\\envs\\quant\\python.exe 数据源关键字段差异报告.py
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import polars as pl

from my_utils.fun import read_day_data


KEYS = ["code", "trading_date"]
PRICE_FIELDS = ["open", "high", "low", "close", "pre_close", "limit_up", "limit_down"]


def _parse_date(value: str | None, default: dt.date) -> dt.date:
    """把命令行日期参数转换成 date；为空时使用默认值。"""
    if not value:
        return default
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _with_prefix(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """除主键外，给字段统一加数据源前缀，避免 join 后字段含义混淆。"""
    return df.rename({col: f"{prefix}_{col}" for col in df.columns if col not in KEYS})


def _safe_ratio_expr(numerator: str, denominator: str) -> pl.Expr:
    """计算口径比例，分母为 0 或空时返回 null，避免极端值误导报告。"""
    return (
        pl.when((pl.col(denominator).is_not_null()) & (pl.col(denominator).abs() > 1e-12))
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
    )


def _field_diff_summary(joined: pl.DataFrame, field_pairs: list[tuple[str, str, str]]) -> pl.DataFrame:
    """生成字段级差异概览：差异数量、最大绝对差、均值差、比例中位数等。"""
    rows = []
    for label, left_col, right_col in field_pairs:
        if left_col not in joined.columns or right_col not in joined.columns:
            rows.append(
                {
                    "field": label,
                    "left_col": left_col,
                    "right_col": right_col,
                    "status": "missing_column",
                    "compare_rows": joined.height,
                    "diff_count": None,
                    "diff_ratio": None,
                    "max_abs_diff": None,
                    "mean_abs_diff": None,
                    "right_div_left_median": None,
                    "right_div_left_p01": None,
                    "right_div_left_p99": None,
                }
            )
            continue

        diff_expr = (pl.col(left_col) - pl.col(right_col)).abs()
        ratio_expr = _safe_ratio_expr(right_col, left_col)
        stats = joined.select(
            pl.len().alias("compare_rows"),
            (diff_expr.fill_null(0) > 1e-9).sum().alias("diff_count"),
            diff_expr.max().alias("max_abs_diff"),
            diff_expr.mean().alias("mean_abs_diff"),
            ratio_expr.median().alias("right_div_left_median"),
            ratio_expr.quantile(0.01).alias("right_div_left_p01"),
            ratio_expr.quantile(0.99).alias("right_div_left_p99"),
        ).to_dicts()[0]
        compare_rows = stats["compare_rows"] or 0
        diff_count = stats["diff_count"] or 0
        rows.append(
            {
                "field": label,
                "left_col": left_col,
                "right_col": right_col,
                "status": "ok",
                "compare_rows": compare_rows,
                "diff_count": diff_count,
                "diff_ratio": diff_count / compare_rows if compare_rows else None,
                "max_abs_diff": stats["max_abs_diff"],
                "mean_abs_diff": stats["mean_abs_diff"],
                "right_div_left_median": stats["right_div_left_median"],
                "right_div_left_p01": stats["right_div_left_p01"],
                "right_div_left_p99": stats["right_div_left_p99"],
            }
        )
    return pl.DataFrame(rows)


def _rank_diff_summary(joined: pl.DataFrame) -> pl.DataFrame:
    """比较按日横截面排名差异，关注高人气策略里用到的成交额/换手率排名。"""
    rank_df = joined.with_columns(
        [
            pl.col("ts_amount").rank(method="average", descending=True).over("trading_date").alias("ts_amount_rank"),
            pl.col("gm_amount").rank(method="average", descending=True).over("trading_date").alias("gm_amount_rank"),
            pl.col("ts_turn_over").rank(method="average", descending=True).over("trading_date").alias("ts_turnover_rank"),
            pl.col("gm_turnover_rate").rank(method="average", descending=True).over("trading_date").alias("gm_turnover_rank"),
        ]
    ).with_columns(
        [
            (pl.col("ts_amount_rank") - pl.col("gm_amount_rank")).abs().alias("amount_rank_abs_diff"),
            (pl.col("ts_turnover_rank") - pl.col("gm_turnover_rank")).abs().alias("turnover_rank_abs_diff"),
        ]
    )

    return pl.DataFrame(
        [
            {
                "rank_field": "amount",
                "compare_rows": rank_df.height,
                "mean_rank_abs_diff": rank_df["amount_rank_abs_diff"].mean(),
                "p95_rank_abs_diff": rank_df["amount_rank_abs_diff"].quantile(0.95),
                "max_rank_abs_diff": rank_df["amount_rank_abs_diff"].max(),
            },
            {
                "rank_field": "turnover",
                "compare_rows": rank_df.height,
                "mean_rank_abs_diff": rank_df["turnover_rank_abs_diff"].mean(),
                "p95_rank_abs_diff": rank_df["turnover_rank_abs_diff"].quantile(0.95),
                "max_rank_abs_diff": rank_df["turnover_rank_abs_diff"].max(),
            },
        ]
    )


def build_report(start_date: dt.date, end_date: dt.date, output_dir: Path, sample_size: int) -> dict[str, pl.DataFrame]:
    """读取两套数据源并生成关键字段差异报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ts_stock_all_data 是高人气策略当前主行情源；gm_stock_all_data 是回测 demo 使用的掘金源。
    ts_fields = KEYS + PRICE_FIELDS + ["volume", "amount", "turn_over", "type", "name", "float_mv", "total_mv"]
    gm_fields = KEYS + PRICE_FIELDS + ["volume", "amount", "turnover_rate", "is_st", "name", "mv_A_free_float", "total_mv"]
    basic_fields = KEYS + ["free_share", "close", "turnover_rate"]

    ts = _with_prefix(read_day_data(start_date, end_date, fields=ts_fields, file_path="ts_stock_all_data"), "ts")
    gm = _with_prefix(read_day_data(start_date, end_date, fields=gm_fields, file_path="gm_stock_all_data"), "gm")
    ts_basic = read_day_data(start_date, end_date, fields=basic_fields, file_path="ts_daily_basic").with_columns(
        [
            # 与高人气策略 Notebook 保持一致：free_share * close / 1e4，单位按“亿”理解。
            (pl.col("free_share") * pl.col("close") / 1e4).alias("ts_basic_free_float_mv"),
            pl.col("turnover_rate").alias("ts_basic_turnover_rate"),
        ]
    ).select(KEYS + ["ts_basic_free_float_mv", "ts_basic_turnover_rate"])

    ts_keys = ts.select(KEYS).unique()
    gm_keys = gm.select(KEYS).unique()
    missing_in_gm = ts_keys.join(gm_keys, on=KEYS, how="anti").sort(KEYS)
    missing_in_ts = gm_keys.join(ts_keys, on=KEYS, how="anti").sort(KEYS)

    joined = ts.join(gm, on=KEYS, how="inner").join(ts_basic, on=KEYS, how="left")
    joined = joined.with_columns(
        [
            (pl.col("gm_mv_A_free_float") / 1e8).alias("gm_mv_A_free_float_yi"),
            (pl.col("gm_total_mv") / 1e8).alias("gm_total_mv_yi"),
            (pl.col("ts_type") == "ST").fill_null(False).alias("ts_is_st"),
            pl.col("gm_is_st").fill_null(False).alias("gm_is_st_filled"),
        ]
    )

    summary = pl.DataFrame(
        [
            {"metric": "start_date", "value": str(start_date)},
            {"metric": "end_date", "value": str(end_date)},
            {"metric": "ts_rows", "value": str(ts.height)},
            {"metric": "gm_rows", "value": str(gm.height)},
            {"metric": "joined_rows", "value": str(joined.height)},
            {"metric": "ts_only_code_date", "value": str(missing_in_gm.height)},
            {"metric": "gm_only_code_date", "value": str(missing_in_ts.height)},
        ]
    )

    field_pairs = [(field, f"ts_{field}", f"gm_{field}") for field in PRICE_FIELDS]
    field_pairs.extend(
        [
            ("volume_raw", "ts_volume", "gm_volume"),
            ("amount_raw", "ts_amount", "gm_amount"),
            ("turnover_ts_turn_over_vs_gm_turnover_rate", "ts_turn_over", "gm_turnover_rate"),
            ("turnover_ts_daily_basic_vs_gm_turnover_rate", "ts_basic_turnover_rate", "gm_turnover_rate"),
            ("free_float_mv_ts_basic_vs_gm_yi", "ts_basic_free_float_mv", "gm_mv_A_free_float_yi"),
            ("float_mv_ts_stock_vs_gm_yi", "ts_float_mv", "gm_mv_A_free_float_yi"),
            ("total_mv_ts_stock_vs_gm_yi", "ts_total_mv", "gm_total_mv_yi"),
        ]
    )
    field_summary = _field_diff_summary(joined, field_pairs)
    rank_summary = _rank_diff_summary(joined)

    limit_diff_sample = joined.filter(
        ((pl.col("ts_limit_up") - pl.col("gm_limit_up")).abs().fill_null(0) > 1e-9)
        | ((pl.col("ts_limit_down") - pl.col("gm_limit_down")).abs().fill_null(0) > 1e-9)
    ).select(
        KEYS
        + [
            "ts_name",
            "gm_name",
            "ts_close",
            "gm_close",
            "ts_limit_up",
            "gm_limit_up",
            "ts_limit_down",
            "gm_limit_down",
        ]
    ).head(sample_size)

    market_value_diff_sample = joined.with_columns(
        ((pl.col("ts_basic_free_float_mv") - pl.col("gm_mv_A_free_float_yi")).abs()).alias("free_float_mv_abs_diff")
    ).sort("free_float_mv_abs_diff", descending=True).select(
        KEYS
        + [
            "ts_name",
            "gm_name",
            "ts_basic_free_float_mv",
            "gm_mv_A_free_float_yi",
            "free_float_mv_abs_diff",
            "ts_float_mv",
            "gm_total_mv_yi",
        ]
    ).head(sample_size)

    st_diff_sample = joined.filter(pl.col("ts_is_st") != pl.col("gm_is_st_filled")).select(
        KEYS + ["ts_name", "gm_name", "ts_type", "gm_is_st_filled"]
    ).head(sample_size)

    missing_samples = pl.concat(
        [
            missing_in_gm.head(sample_size).with_columns(pl.lit("ts_only_missing_in_gm").alias("missing_type")),
            missing_in_ts.head(sample_size).with_columns(pl.lit("gm_only_missing_in_ts").alias("missing_type")),
        ],
        how="vertical",
    ).select(["missing_type"] + KEYS)

    reports = {
        "summary": summary,
        "field_summary": field_summary,
        "rank_summary": rank_summary,
        "limit_diff_sample": limit_diff_sample,
        "market_value_diff_sample": market_value_diff_sample,
        "st_diff_sample": st_diff_sample,
        "missing_code_date_sample": missing_samples,
    }

    for name, df in reports.items():
        df.write_csv(output_dir / f"{name}.csv", include_bom=True)

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Tushare 与掘金关键字段差异报告")
    parser.add_argument("--start-date", default="2024-01-01", help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="结束日期，格式 YYYY-MM-DD；默认今天")
    parser.add_argument("--output-dir", default="signals_v2/data_source_key_diff", help="报告输出目录")
    parser.add_argument("--sample-size", type=int, default=50, help="样例明细最多输出多少行")
    args = parser.parse_args()

    start_date = _parse_date(args.start_date, dt.date(2024, 1, 1))
    end_date = _parse_date(args.end_date, dt.date.today())
    reports = build_report(start_date, end_date, Path(args.output_dir), args.sample_size)

    print("\n=== 数据源关键字段差异摘要 ===")
    print(reports["summary"].to_pandas().to_string(index=False))
    print("\n=== 字段差异概览 ===")
    print(reports["field_summary"].to_pandas().to_string(index=False))
    print("\n=== 排名差异概览 ===")
    print(reports["rank_summary"].to_pandas().to_string(index=False))
    print(f"\n报告已输出到: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
