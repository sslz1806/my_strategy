"""
米筐数据清洗与合并工具模块。

本模块负责将米筐代理 API 返回的原始数据转换为本地统一格式，
包括 MultiIndex 展开、字段归一化、复权因子展开、Schema 对齐和分区写入。

所有 Schema 常量也在此定义，便于多个脚本复用。
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Iterable

import pandas as pd
import polars as pl

DATA_ROOT_DIR = r"E:\working\stock_data"

from my_utils.mapping import convert_code_format, convert_date_format  # noqa: E402

# ============================================================================
# Schema 常量 —— 与掘金 gm_stock_all_data / 15min_stock_data_dir 对齐
# ============================================================================

RQ_DAY_SCHEMA = pl.Schema(
    {
        "code": pl.String,
        "name": pl.String,
        "trading_date": pl.Date,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "pre_close": pl.Float64,
        "pct": pl.Float64,
        "volume": pl.Float64,
        "amount": pl.Float64,
        "limit_up": pl.Float64,
        "limit_down": pl.Float64,
        "is_st": pl.Boolean,
        "is_suspended": pl.Boolean,
        "adj_factor": pl.Float64,
        "turnover_rate": pl.Float64,
        "total_mv": pl.Float64,
        "circulation_mv": pl.Float64,
        "mv_A_free_float": pl.Float64,
    }
)

RQ_MIN_SCHEMA = pl.Schema(
    {
        "code": pl.String,
        "datetime": pl.Datetime("us"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "trading_date": pl.Date,
    }
)

RQ_ADJ_SCHEMA = pl.Schema(
    {
        "code": pl.String,
        "trading_date": pl.Date,
        "adj_factor": pl.Float64,
    }
)


# ============================================================================
# 工具函数
# ============================================================================


def to_date(value) -> dt.date:
    """统一把入参、分区字符串或 pandas 日期转为 date。"""
    converted = convert_date_format(value, to_format="date")
    if converted is None:
        raise ValueError(f"无法识别日期: {value}")
    return converted


def to_date_str(value) -> str:
    return to_date(value).strftime("%Y-%m-%d")


def data_dir(save_dir: str) -> str:
    return os.path.join(DATA_ROOT_DIR, save_dir)


def get_existing_dates(save_dir: str) -> list[dt.date]:
    """从 trading_date=YYYY-MM-DD 分区目录中提取已有日期。"""
    target_dir = data_dir(save_dir)
    if not os.path.exists(target_dir):
        return []

    dates: list[dt.date] = []
    for item in os.listdir(target_dir):
        if not item.startswith("trading_date="):
            continue
        date_text = item.split("=", 1)[1]
        try:
            dates.append(to_date(date_text))
        except ValueError:
            logging.warning("跳过无法解析的分区目录: %s", item)
    return sorted(dates)


def get_local_parquet_schema(save_dir: str) -> pl.Schema | None:
    """读取目标目录下第一个 Parquet 文件的 schema。"""
    target_dir = data_dir(save_dir)
    if not os.path.exists(target_dir):
        return None

    for dirpath, _, filenames in os.walk(target_dir):
        for filename in filenames:
            if not filename.endswith(".parquet"):
                continue
            file_path = os.path.join(dirpath, filename)
            try:
                return pl.read_parquet(file_path, n_rows=0).schema
            except Exception as exc:
                logging.warning("读取 schema 失败，文件: %s，错误: %s", file_path, exc)
                return None
    return None


def infer_start_date(default_start: dt.date, save_dir: str, mode: str) -> dt.date:
    """insert 模式从目标目录最大日期后一日开始，update 模式使用用户给定起点。"""
    if mode != "insert":
        return default_start
    existing_dates = get_existing_dates(save_dir)
    if not existing_dates:
        return default_start
    return max(existing_dates) + dt.timedelta(days=1)


def ensure_rq_codes(codes: Iterable[str]) -> list[str]:
    """把本地 GM/纯数字/后缀代码统一转换为米筐 .XSHE/.XSHG 格式。"""
    rq_codes: list[str] = []
    for code in codes:
        suffix_code = convert_code_format(str(code), format="suffix")
        if suffix_code is None:
            continue
        code_num, market = suffix_code.split(".")
        rq_market = "XSHE" if market == "SZ" else "XSHG"
        rq_codes.append(f"{code_num}.{rq_market}")
    return rq_codes


# ============================================================================
# 米筐原始数据解析
# ============================================================================


def parse_rq_datetime_values(values) -> pd.Series:
    """解析米筐代理返回的时间列；数值型统一按毫秒时间戳处理。"""
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", errors="coerce")
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)


def expand_rq_multiindex(
    df: pd.DataFrame, timestamp_col: str, shift_minutes: int = 0
) -> pd.DataFrame:
    """
    展开米筐代理返回的 MultiIndex。

    米筐行情类接口返回 (order_book_id, date/time) 作为 MultiIndex；
    本地统一使用 code + trading_date/datetime 列。

    Args:
        df: 米筐代理返回的 DataFrame
        timestamp_col: 时间列名称，日线用 "trading_date"，分钟线用 "datetime"
        shift_minutes: 分钟线左移分钟数（米筐分钟线 timestamp 是 K 线结束时间）
    """
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()
    if isinstance(result.index, pd.MultiIndex):
        index_df = result.index.to_frame(index=False)
        if index_df.shape[1] < 2:
            raise ValueError("米筐 MultiIndex 至少需要包含 order_book_id 和时间两列")
        result = result.reset_index(drop=True)
        result["order_book_id"] = index_df.iloc[:, 0].to_list()
        timestamps = parse_rq_datetime_values(index_df.iloc[:, 1])
    else:
        result = result.reset_index()
        if "order_book_id" not in result.columns:
            raise ValueError("米筐数据缺少 order_book_id 索引")
        date_col = "date" if "date" in result.columns else result.columns[1]
        timestamps = parse_rq_datetime_values(result[date_col])

    if shift_minutes:
        timestamps = timestamps - pd.to_timedelta(shift_minutes, unit="m")

    result["code"] = convert_code_format(result["order_book_id"].astype(str), format="gm")
    result[timestamp_col] = timestamps.dt.date if timestamp_col == "trading_date" else timestamps
    result = result.dropna(subset=["code"])
    return result


def expand_rq_matrix(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    展开 is_st_stock / is_suspended 等矩阵结果。

    这类接口返回日期为索引、股票代码为列的矩阵，本函数将其转为长表。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "trading_date", value_name])

    matrix = df.copy()
    matrix.index = pd.to_datetime(matrix.index).date
    long_df = matrix.stack(dropna=False).reset_index()
    long_df.columns = ["trading_date", "order_book_id", value_name]
    long_df["code"] = convert_code_format(long_df["order_book_id"].astype(str), format="gm")
    long_df = long_df.dropna(subset=["code"])
    return long_df[["code", "trading_date", value_name]]


# ============================================================================
# 字段归一化
# ============================================================================


def normalize_instruments(instruments: pd.DataFrame, trade_date: dt.date) -> pd.DataFrame:
    """清洗股票基础信息，过滤当日未上市或已退市标的。"""
    if instruments is None or instruments.empty:
        return pd.DataFrame(columns=["code", "name"])

    info = instruments.copy()
    info["code"] = convert_code_format(info["order_book_id"].astype(str), format="gm")
    info["name"] = info.get("symbol")
    info["listed_date"] = pd.to_datetime(info.get("listed_date"), errors="coerce").dt.date
    delisted = info.get("de_listed_date")
    info["de_listed_date"] = pd.to_datetime(
        delisted.replace("0000-00-00", "2099-12-31"), errors="coerce"
    ).dt.date

    mask = (info["listed_date"] <= trade_date) & (info["de_listed_date"] > trade_date)
    if "status" in info.columns:
        mask &= info["status"].fillna("Active").eq("Active")
    info = info[mask].dropna(subset=["code"])
    return info[["code", "name"]].drop_duplicates(subset=["code"], keep="last")


def normalize_turnover(turnover: pd.DataFrame) -> pd.DataFrame:
    """清洗 get_turnover_rate 返回的换手率数据。"""
    if turnover is None or turnover.empty:
        return pd.DataFrame(columns=["code", "trading_date", "turnover_rate"])
    data = expand_rq_multiindex(turnover, timestamp_col="trading_date")
    if "today" not in data.columns:
        return pd.DataFrame(columns=["code", "trading_date", "turnover_rate"])
    return data.rename(columns={"today": "turnover_rate"})[["code", "trading_date", "turnover_rate"]]


def normalize_shares(
    shares_data: pd.DataFrame,
    trading_dates: list[dt.date],
) -> pd.DataFrame:
    """
    将 get_shares 返回的股本数据展开为每日股本，用于计算市值。

    get_shares 返回 total_a（A股总股本）、circulation_a（流通A股）、
    free_circulation（自由流通股本）。股本数据仅在变动日有记录，
    需要按交易日向前填充。

    Args:
        shares_data: get_shares API 返回的 DataFrame
        trading_dates: 需要覆盖的交易日列表

    Returns:
        DataFrame，含 code, trading_date, total_a, circulation_a, free_circulation
    """
    if shares_data is None or shares_data.empty:
        return pd.DataFrame(
            columns=["code", "trading_date", "total_a", "circulation_a", "free_circulation"]
        )

    data = expand_rq_multiindex(shares_data, timestamp_col="trading_date")
    share_cols = ["total_a", "circulation_a", "free_circulation"]
    available_cols = [c for c in share_cols if c in data.columns]
    if not available_cols:
        logging.warning("get_shares 返回数据缺少股本字段: %s", share_cols)
        return pd.DataFrame(
            columns=["code", "trading_date", "total_a", "circulation_a", "free_circulation"]
        )

    data = data[["code", "trading_date"] + available_cols].dropna(subset=["code"])
    if data.empty:
        return pd.DataFrame(
            columns=["code", "trading_date", "total_a", "circulation_a", "free_circulation"]
        )

    # 构建完整的 code × trading_date 网格
    codes = sorted(data["code"].unique().tolist())
    trading_dates_sorted = sorted(trading_dates)
    full_index = pd.MultiIndex.from_product(
        [codes, trading_dates_sorted], names=["code", "trading_date"]
    )
    grid = pd.DataFrame(index=full_index).reset_index()

    # 合并并向前填充
    data["trading_date"] = pd.to_datetime(data["trading_date"])
    grid["trading_date"] = pd.to_datetime(grid["trading_date"])
    merged = grid.merge(data, on=["code", "trading_date"], how="left")
    merged = merged.sort_values(["code", "trading_date"])
    for col in available_cols:
        merged[col] = merged.groupby("code")[col].ffill()

    # 补充未出现在 shares_data 中的列
    for col in share_cols:
        if col not in merged.columns:
            merged[col] = float("nan")

    merged["trading_date"] = merged["trading_date"].dt.date
    return merged[["code", "trading_date"] + share_cols]


def normalize_ex_factor(ex_factor: pd.DataFrame) -> pd.DataFrame:
    """清洗 get_ex_factor 返回结果，保留除权日和累计复权因子。"""
    if ex_factor is None or ex_factor.empty:
        return pd.DataFrame(columns=["code", "ex_date", "ex_cum_factor"])

    data = ex_factor.copy()
    if "ex_date" in data.columns:
        ex_dates = parse_rq_datetime_values(data["ex_date"]).dt.date
    else:
        ex_dates = parse_rq_datetime_values(data.index).dt.date

    data["code"] = convert_code_format(data["order_book_id"].astype(str), format="gm")
    data["ex_date"] = list(ex_dates)
    data = data.dropna(subset=["code", "ex_date", "ex_cum_factor"])
    return data[["code", "ex_date", "ex_cum_factor"]].sort_values(["code", "ex_date"])


def build_daily_adj_factor(
    ex_factor: pd.DataFrame,
    rq_codes: list[str],
    trading_dates: list[dt.date],
) -> pl.DataFrame:
    """
    将米筐事件型 ex_cum_factor 展开为每日 adj_factor。

    对每只股票按交易日向前填充最近一次累计因子；
    早于第一条因子事件的日期填 1.0。
    """
    if not rq_codes or not trading_dates:
        return pl.DataFrame(schema=RQ_ADJ_SCHEMA)

    event_data = normalize_ex_factor(ex_factor)
    gm_codes = convert_code_format(rq_codes, format="gm")
    trading_dates_ts = pd.to_datetime(sorted(trading_dates))

    calendar = pd.MultiIndex.from_product(
        [gm_codes, trading_dates_ts], names=["code", "trading_date"]
    ).to_frame(index=False)

    if event_data.empty:
        calendar["adj_factor"] = 1.0
    else:
        calendar = calendar.sort_values(["code", "trading_date"])
        event_data = event_data.rename(
            columns={"ex_date": "trading_date", "ex_cum_factor": "adj_factor"}
        )
        event_data["trading_date"] = pd.to_datetime(event_data["trading_date"])
        event_data = event_data.sort_values(["code", "trading_date"])

        merged_parts = []
        for code, group in calendar.groupby("code", sort=False):
            code_events = event_data[event_data["code"] == code]
            merged = pd.merge_asof(
                group.sort_values("trading_date"),
                code_events[["trading_date", "adj_factor"]].sort_values("trading_date"),
                on="trading_date",
                direction="backward",
            )
            merged["code"] = code
            merged_parts.append(merged)
        calendar = pd.concat(merged_parts, ignore_index=True)
        calendar["adj_factor"] = calendar["adj_factor"].fillna(1.0)

    calendar["trading_date"] = pd.to_datetime(calendar["trading_date"]).dt.date
    result = pl.from_pandas(calendar[["code", "trading_date", "adj_factor"]])
    result = align_schema(result, RQ_ADJ_SCHEMA)
    return result.sort(["trading_date", "code"])


# ============================================================================
# 数据合并
# ============================================================================


def compute_market_values(day_data: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """
    将股本数据合并到日线行情，计算三种市值。

    Args:
        day_data: 展开后的日线行情 DataFrame（must have code, trading_date, close）
        shares: normalize_shares 输出的每日股本 DataFrame
    """
    if shares.empty or "close" not in day_data.columns:
        return day_data

    merged = day_data.copy()
    merged["trading_date"] = pd.to_datetime(merged["trading_date"])
    shares_copy = shares.copy()
    shares_copy["trading_date"] = pd.to_datetime(shares_copy["trading_date"])

    merged = merged.merge(shares_copy, on=["code", "trading_date"], how="left")

    for share_col, mv_col in [
        ("total_a", "total_mv"),
        ("circulation_a", "circulation_mv"),
        ("free_circulation", "mv_A_free_float"),
    ]:
        if share_col in merged.columns:
            merged[mv_col] = merged["close"] * merged[share_col]

    merged["trading_date"] = merged["trading_date"].dt.date
    return merged


def normalize_day_data(
    price_data: pd.DataFrame,
    instruments: pd.DataFrame,
    is_st: pd.DataFrame,
    suspended: pd.DataFrame,
    turnover: pd.DataFrame,
    shares: pd.DataFrame,
    adj_factor: pl.DataFrame,
) -> pl.DataFrame:
    """
    把米筐多接口日线结果合并为本地 rq_stock_all_data schema。

    合并顺序：价格 → 名称 → ST/停牌 → 换手率 → 股本(市值) → 复权因子 → pct
    """
    price = expand_rq_multiindex(price_data, timestamp_col="trading_date")
    if price.empty:
        return pl.DataFrame(schema=RQ_DAY_SCHEMA)

    price = price.rename(columns={"prev_close": "pre_close", "total_turnover": "amount"})
    dates = sorted(price["trading_date"].dropna().unique())
    trade_date = dates[0] if dates else dt.date.today()

    # 合并各数据源
    merged = price.merge(normalize_instruments(instruments, trade_date), on="code", how="left")
    merged = merged.merge(expand_rq_matrix(is_st, "is_st"), on=["code", "trading_date"], how="left")
    merged = merged.merge(
        expand_rq_matrix(suspended, "is_suspended"), on=["code", "trading_date"], how="left"
    )
    merged = merged.merge(normalize_turnover(turnover), on=["code", "trading_date"], how="left")

    # 市值：基于股本 × 收盘价
    merged = compute_market_values(merged, shares)

    # 复权因子
    adj_pd = adj_factor.to_pandas() if isinstance(adj_factor, pl.DataFrame) else pd.DataFrame()
    if not adj_pd.empty:
        adj_pd["trading_date"] = pd.to_datetime(adj_pd["trading_date"]).dt.date
        merged = merged.merge(adj_pd, on=["code", "trading_date"], how="left")

    # 布尔字段填充
    merged["is_st"] = merged["is_st"].fillna(False).astype(bool)
    merged["is_suspended"] = merged["is_suspended"].fillna(False).astype(bool)

    result = pl.from_pandas(merged)

    # 计算涨跌幅 pct = (close - pre_close) / pre_close * 100
    if "pct" not in result.columns or result["pct"].null_count() == result.height:
        result = result.with_columns(
            ((pl.col("close") - pl.col("pre_close")) / pl.col("pre_close") * 100).alias("pct")
        )

    result = align_schema(result, RQ_DAY_SCHEMA)
    return result.sort(["trading_date", "code"])


def normalize_minute_data(minute_data: pd.DataFrame, bar_minutes: int = 15) -> pl.DataFrame:
    """把米筐分钟线转换为本地分钟目录 schema。"""
    data = expand_rq_multiindex(minute_data, timestamp_col="datetime", shift_minutes=bar_minutes)
    if data.empty:
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    data["trading_date"] = pd.to_datetime(data["datetime"]).dt.date
    result = pl.from_pandas(data)
    result = align_schema(result, RQ_MIN_SCHEMA)
    return result.sort(["trading_date", "code", "datetime"])


# ============================================================================
# Schema 对齐与写入
# ============================================================================


def align_schema(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """
    对齐输出 schema。

    处理顺序：缺列补空值 → 裁剪额外列 → 严格按目标类型转换。
    保证写入同一目录的 Parquet 分区列顺序和类型稳定。
    """
    # 补齐缺失列
    for col, dtype in schema.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    # 裁剪并转换类型
    exprs = [pl.col(col).cast(dtype, strict=False).alias(col) for col, dtype in schema.items()]
    return df.select(exprs)


def prepare_for_write(
    data: pl.DataFrame, save_dir: str, schema: pl.Schema, mode: str
) -> pl.DataFrame:
    """按模式筛选待写入数据，并按已有目录 schema 对齐。"""
    if data.is_empty():
        return data

    output = align_schema(data, schema).sort(["trading_date", "code"])
    if mode == "insert":
        existing_dates = get_existing_dates(save_dir)
        if existing_dates:
            latest_date = max(existing_dates)
            output = output.filter(pl.col("trading_date") > latest_date)

    existing_schema = get_local_parquet_schema(save_dir)
    if existing_schema:
        output = align_schema(output, existing_schema)
    return output


def write_partitioned(
    data: pl.DataFrame, save_dir: str, schema: pl.Schema, mode: str
) -> int:
    """写入分区 Parquet，返回实际写入行数。

    mode='insert': 仅写入新日期（跳过已有日期）
    mode='update': 先清理目标日期的旧分区，再写入（实现真正的覆盖更新）
    """
    output = prepare_for_write(data, save_dir, schema, mode)
    if output.is_empty():
        logging.info("%s 没有需要写入的新数据", save_dir)
        return 0

    target_dir = data_dir(save_dir)
    os.makedirs(target_dir, exist_ok=True)

    # update 模式：先清理待写入日期的旧分区，避免文件堆积
    if mode == "update":
        dates_to_update = output.select(pl.col("trading_date")).unique().to_series().to_list()
        for d in dates_to_update:
            date_partition = os.path.join(target_dir, f"trading_date={to_date_str(d)}")
            if os.path.exists(date_partition):
                import shutil
                shutil.rmtree(date_partition)
                logging.debug("  清除旧分区: trading_date=%s", to_date_str(d))

    output.write_parquet(target_dir, partition_by=["trading_date"])
    logging.info("%s 写入完成: %s 行", save_dir, output.height)
    return output.height
