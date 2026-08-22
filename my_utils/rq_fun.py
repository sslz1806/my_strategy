"""
米筐数据清洗与合并工具模块。

本模块负责将米筐代理 API 返回的原始数据转换为本地统一格式，
包括 MultiIndex 展开、字段归一化、复权因子展开、Schema 对齐和分区写入。

所有 Schema 常量也在此定义，便于多个脚本复用。
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
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

RQ_ETF_DAY_SCHEMA = pl.Schema(
    {
        "code": pl.String,
        "trading_date": pl.Date,
        "pre_close": pl.Float64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "change": pl.Float64,
        "pct": pl.Float64,
        "volume": pl.Float64,
        "amount": pl.Float64,
    }
)

RQ_ETF_MIN_SCHEMA = pl.Schema(
    {
        "code": pl.String,
        "datetime": pl.Datetime("us"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "amount": pl.Float64,
        "trading_date": pl.Date,
    }
)

MORNING_START_MINUTE = 9 * 60 + 30
MORNING_FIRST_TRADE_MINUTE = 9 * 60 + 31
MORNING_END_MINUTE = 11 * 60 + 30
AFTERNOON_START_MINUTE = 13 * 60
AFTERNOON_FIRST_TRADE_MINUTE = 13 * 60 + 1
AFTERNOON_END_MINUTE = 15 * 60
MINUTE_BAR_SIZE = 15
MINUTE_EXPECTED_BARS_PER_CODE = 18
MINUTE_MARKET_GUARD_MIN_CODES = 100
DAY_MARKET_GUARD_MIN_CODES = 100
# 股票池是跨历史期静态全集，不能要求每天 100% 上市；50% 相对门槛配合
# 100 只绝对下限，可拦截“查询只返回小片段”而不误伤 2021 年后的历史截面。
MARKET_GUARD_MIN_RATIO = 0.5


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


def remove_existing_partitions_in_range(
    save_dir: str,
    start_date: dt.date,
    end_date: dt.date,
) -> int:
    """精确删除日期范围内的分区目录，返回删除数量。"""
    target_dir = data_dir(save_dir)
    if not os.path.isdir(target_dir):
        return 0

    removed = 0
    for item in os.scandir(target_dir):
        if not item.is_dir() or not item.name.startswith("trading_date="):
            continue
        try:
            partition_date = to_date(item.name.split("=", 1)[1])
        except ValueError:
            logging.warning("跳过无法解析的分区目录: %s", item.path)
            continue
        if start_date <= partition_date <= end_date:
            shutil.rmtree(item.path)
            removed += 1

    if removed:
        logging.info(
            "%s 清理旧分区: %s 个（%s ~ %s）",
            save_dir,
            removed,
            to_date_str(start_date),
            to_date_str(end_date),
        )
    return removed


def cleanup_new_failed_partitions(
    save_dir: str,
    failed_dates: Iterable[dt.date],
    dates_before_run: set[dt.date],
) -> list[dt.date]:
    """清理失败批次的新分区，绝不删除任务启动前已有的完整日期。

    写入器通常会自行回滚；本函数是额度中断、进程异常后的第二道保护。
    所有删除目标都必须是 ``DATA_ROOT_DIR/save_dir`` 的直接子目录，避免错误
    参数或路径穿越把清理范围扩大到其他数据目录。
    """
    data_root = Path(DATA_ROOT_DIR).resolve()
    target_root = (data_root / save_dir).resolve()
    if target_root.parent != data_root:
        raise ValueError(f"save_dir is outside DATA_ROOT_DIR: {save_dir}")
    if not target_root.exists():
        return []

    # 原子写入正常情况下会删除暂存目录；只清理目标数据目录内、固定前缀的
    # 遗留项，不接触备份目录或任何其他路径。
    for staging_path in target_root.glob(".rq-staging-*"):
        resolved_staging = staging_path.resolve()
        if resolved_staging.parent == target_root:
            _remove_path(str(resolved_staging))

    protected_dates = {to_date(value) for value in dates_before_run}
    removed: list[dt.date] = []
    for trade_date in sorted({to_date(value) for value in failed_dates}):
        if trade_date in protected_dates:
            continue
        partition = (target_root / f"trading_date={to_date_str(trade_date)}").resolve()
        if partition.parent != target_root:
            raise ValueError(f"partition is outside target directory: {partition}")
        if partition.is_dir():
            shutil.rmtree(partition)
            removed.append(trade_date)

    if removed:
        logging.warning("已清理本轮失败的新 ETF 分区: %s", removed)
    return removed


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


def infer_start_date(
    default_start: dt.date,
    save_dir: str,
    mode: str,
    end_date: dt.date | None = None,
) -> dt.date | None:
    """推断更新起点；目录已覆盖到结束日期时返回 ``None``。"""
    if mode != "insert":
        return default_start
    existing_dates = get_existing_dates(save_dir)
    if not existing_dates:
        return default_start
    inferred_start = max(existing_dates) + dt.timedelta(days=1)
    if end_date is not None and inferred_start > end_date:
        return None
    return inferred_start


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


def _parse_etf_instrument_date(value, default: dt.date) -> dt.date:
    """解析米筐标的日期；空退市日按仍存续处理。"""
    if pd.isna(value) or str(value) in {"", "0000-00-00", "NaT", "None"}:
        return default
    parsed = pd.to_datetime(value, errors="coerce")
    return default if pd.isna(parsed) else parsed.date()


def normalize_etf_instruments(instruments: pd.DataFrame) -> pd.DataFrame:
    """清洗完整 ETF 历史池，不按当前存续状态过滤退市标的。"""
    required = {"order_book_id", "listed_date", "de_listed_date"}
    missing = required - set(instruments.columns)
    if missing:
        raise ValueError(f"ETF instruments missing columns: {sorted(missing)}")

    result = instruments.copy()
    if "type" in result.columns:
        result = result[result["type"].eq("ETF")]
    result = result[
        result["order_book_id"]
        .astype(str)
        .str.fullmatch(r"\d{6}\.(?:XSHG|XSHE)")
    ].copy()
    result["listed_date"] = result["listed_date"].map(
        lambda value: _parse_etf_instrument_date(value, dt.date.max)
    )
    result["de_listed_date"] = result["de_listed_date"].map(
        lambda value: _parse_etf_instrument_date(value, dt.date.max)
    )
    return result.sort_values("order_book_id").reset_index(drop=True)


def filter_etf_codes_for_range(
    instruments: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
) -> list[str]:
    """返回上市区间与目标日期范围有交集的全部 ETF 米筐代码。"""
    if start_date > end_date:
        return []
    active = instruments[
        instruments["listed_date"].le(end_date)
        & instruments["de_listed_date"].ge(start_date)
    ]
    return sorted(active["order_book_id"].dropna().astype(str).unique().tolist())


def quota_remaining_bytes(quota: dict, reserve_bytes: int) -> int | None:
    """返回扣除安全余量后的可用字节；上限为 0 表示官方账户不限流量。"""
    if reserve_bytes < 0:
        raise ValueError(f"reserve_bytes must not be negative: {reserve_bytes}")
    limit = int(quota.get("bytes_limit", 0))
    used = int(quota.get("bytes_used", 0))
    if limit == 0:
        return None
    return max(0, limit - used - reserve_bytes)


def measure_bytes_per_row(before: dict, after: dict, row_count: int) -> float:
    """根据一次真实行情请求前后的官方计数差，计算实际传输字节/行。"""
    if row_count <= 0:
        raise ValueError(f"row_count must be positive: {row_count}")
    consumed = int(after["bytes_used"]) - int(before["bytes_used"])
    if consumed <= 0:
        raise RuntimeError("quota usage did not increase after calibration")
    return consumed / row_count


def select_minute_days_for_quota(
    trading_days: list[dt.date],
    instruments: pd.DataFrame,
    available_bytes: int | None,
    bytes_per_row: float,
    safety_factor: float = 1.75,
) -> list[dt.date]:
    """按日期顺序返回今日额度能够容纳的最长连续交易日前缀。"""
    if bytes_per_row <= 0:
        raise ValueError(f"bytes_per_row must be positive: {bytes_per_row}")
    if safety_factor < 1:
        raise ValueError(f"safety_factor must be at least 1: {safety_factor}")

    days = sorted(set(trading_days))
    if available_bytes is None:
        return days
    if available_bytes <= 0:
        return []

    selected: list[dt.date] = []
    estimated_bytes = 0.0
    for trade_date in days:
        # 米筐每只正常交易 ETF 每日最多返回 240 根 1 分钟 Bar；这里按上限
        # 估算，停牌或缺失 Bar 只会让实际流量更小，不会低估正常交易日的需求。
        active_count = len(
            filter_etf_codes_for_range(instruments, trade_date, trade_date)
        )
        day_bytes = active_count * 240 * bytes_per_row * safety_factor
        if estimated_bytes + day_bytes > available_bytes:
            break
        selected.append(trade_date)
        estimated_bytes += day_bytes
    return selected


@dataclass(frozen=True)
class EtfRequestBatch:
    """一个不可拆代码、仅按连续交易日规划的 ETF 请求批次。"""

    start_date: dt.date
    end_date: dt.date
    trading_days: tuple[dt.date, ...]
    rq_codes: tuple[str, ...]
    estimated_rows: int


def _make_etf_request_batch(
    trading_days: list[dt.date],
    instruments: pd.DataFrame,
    estimated_rows: int,
) -> EtfRequestBatch:
    start_date, end_date = trading_days[0], trading_days[-1]
    return EtfRequestBatch(
        start_date=start_date,
        end_date=end_date,
        trading_days=tuple(trading_days),
        rq_codes=tuple(filter_etf_codes_for_range(instruments, start_date, end_date)),
        estimated_rows=estimated_rows,
    )


def build_etf_minute_batches(
    trading_days: list[dt.date],
    instruments: pd.DataFrame,
    max_rows: int = 3_000_000,
) -> list[EtfRequestBatch]:
    """在不拆 ETF 代码的前提下，按预计行数贪心合并连续交易日。"""
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive: {max_rows}")

    batches: list[EtfRequestBatch] = []
    current_days: list[dt.date] = []
    current_rows = 0
    for trade_date in sorted(set(trading_days)):
        active_codes = filter_etf_codes_for_range(
            instruments, trade_date, trade_date
        )
        if not active_codes:
            continue
        day_rows = len(active_codes) * 240
        if current_days and current_rows + day_rows > max_rows:
            batches.append(
                _make_etf_request_batch(current_days, instruments, current_rows)
            )
            current_days = []
            current_rows = 0
        current_days.append(trade_date)
        current_rows += day_rows

    if current_days:
        batches.append(_make_etf_request_batch(current_days, instruments, current_rows))
    return batches


def _validate_etf_codes(data: pl.DataFrame, rq_codes: list[str]) -> None:
    """确保响应没有混入本次未请求的 ETF。"""
    allowed_codes = set(convert_code_format(rq_codes, format="gm"))
    returned_codes = set(data["code"].drop_nulls().unique().to_list())
    unexpected_codes = sorted(returned_codes - allowed_codes)
    if unexpected_codes:
        raise RuntimeError(f"unexpected ETF codes: {unexpected_codes}")


def _validate_duplicate_keys(data: pl.DataFrame, key_cols: list[str]) -> None:
    """拒绝会让分区结果不确定的重复业务主键。"""
    if data.height and data.select(key_cols).is_duplicated().any():
        raise RuntimeError(f"ETF data duplicate keys: {key_cols}")


def _validate_missing_etf_dates(
    data: pl.DataFrame,
    expected_dates: list[dt.date],
    today: dt.date | None,
) -> None:
    """历史交易日必须完整；当天允许等待收盘后再次更新。"""
    check_date = today or dt.date.today()
    expected = {to_date(value) for value in expected_dates}
    present = set(data["trading_date"].unique().to_list()) if data.height else set()
    missing_history = sorted(value for value in expected - present if value < check_date)
    if missing_history:
        raise RuntimeError(f"ETF data missing trading days: {missing_history}")


def _validate_unexpected_etf_dates(
    data: pl.DataFrame,
    expected_dates: list[dt.date],
) -> None:
    """拒绝响应中越出请求交易日集合的数据。"""
    expected = {to_date(value) for value in expected_dates}
    present = set(data["trading_date"].unique().to_list()) if data.height else set()
    unexpected_dates = sorted(present - expected)
    if unexpected_dates:
        raise RuntimeError(f"ETF data outside requested dates: {unexpected_dates}")


def validate_etf_day_batch(
    data: pl.DataFrame,
    rq_codes: list[str],
    expected_dates: list[dt.date],
    today: dt.date | None = None,
) -> None:
    """校验一个 ETF 日线批次。"""
    _validate_etf_codes(data, rq_codes)
    _validate_unexpected_etf_dates(data, expected_dates)
    _validate_missing_etf_dates(data, expected_dates, today)
    _validate_duplicate_keys(data, ["code", "trading_date"])


def drop_incomplete_current_etf_minute_date(
    data: pl.DataFrame,
    today: dt.date | None = None,
) -> pl.DataFrame:
    """当天没有任何一只 ETF 达到 240 根时，剔除未完成的当日分区。"""
    check_date = today or dt.date.today()
    current = data.filter(pl.col("trading_date") == check_date)
    if current.is_empty():
        return data
    max_bars = current.group_by("code").len()["len"].max()
    if max_bars is not None and max_bars >= 240:
        return data
    logging.warning(
        "ETF 1min 当日数据尚未完成，不写入该日分区: %s", check_date
    )
    return data.filter(pl.col("trading_date") != check_date)


def validate_etf_minute_batch(
    data: pl.DataFrame,
    rq_codes: list[str],
    expected_dates: list[dt.date],
    today: dt.date | None = None,
) -> None:
    """校验一个 ETF 原始 1 分钟批次。"""
    _validate_etf_codes(data, rq_codes)
    _validate_unexpected_etf_dates(data, expected_dates)
    _validate_missing_etf_dates(data, expected_dates, today)
    _validate_duplicate_keys(data, ["code", "datetime"])
    if data.height and data.filter(
        pl.col("datetime").dt.date() != pl.col("trading_date")
    ).height:
        raise RuntimeError("ETF minute datetime does not match trading_date")


def _is_rq_a_share_code_expr(source_col: str = "order_book_id") -> pl.Expr:
    """判断米筐代码是否属于本项目覆盖的沪深 A 股代码段。"""
    code = pl.col(source_col)
    sh_a_share = code.str.contains(r"^(600|601|603|605|688|689)\d{3}\.XSHG$")
    sz_a_share = code.str.contains(r"^(000|001|002|003|300|301)\d{3}\.XSHE$")
    return sh_a_share | sz_a_share


def filter_to_stock_universe(data: pl.DataFrame, rq_codes: list[str]) -> pl.DataFrame:
    """按米筐股票池与沪深 A 股代码段双重过滤行情数据。"""
    if data.is_empty() or not rq_codes or "order_book_id" not in data.columns:
        return data
    return data.filter(
        pl.col("order_book_id").is_in(rq_codes)
        & _is_rq_a_share_code_expr("order_book_id")
    )


def normalize_ddb_day_range(
    kline: pd.DataFrame,
    is_st: pd.DataFrame,
    shares: pd.DataFrame,
    ex_factor: pd.DataFrame,
    instruments: pd.DataFrame,
    rq_codes: list[str],
    allowed_dates: Iterable[dt.date] | None = None,
) -> pl.DataFrame:
    """把 DDB 多表日线结果合并为统一的米筐本地日线 Schema。"""
    if kline is None or kline.empty or not rq_codes:
        return pl.DataFrame(schema=RQ_DAY_SCHEMA)

    result = pl.from_pandas(kline).with_columns(pl.col("trading_date").cast(pl.Date))
    if allowed_dates is not None:
        result = result.filter(pl.col("trading_date").is_in(list(allowed_dates)))
    result = filter_to_stock_universe(result, rq_codes)
    if result.is_empty():
        return pl.DataFrame(schema=RQ_DAY_SCHEMA)

    result = result.with_columns(
        [
            ((pl.col("close") / pl.col("pre_close") - 1) * 100).alias("pct"),
            (pl.col("volume") == 0)
            .and_(pl.col("amount") == 0)
            .alias("is_suspended"),
        ]
    )

    if is_st is not None and not is_st.empty:
        is_st_data = pl.from_pandas(is_st).with_columns(
            pl.col("trading_date").cast(pl.Date)
        )
        result = result.join(
            is_st_data,
            on=["order_book_id", "trading_date"],
            how="left",
        )
    else:
        result = result.with_columns(pl.lit(False).alias("is_st"))

    if shares is not None and not shares.empty:
        shares_data = pl.from_pandas(shares).with_columns(
            pl.col("trading_date").cast(pl.Date)
        )
        result = result.join(
            shares_data,
            on=["order_book_id", "trading_date"],
            how="left",
        )
    else:
        result = result.with_columns(
            [
                pl.lit(None, pl.Float64).alias("circulation_a"),
                pl.lit(None, pl.Float64).alias("total_a"),
                pl.lit(None, pl.Float64).alias("free_circulation"),
            ]
        )

    if ex_factor is not None and not ex_factor.empty:
        factor_data = (
            pl.from_pandas(ex_factor)
            .with_columns(pl.col("ex_date").cast(pl.Date))
            .sort(["order_book_id", "ex_date"])
        )
        result = result.sort(["order_book_id", "trading_date"]).join_asof(
            factor_data,
            left_on="trading_date",
            right_on="ex_date",
            by="order_book_id",
            strategy="backward",
        )
    else:
        result = result.with_columns(pl.lit(None, pl.Float64).alias("adj_factor"))

    if instruments is not None and not instruments.empty:
        instrument_data = pl.from_pandas(instruments).select(["order_book_id", "name"])
        result = result.join(instrument_data, on="order_book_id", how="left")
    else:
        result = result.with_columns(pl.lit(None, pl.String).alias("name"))

    result = result.with_columns(
        [
            pl.col("is_st").fill_null(False),
            pl.col("is_suspended").fill_null(True),
            pl.col("adj_factor").fill_null(1.0),
            pl.col("name").fill_null(""),
            pl.col("circulation_a").fill_null(0.0),
            pl.col("total_a").fill_null(0.0),
            pl.col("free_circulation").fill_null(0.0),
        ]
    ).with_columns(
        [
            pl.when(pl.col("circulation_a") > 0)
            .then(pl.col("volume").cast(pl.Float64) / pl.col("circulation_a") * 100)
            .otherwise(0.0)
            .alias("turnover_rate"),
            (pl.col("close") * pl.col("total_a")).alias("total_mv"),
            (pl.col("close") * pl.col("circulation_a")).alias("circulation_mv"),
            (pl.col("close") * pl.col("free_circulation")).alias("mv_A_free_float"),
            convert_code_format(pl.col("order_book_id"), format="gm").alias("code"),
        ]
    )

    result = result.select(list(RQ_DAY_SCHEMA.keys()))
    return align_schema(result, RQ_DAY_SCHEMA)


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


def _require_rq_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    """在 Schema 补空值前校验原始响应，避免接口缺列被静默写成空列。"""
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def normalize_etf_day_data(price_data: pd.DataFrame) -> pl.DataFrame:
    """将官方米筐 ETF 不复权日线转换为本地精简 Schema。"""
    if price_data is None or price_data.empty:
        return pl.DataFrame(schema=RQ_ETF_DAY_SCHEMA)

    data = expand_rq_multiindex(price_data, timestamp_col="trading_date")
    _require_rq_columns(
        data,
        {
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "volume",
            "total_turnover",
        },
        "ETF day data",
    )
    result = pl.from_pandas(
        data.rename(columns={"prev_close": "pre_close", "total_turnover": "amount"})
    )
    valid_pre_close = pl.col("pre_close").is_not_null() & (pl.col("pre_close") != 0)
    result = result.with_columns(
        [
            pl.when(valid_pre_close)
            .then(pl.col("close") - pl.col("pre_close"))
            .otherwise(None)
            .alias("change"),
            pl.when(valid_pre_close)
            .then((pl.col("close") / pl.col("pre_close") - 1) * 100)
            .otherwise(None)
            .alias("pct"),
        ]
    )
    return align_schema(result, RQ_ETF_DAY_SCHEMA).sort(["trading_date", "code"])


def normalize_etf_minute_data(price_data: pd.DataFrame) -> pl.DataFrame:
    """转换官方 ETF 1 分钟线，保留米筐原始 Bar 结束时间。"""
    if price_data is None or price_data.empty:
        return pl.DataFrame(schema=RQ_ETF_MIN_SCHEMA)

    data = expand_rq_multiindex(
        price_data, timestamp_col="datetime", shift_minutes=0
    )
    _require_rq_columns(
        data,
        {"open", "high", "low", "close", "volume", "total_turnover"},
        "ETF minute data",
    )
    data = data.rename(columns={"total_turnover": "amount"})
    data["trading_date"] = pd.to_datetime(data["datetime"]).dt.date
    result = pl.from_pandas(data)
    return align_schema(result, RQ_ETF_MIN_SCHEMA).sort(
        ["trading_date", "code", "datetime"]
    )


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


def aggregate_right_aligned_15min(
    raw: pl.DataFrame,
    rq_codes: Iterable[str] | None = None,
    allowed_dates: set[dt.date] | None = None,
) -> pl.DataFrame:
    """将 DDB 1 分钟线合成为项目约定的右对齐 15 分钟线。"""
    if raw.is_empty():
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    data = raw
    if rq_codes is not None:
        rq_code_list = list(rq_codes)
        if not rq_code_list:
            return pl.DataFrame(schema=RQ_MIN_SCHEMA)
        data = filter_to_stock_universe(data, rq_code_list)
        if data.is_empty():
            return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    data = data.with_columns(
        [
            pl.col("trade_time").cast(pl.Datetime("us")).alias("trade_time_dt"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ]
    ).with_columns(
        [
            pl.col("trade_time_dt").dt.date().alias("trading_date"),
            (
                pl.col("trade_time_dt").dt.hour().cast(pl.Int32) * 60
                + pl.col("trade_time_dt").dt.minute().cast(pl.Int32)
            ).alias("minute_of_day"),
        ]
    )

    if allowed_dates is not None:
        data = data.filter(pl.col("trading_date").is_in(list(allowed_dates)))
        if data.is_empty():
            return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    in_morning = pl.col("minute_of_day").is_between(
        MORNING_FIRST_TRADE_MINUTE,
        MORNING_END_MINUTE,
    )
    in_afternoon = pl.col("minute_of_day").is_between(
        AFTERNOON_FIRST_TRADE_MINUTE,
        AFTERNOON_END_MINUTE,
    )
    data = data.filter(in_morning | in_afternoon).with_columns(
        [
            pl.when(in_morning)
            .then(
                MORNING_START_MINUTE
                + (
                    (pl.col("minute_of_day") - MORNING_FIRST_TRADE_MINUTE)
                    // MINUTE_BAR_SIZE
                    + 1
                )
                * MINUTE_BAR_SIZE
            )
            .otherwise(
                AFTERNOON_START_MINUTE
                + (
                    (pl.col("minute_of_day") - AFTERNOON_FIRST_TRADE_MINUTE)
                    // MINUTE_BAR_SIZE
                    + 1
                )
                * MINUTE_BAR_SIZE
            )
            .alias("bucket_end_minute")
        ]
    )
    if data.is_empty():
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    data = data.with_columns(
        (
            pl.datetime(
                pl.col("trading_date").dt.year(),
                pl.col("trading_date").dt.month(),
                pl.col("trading_date").dt.day(),
            )
            + pl.duration(minutes=pl.col("bucket_end_minute"))
        ).alias("datetime")
    ).sort(["order_book_id", "trading_date", "trade_time_dt"])

    bars = (
        data.group_by(["order_book_id", "trading_date", "datetime"], maintain_order=True)
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .sort(["order_book_id", "trading_date", "datetime"])
    )

    first_session_bars = bars.with_columns(
        (
            pl.col("datetime").dt.hour().cast(pl.Int32) * 60
            + pl.col("datetime").dt.minute().cast(pl.Int32)
        ).alias("bar_minute")
    ).filter(pl.col("bar_minute").is_in([9 * 60 + 45, 13 * 60 + 15]))

    snapshots = first_session_bars.with_columns(
        [
            (pl.col("datetime") - pl.duration(minutes=MINUTE_BAR_SIZE)).alias("datetime"),
            pl.col("open").alias("high"),
            pl.col("open").alias("low"),
            pl.col("open").alias("close"),
        ]
    ).select(bars.columns)

    result = pl.concat([bars, snapshots], how="vertical").unique(
        subset=["order_book_id", "trading_date", "datetime"],
        keep="first",
    )
    result = result.with_columns(
        convert_code_format(pl.col("order_book_id"), format="gm").alias("code")
    ).select(list(RQ_MIN_SCHEMA.keys()))
    return align_schema(result, RQ_MIN_SCHEMA).sort(["trading_date", "code", "datetime"])


def validate_minute_expected_trading_days(
    minute_data: pl.DataFrame,
    allowed_dates: Iterable[dt.date] | None = None,
    rq_codes: list[str] | None = None,
) -> None:
    """阻止全市场历史分钟数据在缺少应有交易日时覆盖旧分区。"""
    if not allowed_dates or not rq_codes or len(rq_codes) < MINUTE_MARKET_GUARD_MIN_CODES:
        return

    expected_dates = {to_date(value) for value in allowed_dates}
    present_dates = (
        set()
        if minute_data.is_empty()
        else {to_date(value) for value in minute_data["trading_date"].unique().to_list()}
    )
    missing_dates = sorted(expected_dates - present_dates)
    historical_missing = [value for value in missing_dates if value < dt.date.today()]
    if historical_missing:
        details = ", ".join(to_date_str(value) for value in historical_missing)
        raise RuntimeError(f"minute data missing trading days: {details}")
    if missing_dates:
        logging.warning(
            "分钟线交易日暂未返回数据（可能为当日盘中或数据源尚未更新）: %s",
            ", ".join(to_date_str(value) for value in missing_dates),
        )


def validate_day_expected_trading_days(
    day_data: pl.DataFrame,
    allowed_dates: Iterable[dt.date] | None = None,
    rq_codes: list[str] | None = None,
) -> None:
    """阻止全市场历史日线在缺少应有交易日时覆盖旧分区。"""
    if not allowed_dates or not rq_codes or len(rq_codes) < DAY_MARKET_GUARD_MIN_CODES:
        return

    expected_dates = {to_date(value) for value in allowed_dates}
    present_dates = (
        set()
        if day_data.is_empty()
        else {to_date(value) for value in day_data["trading_date"].unique().to_list()}
    )
    missing_dates = sorted(expected_dates - present_dates)
    historical_missing = [value for value in missing_dates if value < dt.date.today()]
    if historical_missing:
        details = ", ".join(to_date_str(value) for value in historical_missing)
        raise RuntimeError(f"day data missing trading days: {details}")
    if missing_dates:
        logging.warning(
            "日线交易日暂未返回数据（可能为当日盘中或数据源尚未更新）: %s",
            ", ".join(to_date_str(value) for value in missing_dates),
        )


def validate_day_market_coverage(
    day_data: pl.DataFrame,
    rq_codes: list[str] | None = None,
) -> None:
    """校验全市场日线的每日股票覆盖度及标准输出字段。"""
    if day_data.is_empty() or not rq_codes or len(rq_codes) < DAY_MARKET_GUARD_MIN_CODES:
        return

    minimum_codes = max(
        DAY_MARKET_GUARD_MIN_CODES,
        math.ceil(len(set(rq_codes)) * MARKET_GUARD_MIN_RATIO),
    )
    coverage = day_data.group_by("trading_date").agg(
        pl.col("code").n_unique().alias("code_count")
    ).sort("trading_date")
    bad_days = coverage.filter(pl.col("code_count") < minimum_codes)
    if not bad_days.is_empty():
        details = "; ".join(
            f"{row['trading_date']}: {row['code_count']} codes"
            for row in bad_days.iter_rows(named=True)
        )
        raise RuntimeError(
            "day coverage too low: "
            f"{details}; expected at least {minimum_codes} stocks "
            "for all-market update"
        )

    missing_columns = set(RQ_DAY_SCHEMA) - set(day_data.columns)
    if missing_columns:
        raise RuntimeError(f"day schema invalid: missing columns {sorted(missing_columns)}")


def validate_minute_market_coverage(
    minute_data: pl.DataFrame,
    rq_codes: list[str] | None = None,
) -> None:
    """校验全市场分钟线的每日覆盖度、每股 Bar 数量和 snapshot 口径。"""
    if minute_data.is_empty() or not rq_codes:
        return
    if len(rq_codes) < MINUTE_MARKET_GUARD_MIN_CODES:
        return

    minimum_codes = max(
        MINUTE_MARKET_GUARD_MIN_CODES,
        math.ceil(len(set(rq_codes)) * MARKET_GUARD_MIN_RATIO),
    )
    coverage = minute_data.group_by("trading_date").agg(
        [
            pl.col("code").n_unique().alias("code_count"),
            pl.len().alias("row_count"),
        ]
    ).sort("trading_date")
    bad_days = coverage.filter(pl.col("code_count") < minimum_codes)
    if not bad_days.is_empty():
        details = "; ".join(
            f"{row['trading_date']}: {row['code_count']} codes, {row['row_count']} rows"
            for row in bad_days.iter_rows(named=True)
        )
        raise RuntimeError(
            "minute coverage too low: "
            f"{details}; expected at least {minimum_codes} stocks "
            f"for all-market update and {MINUTE_EXPECTED_BARS_PER_CODE} bars per stock"
        )

    bar_counts = minute_data.group_by(["trading_date", "code"]).agg(
        [
            pl.len().alias("row_count"),
            pl.col("datetime").n_unique().alias("bar_count"),
        ]
    )
    incomplete = bar_counts.filter(
        (pl.col("row_count") != MINUTE_EXPECTED_BARS_PER_CODE)
        | (pl.col("bar_count") != MINUTE_EXPECTED_BARS_PER_CODE)
    )
    if not incomplete.is_empty():
        samples = incomplete.sort(["trading_date", "code"]).head(5)
        details = "; ".join(
            f"{row['trading_date']} {row['code']}: "
            f"{row['bar_count']} unique bars/{row['row_count']} rows"
            for row in samples.iter_rows(named=True)
        )
        raise RuntimeError(
            "minute bar count incomplete: "
            f"{details}; {incomplete.height} stock-days affected; "
            f"expected exactly {MINUTE_EXPECTED_BARS_PER_CODE} bars"
        )

    expected_minutes = [
        9 * 60 + 30,
        *range(9 * 60 + 45, 11 * 60 + 31, 15),
        13 * 60,
        *range(13 * 60 + 15, 15 * 60 + 1, 15),
    ]
    minute_of_day = (
        pl.col("datetime").dt.hour().cast(pl.Int32) * 60
        + pl.col("datetime").dt.minute().cast(pl.Int32)
    )
    unexpected_times = minute_data.filter(~minute_of_day.is_in(expected_minutes))
    if not unexpected_times.is_empty():
        raise RuntimeError(
            "minute timestamps invalid: found bars outside the right-aligned schedule"
        )

    _validate_minute_snapshots(minute_data)


def _validate_minute_snapshots(minute_data: pl.DataFrame) -> None:
    """校验上午、下午 snapshot 与对应首根完整 Bar 的复制规则。"""
    keys = ["trading_date", "code"]
    value_columns = ["open", "high", "low", "close", "volume"]
    minute_of_day = (
        pl.col("datetime").dt.hour().cast(pl.Int32) * 60
        + pl.col("datetime").dt.minute().cast(pl.Int32)
    )

    for snapshot_minute, first_bar_minute in [(9 * 60 + 30, 9 * 60 + 45), (13 * 60, 13 * 60 + 15)]:
        snapshot = minute_data.filter(minute_of_day == snapshot_minute).select(
            [
                *keys,
                *[pl.col(column).alias(f"snapshot_{column}") for column in value_columns],
            ]
        )
        first_bar = minute_data.filter(minute_of_day == first_bar_minute).select(
            [
                *keys,
                pl.col("open").alias("first_open"),
                pl.col("volume").alias("first_volume"),
            ]
        )
        comparison = snapshot.join(first_bar, on=keys, how="full", coalesce=True)
        invalid = comparison.filter(
            pl.any_horizontal(
                [
                    pl.col(f"snapshot_{column}").is_null()
                    for column in value_columns
                ]
                + [
                    pl.col("first_open").is_null(),
                    pl.col("first_volume").is_null(),
                    *[
                        (pl.col(f"snapshot_{column}") - pl.col("first_open")).abs()
                        > 1e-6
                        for column in ["open", "high", "low", "close"]
                    ],
                    (pl.col("snapshot_volume") - pl.col("first_volume")).abs() > 1e-6,
                ]
            )
        )
        if not invalid.is_empty():
            raise RuntimeError(
                "minute snapshot invalid: "
                f"{invalid.height} stock-days violate the {snapshot_minute}-minute rule"
            )


def run_ddb_quality_gate(source) -> bool:
    """用固定小样本检查 DDB 标准日线是否存在系统性污染。"""
    rq_codes = source.get_stock_universe()
    if not rq_codes:
        logging.error("质量门失败: 股票池为空")
        return False

    trade_sample = source.fetch_day_range(
        dt.date(2021, 1, 4),
        dt.date(2021, 1, 4),
        rq_codes=rq_codes,
    )
    if trade_sample.is_empty():
        logging.error("质量门失败: 2021-01-04 股票行情为空")
        return False

    bad_codes = {"SHSE.000001", "SHSE.H50066", "SHSE.H50069", "SZSE.980001"}
    if bad_codes & set(trade_sample["code"].to_list()):
        logging.error("质量门失败: 样本交易日仍含非股票代码")
        return False

    holiday_sample = source.fetch_day_range(
        dt.date(2021, 2, 11),
        dt.date(2021, 2, 11),
        rq_codes=rq_codes,
    )
    if not holiday_sample.is_empty():
        logging.error("质量门失败: 2021-02-11 节假日仍有股票行情")
        return False

    required_columns = {"code", "trading_date", "open", "close", "adj_factor"}
    missing_columns = required_columns - set(trade_sample.columns)
    if missing_columns:
        logging.error("质量门失败: 样本缺少字段 %s", sorted(missing_columns))
        return False

    logging.info("DDB 质量门通过: 样本无非股票污染，节假日为空，字段完整")
    return True


def run_minute_quality_gate(source) -> bool:
    """抽查单票单日的右对齐时间戳和两个 session snapshot。"""
    sample_date = dt.date(2021, 1, 4)
    sample_rq_code = "000001.XSHE"
    sample_gm_code = "SZSE.000001"
    minute_sample = source.fetch_minute_range(
        sample_date,
        sample_date,
        allowed_dates={sample_date},
        rq_codes=[sample_rq_code],
    ).filter(pl.col("code") == sample_gm_code)
    if minute_sample.height != MINUTE_EXPECTED_BARS_PER_CODE:
        logging.error(
            "分钟质量门失败: DDB 合成样本行数为 %s，期望 %s",
            minute_sample.height,
            MINUTE_EXPECTED_BARS_PER_CODE,
        )
        return False

    gm_partition = (
        Path(data_dir("15min_stock_data_right_dir"))
        / f"trading_date={to_date_str(sample_date)}"
    )
    gm_files = sorted(gm_partition.glob("*.parquet"))
    if not gm_files:
        logging.error("分钟质量门失败: 找不到 GM 右对齐样本 %s", gm_partition)
        return False

    compare_columns = ["open", "high", "low", "close", "volume"]
    gm_sample = (
        pl.read_parquet([str(path) for path in gm_files])
        .filter(pl.col("code") == sample_gm_code)
        .select(["datetime", *compare_columns])
        .sort("datetime")
    )
    if gm_sample.height != MINUTE_EXPECTED_BARS_PER_CODE:
        logging.error(
            "分钟质量门失败: GM 右对齐样本行数为 %s，期望 %s",
            gm_sample.height,
            MINUTE_EXPECTED_BARS_PER_CODE,
        )
        return False

    ddb_sample = minute_sample.select(["datetime", *compare_columns]).sort("datetime")
    if ddb_sample["datetime"].to_list() != gm_sample["datetime"].to_list():
        logging.error("分钟质量门失败: DDB 合成时间戳与 GM 右对齐样本不一致")
        return False

    for snapshot_time, first_bar_time in [
        (
            dt.datetime.combine(sample_date, dt.time(9, 30)),
            dt.datetime.combine(sample_date, dt.time(9, 45)),
        ),
        (
            dt.datetime.combine(sample_date, dt.time(13, 0)),
            dt.datetime.combine(sample_date, dt.time(13, 15)),
        ),
    ]:
        snapshot_rows = ddb_sample.filter(pl.col("datetime") == snapshot_time)
        first_bar_rows = ddb_sample.filter(pl.col("datetime") == first_bar_time)
        if snapshot_rows.height != 1 or first_bar_rows.height != 1:
            logging.error(
                "分钟质量门失败: 缺少 snapshot 或首根完整 Bar: %s / %s",
                snapshot_time,
                first_bar_time,
            )
            return False

        snapshot = snapshot_rows.row(0, named=True)
        first_bar = first_bar_rows.row(0, named=True)
        expected = {
            "open": first_bar["open"],
            "high": first_bar["open"],
            "low": first_bar["open"],
            "close": first_bar["open"],
            "volume": first_bar["volume"],
        }
        for column, expected_value in expected.items():
            if abs(snapshot[column] - expected_value) > 1e-6:
                logging.error(
                    "分钟质量门失败: %s snapshot 字段 %s=%s，期望 %s",
                    snapshot_time,
                    column,
                    snapshot[column],
                    expected_value,
                )
                return False

    joined = gm_sample.rename(
        {column: f"gm_{column}" for column in compare_columns}
    ).join(ddb_sample, on="datetime", how="inner")
    max_diff = joined.select(
        [
            (pl.col(f"gm_{column}") - pl.col(column)).abs().max().alias(column)
            for column in compare_columns
        ]
    ).to_dicts()[0]
    bad_diff = {
        column: value
        for column, value in max_diff.items()
        if value is not None and value > 1e-6
    }
    if bad_diff:
        logging.warning(
            "分钟质量门提示: RQ/DDB 1min 聚合值与 GM 15min 样本存在源数据差异 %s",
            bad_diff,
        )
    logging.info("分钟质量门通过: DDB 1min 合成时间戳与 snapshot 规则符合 GM 右对齐口径")
    return True


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
    """先写临时分区再替换正式分区，返回实际写入行数。"""
    output = prepare_for_write(data, save_dir, schema, mode)
    if output.is_empty():
        logging.info("%s 没有需要写入的新数据", save_dir)
        return 0

    staging_root, replacements = _stage_partitioned_output(output, save_dir)
    _commit_staged_partitions([staging_root], replacements)
    logging.info("%s 写入完成: %s 行", save_dir, output.height)
    return output.height


def write_rq_day_and_adj_partitioned(
    day_data: pl.DataFrame,
    day_save_dir: str,
    adj_save_dir: str,
    mode: str,
) -> int:
    """在一次可回滚替换中同步写入标准日线和复权因子分区。"""
    day_output = prepare_for_write(day_data, day_save_dir, RQ_DAY_SCHEMA, mode)
    adj_output = prepare_for_write(
        day_data.select(["code", "trading_date", "adj_factor"]),
        adj_save_dir,
        RQ_ADJ_SCHEMA,
        mode,
    )
    outputs = [
        (day_output, day_save_dir),
        (adj_output, adj_save_dir),
    ]
    outputs = [(output, save_dir) for output, save_dir in outputs if not output.is_empty()]
    if not outputs:
        logging.info("%s / %s 没有需要写入的新数据", day_save_dir, adj_save_dir)
        return 0

    staging_roots: list[str] = []
    replacements: list[tuple[str, str]] = []
    try:
        for output, save_dir in outputs:
            staging_root, staged_replacements = _stage_partitioned_output(output, save_dir)
            staging_roots.append(staging_root)
            replacements.extend(staged_replacements)
    except Exception:
        for staging_root in staging_roots:
            shutil.rmtree(staging_root, ignore_errors=True)
        raise

    _commit_staged_partitions(staging_roots, replacements)
    total_rows = sum(output.height for output, _ in outputs)
    logging.info(
        "%s / %s 同步写入完成: %s 行",
        day_save_dir,
        adj_save_dir,
        total_rows,
    )
    return total_rows


def _stage_partitioned_output(
    output: pl.DataFrame,
    save_dir: str,
) -> tuple[str, list[tuple[str, str]]]:
    """在目标目录所在磁盘写临时分区，不接触已有正式分区。"""
    target_dir = data_dir(save_dir)
    os.makedirs(target_dir, exist_ok=True)
    # tempfile.mkdtemp() 在 Windows 上会创建受保护 ACL；若定时任务使用
    # 其他账号写入，移动后的正式分区可能对当前用户不可读。显式使用
    # os.makedirs() 让暂存目录继承数据目录 ACL。
    staging_root = os.path.join(target_dir, f".rq-staging-{uuid.uuid4().hex}")
    os.makedirs(staging_root)
    try:
        output.write_parquet(staging_root, partition_by=["trading_date"])
        replacements = []
        for trade_date in output["trading_date"].unique().sort().to_list():
            partition_name = f"trading_date={to_date_str(trade_date)}"
            staged_partition = os.path.join(staging_root, partition_name)
            target_partition = os.path.join(target_dir, partition_name)
            if not os.path.isdir(staged_partition):
                raise RuntimeError(f"临时分区写入不完整: {staged_partition}")
            replacements.append((staged_partition, target_partition))
        return staging_root, replacements
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _commit_staged_partitions(
    staging_roots: list[str],
    replacements: list[tuple[str, str]],
) -> None:
    """替换全部已暂存分区；任一替换失败时恢复本轮已移动的旧分区。"""
    changes: list[dict[str, str | bool | None]] = []
    try:
        for staged_partition, target_partition in replacements:
            backup_partition = None
            change: dict[str, str | bool | None] = {
                "target": target_partition,
                "backup": None,
                "old_moved": False,
                "new_moved": False,
            }
            changes.append(change)

            if os.path.exists(target_partition):
                backup_partition = f"{target_partition}.backup-{uuid.uuid4().hex}"
                change["backup"] = backup_partition
                os.replace(target_partition, backup_partition)
                change["old_moved"] = True

            os.replace(staged_partition, target_partition)
            change["new_moved"] = True
            logging.debug("替换分区: %s", target_partition)
    except Exception:
        for change in reversed(changes):
            target_partition = str(change["target"])
            backup_partition = change["backup"]
            if change["new_moved"] and os.path.exists(target_partition):
                _remove_path(target_partition)
            if change["old_moved"] and backup_partition and os.path.exists(str(backup_partition)):
                os.replace(str(backup_partition), target_partition)
        raise
    else:
        for change in changes:
            backup_partition = change["backup"]
            if backup_partition and os.path.exists(str(backup_partition)):
                try:
                    _remove_path(str(backup_partition))
                except Exception as exc:
                    # 正式分区已全部提交，备份清理失败不能把成功写入伪装成失败。
                    # 保留备份目录便于人工清理，并记录其精确路径。
                    logging.warning("旧分区备份清理失败，已保留 %s: %s", backup_partition, exc)
    finally:
        for staging_root in staging_roots:
            shutil.rmtree(staging_root, ignore_errors=True)


def _remove_path(path: str) -> None:
    """删除已解析出的单个临时或备份路径。"""
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
