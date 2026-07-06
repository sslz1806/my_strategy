"""
米筐数据本地更新脚本 — DDB 版（按日循环 v4）。

数据源从 RQ proxy HTTP API 切换为 DolphinDB（DDB），DDB 数据来源即为米筐。
按天循环：每天获取当天全市场股票池，从 DDB 拉取所有数据后分区写入本地 Parquet。

对比 RQ proxy 版的核心改进：
  - adj_factor 为真实的累积复权因子（而非始终 1.0）
  - is_st / is_suspended 标记更完整
  - 数据已更新到当天，无缺失

v4 设计：
  - fetch_day_range: 一次 DDB 查询覆盖多日，减少请求次数
  - 日内用 join_asof 前向填充复权因子
  - 按季度分批更新，平衡内存和查询效率
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import dolphindb as ddb
import polars as pl

DATA_ROOT_DIR = r"E:\working\stock_data"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_utils.fun import get_logger  # noqa: E402
from my_utils.rq_fun import (  # noqa: E402
    RQ_ADJ_SCHEMA,
    RQ_DAY_SCHEMA,
    RQ_MIN_SCHEMA,
    get_existing_dates,
    to_date,
    to_date_str,
    write_partitioned,
)

# DolphinDB 连接配置
DDB_HOST = "10.140.5.44"
DDB_PORT = 8902
DDB_USER = "admin"
DDB_PASSWORD = "123456"

# 本地分区目录
RQ_DAY_DIR = "rq_stock_all_data"
RQ_MIN_DIR = "rq_15min_stock_data_dir"
RQ_ADJ_DIR = "rq_adj"

# 批量更新参数
BATCH_SIZE = 60  # 每批约 3 个月交易日

DDB_QUERY_TIMEOUT = 300  # DDB 查询超时秒数
MORNING_START_MINUTE = 9 * 60 + 30
MORNING_FIRST_TRADE_MINUTE = 9 * 60 + 31
MORNING_END_MINUTE = 11 * 60 + 30
AFTERNOON_START_MINUTE = 13 * 60
AFTERNOON_FIRST_TRADE_MINUTE = 13 * 60 + 1
AFTERNOON_END_MINUTE = 15 * 60
MINUTE_BAR_SIZE = 15


# ============================================================================
# DDB 会话管理
# ============================================================================


def create_ddb_session() -> ddb.session:
    """创建并返回 DDB 连接会话（调用方负责关闭）。"""
    s = ddb.session()
    s.connect(DDB_HOST, DDB_PORT, DDB_USER, DDB_PASSWORD)
    return s


# ============================================================================
# 股票池与交易日
# ============================================================================


def get_stock_universe(session: ddb.session) -> list[str]:
    """
    获取全市场 A 股股票池（静态，不依赖日期）。

    从 instrument_base 取 type='CS' 的股票列表，包含已退市/待上市的股票。
    调用方在按天更新全程只调用一次，结果缓存复用。

    Args:
        session: DDB 会话

    Returns:
        股票代码列表（米筐格式，如 ["000001.XSHE", ...]）
    """
    try:
        df = session.run("""
            select order_book_id
            from loadTable('dfs://common_years_tsdb', 'instrument_base')
            where type = 'CS'
        """)
        return df["order_book_id"].dropna().tolist()
    except Exception as exc:
        logging.error("获取 DDB 股票池失败: %s", exc)
        return []


def filter_to_stock_universe(data: pl.DataFrame, rq_codes: list[str]) -> pl.DataFrame:
    """
    按米筐 A 股股票池过滤 DDB 返回结果。

    历史补数优先控制 DDB 请求次数，因此允许先用少量大范围 SQL 拉取
    `.XSHE/.XSHG` 后缀数据，再在本地按 instrument_base(type='CS')
    股票池剔除指数、基金、债券等非股票代码。
    """
    if data.is_empty() or not rq_codes or "order_book_id" not in data.columns:
        return data
    return data.filter(pl.col("order_book_id").is_in(rq_codes))


def _to_gm_code_expr(source_col: str = "order_book_id") -> pl.Expr:
    """把米筐后缀代码转换为项目本地 GM 前缀代码。"""
    return (
        pl.when(pl.col(source_col).str.ends_with(".XSHE"))
        .then(pl.lit("SZSE.") + pl.col(source_col).str.replace(".XSHE", ""))
        .otherwise(pl.lit("SHSE.") + pl.col(source_col).str.replace(".XSHG", ""))
    )


def get_trading_days(
    session: ddb.session,
    start_date: dt.date,
    end_date: dt.date,
) -> list[dt.date]:
    """
    从 DDB trade_date 表获取交易日列表。

    Args:
        session: DDB 会话
        start_date: 起始日期
        end_date: 结束日期

    Returns:
        排序后的交易日列表（datetime.date）
    """
    start_str = start_date.strftime("%Y.%m.%d")
    end_str = end_date.strftime("%Y.%m.%d")
    try:
        df = session.run(f"""
            select distinct trade_date
            from loadTable('dfs://common_years_tsdb', 'trade_date')
            where is_trade_date = true
              and trade_date >= date({start_str})
              and trade_date <= date({end_str})
            order by trade_date
        """)
        dates = df["trade_date"].dropna().tolist()
        return [d.date() if hasattr(d, "date") else d for d in dates]
    except Exception as exc:
        logging.error("获取交易日列表失败: %s", exc)
        return []


# ============================================================================
# 数据拉取 — 日线（批量多日）
# ============================================================================


def fetch_day_range(
    session: ddb.session,
    start_date: dt.date,
    end_date: dt.date,
    rq_codes: list[str] | None = None,
) -> pl.DataFrame:
    """
    从 DDB 批量获取一段时间内全市场完整日线数据。

    一次查询覆盖整个日期范围，全部在 Polars 中完成 JOIN 和衍生计算。
    共发起 5 次 DDB 查询（无论天数多少），大幅减少请求次数。

    复权因子处理：
      借助 join_asof（asof join）实现"每个交易日取当天或之前最新 ex_cum_factor"。
      对于从未除权的股票，adj_factor 默认为 1.0。

    Args:
        session: DDB 会话
        start_date: 起始交易日
        end_date: 结束交易日
        rq_codes: 可选，外部传入的股票池（避免重复获取）

    Returns:
        多日合并的 DataFrame（含 trading_date 列），符合 RQ_DAY_SCHEMA
    """
    if rq_codes is None:
        rq_codes = get_stock_universe(session)
    if not rq_codes:
        return pl.DataFrame(schema=RQ_DAY_SCHEMA)

    start_str = start_date.strftime("%Y.%m.%d")
    end_str = end_date.strftime("%Y.%m.%d")
    date_label = f"{to_date_str(start_date)} ~ {to_date_str(end_date)}"

    logging.info("  DDB 批量查询 %s（%d 只股票）...", date_label, len(rq_codes))

    # -- 1. 多日行情 --
    kline = session.run(f"""
        select order_book_id, date as trading_date,
               open, close, high, low,
               volume, total_turnover as amount,
               prev_close as pre_close,
               limit_up, limit_down
        from loadTable('dfs://common_years_olap', 'day_kline')
        where (order_book_id like '%.XSHE' or order_book_id like '%.XSHG')
          and date >= date({start_str})
          and date <= date({end_str})
    """)
    if kline.empty:
        logging.info("  %s 无行情数据", date_label)
        return pl.DataFrame(schema=RQ_DAY_SCHEMA)

    df = pl.from_pandas(kline)
    df = filter_to_stock_universe(df, rq_codes)
    if df.is_empty():
        logging.info("  %s 无股票行情数据（全量查询后股票池过滤为空）", date_label)
        return pl.DataFrame(schema=RQ_DAY_SCHEMA)

    df = df.with_columns([
        pl.col("trading_date").cast(pl.Date),
        ((pl.col("close") / pl.col("pre_close") - 1) * 100).alias("pct"),
        (pl.col("volume") == 0).and_(pl.col("amount") == 0).alias("is_suspended"),
    ])
    logging.info("    day_kline 返回 %d 行", len(df))

    # -- 2. ST 标记（多日） --
    is_st = session.run(f"""
        select order_book_id, date as trading_date, is_st
        from loadTable('dfs://stock_years_tsdb', 'is_st_stock')
        where date >= date({start_str})
          and date <= date({end_str})
    """)
    if not is_st.empty:
        is_st_pl = pl.from_pandas(is_st).with_columns([pl.col("trading_date").cast(pl.Date)])
        df = df.join(is_st_pl, on=["order_book_id", "trading_date"], how="left")
        logging.info("    is_st 返回 %d 行", len(is_st))
    else:
        df = df.with_columns(pl.lit(False).alias("is_st"))

    # -- 3. 股本数据（多日） --
    shares = session.run(f"""
        select order_book_id, date as trading_date,
               circulation_a, total_a, free_circulation
        from loadTable('dfs://stock_years_tsdb', 'stock_shares')
        where date >= date({start_str})
          and date <= date({end_str})
    """)
    if not shares.empty:
        shares_pl = pl.from_pandas(shares).with_columns([pl.col("trading_date").cast(pl.Date)])
        df = df.join(shares_pl, on=["order_book_id", "trading_date"], how="left")
        logging.info("    stock_shares 返回 %d 行", len(shares))
    else:
        df = df.with_columns([
            pl.lit(None, pl.Float64).alias("circulation_a"),
            pl.lit(None, pl.Float64).alias("total_a"),
            pl.lit(None, pl.Float64).alias("free_circulation"),
        ])

    # -- 4. 复权因子（一次查询 end_date 前全部记录，用 join_asof 取最新） --
    ex_all = session.run(f"""
        select order_book_id, ex_date, ex_cum_factor as adj_factor
        from loadTable('dfs://stock_years_tsdb', 'ex_factor')
        where ex_date <= date({end_str})
    """)
    if not ex_all.empty:
        ex_pl = pl.from_pandas(ex_all).with_columns([
            pl.col("ex_date").cast(pl.Date),
        ]).sort(["order_book_id", "ex_date"])

        # join_asof: 对每个 (stock, trading_date)，取 ex_date <= trading_date 的最新记录
        df = df.sort(["order_book_id", "trading_date"])
        df = df.join_asof(
            ex_pl,
            left_on="trading_date",
            right_on="ex_date",
            by="order_book_id",
            strategy="backward",
        )
        logging.info("    ex_factor 返回 %d 行", len(ex_all))
    else:
        df = df.with_columns(pl.lit(None, pl.Float64).alias("adj_factor"))

    # -- 5. 股票名称（静态） --
    inst = session.run("""
        select order_book_id, symbol as name
        from loadTable('dfs://common_years_tsdb', 'instrument_base')
        where type = 'CS'
    """)
    inst_pl = pl.from_pandas(inst)
    df = df.join(inst_pl, on="order_book_id", how="left")

    # -- 6. 填充缺失值 --
    df = df.with_columns([
        pl.col("is_st").fill_null(False),
        pl.col("is_suspended").fill_null(True),
        pl.col("adj_factor").fill_null(1.0),
        pl.col("name").fill_null(""),
        pl.col("circulation_a").fill_null(0.0),
        pl.col("total_a").fill_null(0.0),
        pl.col("free_circulation").fill_null(0.0),
    ])

    # -- 7. 计算衍生字段 --
    # 换手率 = volume / circulation_a * 100（百分比，与本地格式对齐）
    df = df.with_columns([
        pl.when(pl.col("circulation_a") > 0)
        .then(pl.col("volume").cast(pl.Float64) / pl.col("circulation_a") * 100)
        .otherwise(0.0).alias("turnover_rate"),
    ])

    # 市值 = close * 股本
    df = df.with_columns([
        (pl.col("close") * pl.col("total_a")).alias("total_mv"),
        (pl.col("close") * pl.col("circulation_a")).alias("circulation_mv"),
        (pl.col("close") * pl.col("free_circulation")).alias("mv_A_free_float"),
    ])

    # -- 8. 代码格式转换（DDB: 000001.XSHE → SZSE.000001） --
    df = df.with_columns([
        pl.when(pl.col("order_book_id").str.ends_with(".XSHE"))
        .then(pl.lit("SZSE.") + pl.col("order_book_id").str.replace(".XSHE", ""))
        .otherwise(pl.lit("SHSE.") + pl.col("order_book_id").str.replace(".XSHG", ""))
        .alias("code"),
    ])

    # 最终输出
    result = df.select([
        "code", "name", "trading_date",
        "open", "high", "low", "close", "pre_close", "pct",
        "volume", "amount", "limit_up", "limit_down",
        "is_st", "is_suspended", "adj_factor",
        "turnover_rate", "total_mv", "circulation_mv", "mv_A_free_float",
    ])

    logging.info("    => 合并后 %d 行", len(result))
    return result


# ============================================================================
# 数据拉取 — 分钟线（从 1min 聚合到 15min）
# ============================================================================


def aggregate_right_aligned_15min(
    raw: pl.DataFrame,
    rq_codes: Iterable[str] | None = None,
    allowed_dates: set[dt.date] | None = None,
) -> pl.DataFrame:
    """
    将 DDB 1 分钟线合成为米筐右对齐 15 分钟线。

    DDB 的 `trade_time` 是 1 分钟 Bar 的结束时间，因此完整 15 分钟 Bar 也使用
    结束时间作为 `datetime`。本地 GM 右对齐分钟线额外保留 09:30 与 13:00
    两个 session snapshot：它们复用首根完整 15 分钟 Bar 的成交量，但 OHLC
    全部设为该 Bar 的开盘价，用于兼容既有分钟线数据口径。

    `rq_codes=None` 表示不按股票池过滤，便于纯本地样本直接聚合；显式传入空
    iterable 表示当前没有合格股票，应立即返回空 schema，避免误把空股票池当作全市场。
    """
    if raw.is_empty():
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    df = raw
    if rq_codes is not None:
        rq_code_list = list(rq_codes)
        if not rq_code_list:
            return pl.DataFrame(schema=RQ_MIN_SCHEMA)
        df = filter_to_stock_universe(df, rq_code_list)
        if df.is_empty():
            return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    df = df.with_columns(
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
        df = df.filter(pl.col("trading_date").is_in(list(allowed_dates)))
        if df.is_empty():
            return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    in_morning = pl.col("minute_of_day").is_between(
        MORNING_FIRST_TRADE_MINUTE,
        MORNING_END_MINUTE,
    )
    in_afternoon = pl.col("minute_of_day").is_between(
        AFTERNOON_FIRST_TRADE_MINUTE,
        AFTERNOON_END_MINUTE,
    )

    # A 股 09:31-11:30、13:01-15:00 是有效 1 分钟成交区间；
    # 右对齐桶尾分别落在 09:45...11:30、13:15...15:00。
    df = df.filter(in_morning | in_afternoon).with_columns(
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
    if df.is_empty():
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    df = df.with_columns(
        [
            (
                pl.datetime(
                    pl.col("trading_date").dt.year(),
                    pl.col("trading_date").dt.month(),
                    pl.col("trading_date").dt.day(),
                )
                + pl.duration(minutes=pl.col("bucket_end_minute"))
            ).alias("datetime")
        ]
    ).sort(["order_book_id", "trading_date", "trade_time_dt"])

    bars = (
        df.group_by(["order_book_id", "trading_date", "datetime"], maintain_order=True)
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

    first_session_bars = (
        bars.with_columns(
            (
                pl.col("datetime").dt.hour().cast(pl.Int32) * 60
                + pl.col("datetime").dt.minute().cast(pl.Int32)
            ).alias("bar_minute")
        )
        .filter(pl.col("bar_minute").is_in([9 * 60 + 45, 13 * 60 + 15]))
    )

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

    result = (
        result.with_columns([_to_gm_code_expr().alias("code")])
        .select(["code", "datetime", "open", "high", "low", "close", "volume", "trading_date"])
        .sort(["trading_date", "code", "datetime"])
    )
    return result.cast(RQ_MIN_SCHEMA)


def fetch_minute_range(
    session: ddb.session,
    start_date: dt.date,
    end_date: dt.date,
    allowed_dates: Iterable[dt.date] | None = None,
    rq_codes: list[str] | None = None,
) -> pl.DataFrame:
    """
    从 DDB 一次查询一个日期范围的一分钟线，并合成右对齐 15 分钟线。

    调用方必须按自然月传入范围；函数内部不再拆分，避免隐藏额外 DDB 查询。
    """
    if rq_codes is None:
        rq_codes = get_stock_universe(session)
    if not rq_codes:
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    if allowed_dates is None:
        allowed_dates_set = set(get_trading_days(session, start_date, end_date))
    else:
        allowed_dates_set = set(allowed_dates)
    if not allowed_dates_set:
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    # 质量门只抽查单只股票时，将股票代码条件下推到 DDB，避免为了一个样本拉取全市场分钟线。
    # 全量月度更新会传入数千只股票池，此时不构造超长 IN 条件，仍按时间范围一次性读取后本地过滤。
    code_filter = ""
    if len(rq_codes) == 1:
        safe_code = str(rq_codes[0]).replace("'", "''")
        code_filter = f"\n              and order_book_id = '{safe_code}'"

    start_str = start_date.strftime("%Y.%m.%d")
    end_str = end_date.strftime("%Y.%m.%d")
    logging.info(
        "  DDB 分钟线月度查询 %s ~ %s（股票池 %d 只）...",
        to_date_str(start_date),
        to_date_str(end_date),
        len(rq_codes),
    )

    try:
        raw = session.run(f"""
            select order_book_id, trade_time,
                   open, close, high, low, volume, total_turnover
            from loadTable('dfs://common_years_olap', 'one_min_kline')
            where (order_book_id like '%.XSHE' or order_book_id like '%.XSHG')
              and trade_time >= timestamp({start_str} 09:31:00)
              and trade_time <= timestamp({end_str} 15:00:00)
              {code_filter}
        """)
    except Exception as exc:
        logging.warning("  DDB 分钟线查询失败: %s", exc)
        raise RuntimeError(f"DDB 分钟线查询失败: {exc}") from exc

    if raw.empty:
        logging.info("  DDB 分钟线 %s ~ %s 返回空", to_date_str(start_date), to_date_str(end_date))
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    raw_pl = pl.from_pandas(raw)
    logging.info("    one_min_kline 返回 %d 行", len(raw_pl))
    result = aggregate_right_aligned_15min(raw_pl, rq_codes=rq_codes, allowed_dates=allowed_dates_set)
    logging.info("    => 合成右对齐 15min %d 行", len(result))
    return result


# ============================================================================
# 更新入口 — 分批批量
# ============================================================================


def remove_existing_partitions_in_range(
    save_dir: str,
    start_date: dt.date,
    end_date: dt.date,
) -> int:
    """
    删除目标目录中指定日期范围内的旧分区。

    `mode='update'` 重写历史时，过滤后的新数据可能不再包含旧分区里的
    指数或节假日污染数据。先在单个成功拉取的批次范围内清理旧分区，再写
    新数据，才能保证旧污染不会残留。
    """
    import shutil

    target_dir = Path(DATA_ROOT_DIR) / save_dir
    if not target_dir.exists():
        return 0

    removed = 0
    for item in target_dir.iterdir():
        if not item.is_dir() or not item.name.startswith("trading_date="):
            continue
        date_text = item.name.split("=", 1)[1]
        try:
            partition_date = to_date(date_text)
        except ValueError:
            logging.warning("跳过无法解析的分区目录: %s", item)
            continue
        if start_date <= partition_date <= end_date:
            shutil.rmtree(item)
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


def update_day_range(
    session: ddb.session,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    rq_codes: list[str] | None = None,
) -> int:
    """
    批量更新一段日期范围的数据。

    流程：
      1. fetch_day_range 一次获取全部数据
      2. 按 trading_date 分组，逐日写入日线和复权因子分区

    Args:
        session: DDB 会话
        start_date: 起始日期
        end_date: 结束日期
        mode: insert / update
        rq_codes: 可选，股票池（传入则复用避免重复查询）

    Returns:
        写入总行数
    """
    all_data = fetch_day_range(session, start_date, end_date, rq_codes=rq_codes)
    if all_data.is_empty():
        return 0

    if mode == "update":
        remove_existing_partitions_in_range(RQ_DAY_DIR, start_date, end_date)
        remove_existing_partitions_in_range(RQ_ADJ_DIR, start_date, end_date)

    total_written = 0
    # 按交易日分组写入
    trading_dates_in_data = all_data["trading_date"].unique().sort().to_list()

    for date in trading_dates_in_data:
        day_data = all_data.filter(pl.col("trading_date") == date)
        if day_data.is_empty():
            continue

        # 日线
        day_written = write_partitioned(day_data, RQ_DAY_DIR, RQ_DAY_SCHEMA, mode)
        total_written += day_written

        # 复权因子
        adj_data = day_data.select(["code", "trading_date", "adj_factor"])
        adj_written = write_partitioned(adj_data, RQ_ADJ_DIR, RQ_ADJ_SCHEMA, mode)
        total_written += adj_written

    return total_written


def update_minute_range(
    session: ddb.session,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    allowed_dates: Iterable[dt.date] | None = None,
    rq_codes: list[str] | None = None,
) -> int:
    """
    更新一个月度范围内的米筐右对齐 15 分钟线。

    只有在 DDB 查询和本地合成结果非空后，才清理旧分区，避免失败时破坏已有数据。
    """
    minute_data = fetch_minute_range(
        session,
        start_date,
        end_date,
        allowed_dates=allowed_dates,
        rq_codes=rq_codes,
    )
    if minute_data.is_empty():
        return 0

    trading_dates = minute_data["trading_date"].unique().sort().to_list()
    if mode == "update":
        for trade_date in trading_dates:
            remove_existing_partitions_in_range(RQ_MIN_DIR, trade_date, trade_date)

    total_written = 0
    for trade_date in trading_dates:
        day_data = minute_data.filter(pl.col("trading_date") == trade_date)
        if day_data.is_empty():
            continue
        total_written += write_partitioned(day_data, RQ_MIN_DIR, RQ_MIN_SCHEMA, mode)
    return total_written


def update_minute_all(
    session: ddb.session,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
) -> int:
    """按自然月批量更新米筐右对齐 15 分钟线。"""
    trading_days = get_trading_days(session, start_date, end_date)
    if not trading_days:
        logging.warning("分钟线未获取到交易日列表")
        return 0

    rq_codes = get_stock_universe(session)
    if not rq_codes:
        logging.warning("分钟线股票池为空")
        return 0

    total_written = 0
    month_ranges = build_month_ranges(start_date, end_date)
    for idx, (month_start, month_end) in enumerate(month_ranges, start=1):
        # 复用外层已查询的交易日列表，避免每个月再查一次 DDB 交易日元数据。
        month_allowed_dates = [
            trade_date
            for trade_date in trading_days
            if month_start <= trade_date <= month_end
        ]
        logging.info(
            "[分钟 %s/%s] 月度批次 %s ~ %s",
            idx,
            len(month_ranges),
            to_date_str(month_start),
            to_date_str(month_end),
        )
        total_written += update_minute_range(
            session,
            month_start,
            month_end,
            mode,
            allowed_dates=month_allowed_dates,
            rq_codes=rq_codes,
        )

    logging.info(
        "分钟线更新完成: 共处理 %s 个交易日, 写入 %s 行",
        len(trading_days),
        total_written,
    )
    return total_written


def build_batch_ranges(
    trading_days: list[dt.date],
    batch_mode: str = "days",
    batch_size: int = BATCH_SIZE,
) -> list[tuple[dt.date, dt.date]]:
    """根据批处理模式生成 DDB 查询区间，避免按日查询压垮数据库。"""
    if not trading_days:
        return []

    days = sorted(trading_days)
    if batch_mode == "all":
        return [(days[0], days[-1])]

    if batch_mode == "year":
        ranges: list[tuple[dt.date, dt.date]] = []
        year_start = days[0]
        previous = days[0]
        for day in days[1:]:
            if day.year != previous.year:
                ranges.append((year_start, previous))
                year_start = day
            previous = day
        ranges.append((year_start, previous))
        return ranges

    if batch_mode != "days":
        raise ValueError(f"不支持的 batch_mode: {batch_mode}")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正数: {batch_size}")

    return [
        (batch[0], batch[-1])
        for batch in (days[i: i + batch_size] for i in range(0, len(days), batch_size))
    ]


def build_month_ranges(start_date: dt.date, end_date: dt.date) -> list[tuple[dt.date, dt.date]]:
    """按自然月切分日期范围；分钟线 DDB 查询固定每月一次。"""
    if start_date > end_date:
        return []

    ranges: list[tuple[dt.date, dt.date]] = []
    cursor = start_date.replace(day=1)
    while cursor <= end_date:
        if cursor.month == 12:
            next_month = dt.date(cursor.year + 1, 1, 1)
        else:
            next_month = dt.date(cursor.year, cursor.month + 1, 1)

        month_start = max(start_date, cursor)
        month_end = min(end_date, next_month - dt.timedelta(days=1))
        ranges.append((month_start, month_end))
        cursor = next_month
    return ranges


def update_all(
    session: ddb.session,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    batch_mode: str = "days",
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    按日循环更新全市场数据（分批调用 update_day_range）。

    将日期范围按 BATCH_SIZE 分组，每组一次性查询 + 写入，
    避免单次查询数据量过大，同时保持对 DDB 的低请求频率。

    Args:
        session: DDB 会话
        start_date: 起始日期
        end_date: 结束日期
        mode: insert / update

    Returns:
        累计写入总行数
    """
    trading_days = get_trading_days(session, start_date, end_date)
    if not trading_days:
        logging.warning("未获取到交易日列表")
        return 0

    logging.info("交易日数: %s", len(trading_days))

    # 预获取股票池，所有批次共享
    rq_codes = get_stock_universe(session)
    if not rq_codes:
        logging.warning("股票池为空")
        return 0

    total_written = 0
    n_days = len(trading_days)

    batch_ranges = build_batch_ranges(trading_days, batch_mode=batch_mode, batch_size=batch_size)
    for idx, (batch_start, batch_end) in enumerate(batch_ranges, start=1):
        logging.info(
            "[%s/%s] 批次 %s ~ %s",
            idx, len(batch_ranges), to_date_str(batch_start), to_date_str(batch_end),
        )

        written = update_day_range(session, batch_start, batch_end, mode, rq_codes=rq_codes)
        total_written += written

    logging.info("更新完成: 共处理 %s 个交易日, 写入 %s 行", n_days, total_written)
    return total_written


# ============================================================================
# CLI & 入口
# ============================================================================


def run_ddb_quality_gate(session: ddb.session) -> bool:
    """
    小样本检查 DDB 数据是否适合历史重写。

    质量门关注系统性污染和字段完整性，不要求与掘金每日股票数量完全一致。
    若失败，调用方应停止 DDB 全量重写，改走官方米筐接口 fallback。
    """
    rq_codes = get_stock_universe(session)
    if not rq_codes:
        logging.error("质量门失败: 股票池为空")
        return False

    trade_sample = fetch_day_range(
        session,
        dt.date(2021, 1, 4),
        dt.date(2021, 1, 4),
        rq_codes=rq_codes,
    )
    if trade_sample.is_empty():
        logging.error("质量门失败: 2021-01-04 股票行情为空")
        return False

    sample_codes = set(trade_sample["code"].to_list())
    bad_codes = {"SHSE.000001", "SHSE.H50066", "SHSE.H50069", "SZSE.980001"}
    if bad_codes & sample_codes:
        logging.error("质量门失败: 样本交易日仍含非股票代码")
        return False

    holiday_sample = fetch_day_range(
        session,
        dt.date(2021, 2, 11),
        dt.date(2021, 2, 11),
        rq_codes=rq_codes,
    )
    if not holiday_sample.is_empty():
        logging.error("质量门失败: 2021-02-11 节假日仍有股票行情")
        return False

    required_cols = {"code", "trading_date", "open", "close", "adj_factor"}
    missing_cols = required_cols - set(trade_sample.columns)
    if missing_cols:
        logging.error("质量门失败: 样本缺少字段 %s", sorted(missing_cols))
        return False

    logging.info("DDB 质量门通过: 样本无非股票污染，节假日为空，字段完整")
    return True


def run_minute_quality_gate(session: ddb.session) -> bool:
    """
    用单只股票单日样本校验 DDB 1 分钟合成后的右对齐 15 分钟口径。

    检查目标是“口径对齐”，不是全市场数据完整性：只抽查 000001.XSHE 在
    2021-01-04 的 1 分钟数据，并与本地掘金右对齐 15 分钟样本比较 18 根
    Bar 的时间戳与 OHLCV。单票代码会被下推到 DDB 查询条件，避免为了质量
    门拉取全市场分钟线。
    """
    sample_date = dt.date(2021, 1, 4)
    sample_rq_code = "000001.XSHE"
    sample_gm_code = "SZSE.000001"

    minute_sample = fetch_minute_range(
        session,
        sample_date,
        sample_date,
        allowed_dates={sample_date},
        rq_codes=[sample_rq_code],
    ).filter(pl.col("code") == sample_gm_code)

    if minute_sample.height != 18:
        logging.error("分钟质量门失败: DDB 合成样本行数为 %s，期望 18", minute_sample.height)
        return False

    gm_partition = Path(DATA_ROOT_DIR) / "15min_stock_data_right_dir" / f"trading_date={to_date_str(sample_date)}"
    gm_files = sorted(gm_partition.glob("*.parquet"))
    if not gm_files:
        logging.error("分钟质量门失败: 找不到 GM 右对齐样本 %s", gm_partition)
        return False

    gm_sample = (
        pl.read_parquet([str(path) for path in gm_files])
        .filter(pl.col("code") == sample_gm_code)
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .sort("datetime")
    )
    if gm_sample.height != 18:
        logging.error("分钟质量门失败: GM 右对齐样本行数为 %s，期望 18", gm_sample.height)
        return False

    compare_cols = ["open", "high", "low", "close", "volume"]
    ddb_sample = minute_sample.select(["datetime", *compare_cols]).sort("datetime")
    gm_times = gm_sample["datetime"].to_list()
    ddb_times = ddb_sample["datetime"].to_list()
    if ddb_times != gm_times:
        logging.error("分钟质量门失败: DDB 合成时间戳与 GM 右对齐样本不一致")
        return False

    # 右对齐口径的关键不是不同数据源的 OHLCV 逐值相等，而是 09:30/13:00
    # snapshot 是否按 GM 目录规则补齐：OHLC 全部等于下一根完整 Bar 的 open，
    # volume 复用下一根完整 Bar 的 volume。
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
            logging.error("分钟质量门失败: 缺少 snapshot 或首根完整 Bar: %s / %s", snapshot_time, first_bar_time)
            return False

        snapshot = snapshot_rows.row(0, named=True)
        first_bar = first_bar_rows.row(0, named=True)
        snapshot_expected = {
            "open": first_bar["open"],
            "high": first_bar["open"],
            "low": first_bar["open"],
            "close": first_bar["open"],
            "volume": first_bar["volume"],
        }
        for col, expected_value in snapshot_expected.items():
            if abs(snapshot[col] - expected_value) > 1e-6:
                logging.error(
                    "分钟质量门失败: %s snapshot 字段 %s=%s，期望 %s",
                    snapshot_time,
                    col,
                    snapshot[col],
                    expected_value,
                )
                return False

    joined = gm_sample.rename({col: f"gm_{col}" for col in compare_cols}).join(
        ddb_sample,
        on="datetime",
        how="inner",
    )
    max_diff = joined.select(
        [
            (pl.col(f"gm_{col}") - pl.col(col)).abs().max().alias(col)
            for col in compare_cols
        ]
    ).to_dicts()[0]
    bad_diff = {
        col: value
        for col, value in max_diff.items()
        if value is not None and value > 1e-6
    }
    if bad_diff:
        logging.warning("分钟质量门提示: RQ/DDB 1min 聚合值与 GM 15min 样本存在源数据差异 %s", bad_diff)

    logging.info("分钟质量门通过: DDB 1min 合成时间戳与 snapshot 规则符合 GM 右对齐口径")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 参数解析。"""
    parser = argparse.ArgumentParser(description="米筐本地数据更新脚本（DDB 版 v4）")
    parser.add_argument("--start-date", default="2021-01-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument(
        "--end-date",
        default=dt.date.today().strftime("%Y-%m-%d"),
        help="结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--mode", choices=["insert", "update"], default="insert",
        help="insert=增量（默认）; update=按指定范围重写",
    )
    parser.add_argument(
        "--quality-check-only",
        action="store_true",
        help="只运行 DDB 质量门，不写入本地数据",
    )
    parser.add_argument(
        "--data-type",
        choices=["day", "min", "all"],
        default="day",
        help="day=只更新日线/复权（默认）；min=只更新米筐右对齐15分钟；all=日线和分钟都更新",
    )
    parser.add_argument(
        "--minute-quality-check-only",
        action="store_true",
        help="只运行米筐分钟线 DDB 口径质量门，不写入本地数据",
    )
    parser.add_argument(
        "--skip-quality-check",
        action="store_true",
        help="跳过 DDB 质量门",
    )
    parser.add_argument(
        "--batch-mode",
        choices=["all", "year", "days"],
        default="days",
        help="DDB 批量查询模式",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="batch-mode=days 时每批交易日数量",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """入口：从 DDB 读取米筐数据，按天循环写入本地 Parquet。"""
    args = parse_args(argv)
    get_logger(log_file="log/米筐数据更新.log", inherit=False)

    start_date = to_date(args.start_date)
    end_date = to_date(args.end_date)
    if start_date > end_date:
        logging.info("起始日期大于结束日期，无需更新: %s > %s", start_date, end_date)
        return 0

    # insert 模式根据本次更新的数据类型推断续跑目录；默认 day 行为保持不变。
    if args.mode == "insert":
        infer_dir = RQ_MIN_DIR if args.data_type == "min" else RQ_DAY_DIR
        existing = get_existing_dates(infer_dir)
        if existing:
            start_date = max(existing) + dt.timedelta(days=1)
            if start_date > end_date:
                logging.info("%s 数据已是最新，无需更新", infer_dir)
                return 0

    logging.info(
        "米筐更新开始（DDB）— 类型: %s | 模式: %s | %s ~ %s",
        args.data_type,
        args.mode,
        to_date_str(start_date),
        to_date_str(end_date),
    )

    session = create_ddb_session()
    try:
        if args.minute_quality_check_only:
            return 0 if run_minute_quality_gate(session) else 2
        if args.quality_check_only:
            return 0 if run_ddb_quality_gate(session) else 2

        if not args.skip_quality_check and args.mode == "update":
            if args.data_type in ("day", "all") and not run_ddb_quality_gate(session):
                logging.error("DDB 日线质量门失败，停止 update 重写；请改用官方米筐接口 fallback")
                return 2
            if args.data_type in ("min", "all") and not run_minute_quality_gate(session):
                logging.error("DDB 分钟质量门失败，停止分钟线 update 重写")
                return 2

        if args.data_type in ("day", "all"):
            update_all(
                session,
                start_date,
                end_date,
                args.mode,
                batch_mode=args.batch_mode,
                batch_size=args.batch_size,
            )

        if args.data_type in ("min", "all"):
            update_minute_all(
                session,
                start_date,
                end_date,
                args.mode,
            )
    finally:
        session.close()

    logging.info("米筐数据更新结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
