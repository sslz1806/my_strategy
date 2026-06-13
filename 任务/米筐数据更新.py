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


def fetch_minute_full(
    session: ddb.session,
    trade_date: dt.date,
) -> pl.DataFrame:
    """
    从 DDB 1分钟线聚合生成 15分钟 K 线。

    Args:
        session: DDB 会话
        trade_date: 交易日

    Returns:
        符合 RQ_MIN_SCHEMA 的 DataFrame
    """
    date_str = trade_date.strftime("%Y.%m.%d")
    try:
        raw = session.run(f"""
            select order_book_id, trade_time,
                   open, close, high, low, volume
            from loadTable('dfs://common_years_olap', 'one_min_kline')
            where (order_book_id like '%.XSHE' or order_book_id like '%.XSHG')
              and trade_time >= timestamp({date_str} 09:30:00)
              and trade_time <= timestamp({date_str} 15:00:00)
        """)
    except Exception as exc:
        logging.warning("  DDB 分钟线查询失败: %s", exc)
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    if raw.empty:
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    df = pl.from_pandas(raw)

    # 计算 15 分钟分桶标签（从 09:30 起，每 15 分钟一桶）
    df = df.with_columns([
        pl.col("trade_time").cast(pl.Datetime).alias("trade_time_dt"),
    ])

    base_minutes = 9 * 60 + 30  # 09:30 = 570
    df = df.with_columns([
        (pl.col("trade_time_dt").dt.hour() * 60 + pl.col("trade_time_dt").dt.minute()).alias("minute_of_day"),
    ])
    df = df.with_columns([
        ((pl.col("minute_of_day") - base_minutes) // 15).alias("bucket_id"),
    ])

    # 按股票 + 分桶聚合
    agg = df.group_by(["order_book_id", "bucket_id"]).agg([
        pl.col("open").first().alias("open"),
        pl.col("close").last().alias("close"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("volume").sum().alias("volume"),
        pl.col("trade_time_dt").first().alias("datetime"),
    ])

    # 代码格式转换
    agg = agg.with_columns([
        pl.when(pl.col("order_book_id").str.ends_with(".XSHE"))
        .then(pl.lit("SZSE.") + pl.col("order_book_id").str.replace(".XSHE", ""))
        .otherwise(pl.lit("SHSE.") + pl.col("order_book_id").str.replace(".XSHG", ""))
        .alias("code"),
    ])

    result = agg.with_columns([
        pl.lit(trade_date, dtype=pl.Date).alias("trading_date"),
    ]).select([
        "code", "datetime", "open", "high", "low", "close", "volume", "trading_date",
    ]).sort(["code", "datetime"])

    return result.cast(RQ_MIN_SCHEMA)


# ============================================================================
# 更新入口 — 分批批量
# ============================================================================


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


def update_all(session: ddb.session, start_date: dt.date, end_date: dt.date, mode: str) -> int:
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

    for i in range(0, n_days, BATCH_SIZE):
        batch = trading_days[i : i + BATCH_SIZE]
        batch_start = batch[0]
        batch_end = batch[-1]
        logging.info(
            "[%s/%s] 批次 %s ~ %s",
            i + 1, n_days, to_date_str(batch_start), to_date_str(batch_end),
        )

        written = update_day_range(session, batch_start, batch_end, mode, rq_codes=rq_codes)
        total_written += written

    logging.info("更新完成: 共处理 %s 个交易日, 写入 %s 行", n_days, total_written)
    return total_written


# ============================================================================
# CLI & 入口
# ============================================================================


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

    # insert 模式：从已有数据最大日期的后一天开始
    if args.mode == "insert":
        existing = get_existing_dates(RQ_DAY_DIR)
        if existing:
            start_date = max(existing) + dt.timedelta(days=1)
            if start_date > end_date:
                logging.info("日线数据已是最新，无需更新")
                return 0

    logging.info(
        "米筐更新开始（DDB）— 模式: %s | %s ~ %s",
        args.mode, to_date_str(start_date), to_date_str(end_date),
    )

    session = create_ddb_session()
    try:
        update_all(session, start_date, end_date, args.mode)
    finally:
        session.close()

    logging.info("米筐数据更新结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
