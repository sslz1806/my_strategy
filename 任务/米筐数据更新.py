"""从 DolphinDB 更新米筐日线、复权因子和右对齐 15 分钟数据。"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_utils.fun import get_logger  # noqa: E402
from my_utils.rqdata import DDBData  # noqa: E402
from my_utils.rq_fun import (  # noqa: E402
    RQ_MIN_SCHEMA,
    infer_start_date,
    run_ddb_quality_gate,
    run_minute_quality_gate,
    to_date,
    to_date_str,
    validate_day_expected_trading_days,
    validate_day_market_coverage,
    validate_minute_expected_trading_days,
    validate_minute_market_coverage,
    write_partitioned,
    write_rq_day_and_adj_partitioned,
)

RQ_DAY_DIR = "rq_stock_all_data"
RQ_MIN_DIR = "rq_15min_stock_data_dir"
RQ_ADJ_DIR = "rq_adj"
BATCH_SIZE = 60


def update_day_range(
    source: DDBData,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    rq_codes: list[str] | None = None,
    allowed_dates: Iterable[dt.date] | None = None,
) -> int:
    """查询并写入一批日线和复权因子数据。"""
    all_data = source.fetch_day_range(
        start_date,
        end_date,
        rq_codes=rq_codes,
        allowed_dates=allowed_dates,
    )
    validate_day_expected_trading_days(
        all_data,
        allowed_dates=allowed_dates,
        rq_codes=rq_codes,
    )
    if all_data.is_empty():
        return 0
    validate_day_market_coverage(all_data, rq_codes=rq_codes)

    return write_rq_day_and_adj_partitioned(
        all_data,
        RQ_DAY_DIR,
        RQ_ADJ_DIR,
        mode,
    )


def update_minute_range(
    source: DDBData,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    allowed_dates: Iterable[dt.date] | None = None,
    rq_codes: list[str] | None = None,
) -> int:
    """查询、校验并写入一个月度范围内的右对齐 15 分钟数据。"""
    minute_data = source.fetch_minute_range(
        start_date,
        end_date,
        allowed_dates=allowed_dates,
        rq_codes=rq_codes,
    )
    validate_minute_expected_trading_days(
        minute_data,
        allowed_dates=allowed_dates,
        rq_codes=rq_codes,
    )
    if minute_data.is_empty():
        return 0
    validate_minute_market_coverage(minute_data, rq_codes=rq_codes)

    return write_partitioned(minute_data, RQ_MIN_DIR, RQ_MIN_SCHEMA, mode)


def build_batch_ranges(
    trading_days: list[dt.date],
    batch_mode: str = "days",
    batch_size: int = BATCH_SIZE,
) -> list[tuple[dt.date, dt.date]]:
    """按全部、自然年或固定交易日数量生成日线查询区间。"""
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
        for batch in (
            days[index : index + batch_size]
            for index in range(0, len(days), batch_size)
        )
    ]


def build_month_ranges(
    start_date: dt.date,
    end_date: dt.date,
) -> list[tuple[dt.date, dt.date]]:
    """按自然月切分分钟线查询范围。"""
    if start_date > end_date:
        return []

    ranges: list[tuple[dt.date, dt.date]] = []
    cursor = start_date.replace(day=1)
    while cursor <= end_date:
        if cursor.month == 12:
            next_month = dt.date(cursor.year + 1, 1, 1)
        else:
            next_month = dt.date(cursor.year, cursor.month + 1, 1)
        ranges.append(
            (
                max(start_date, cursor),
                min(end_date, next_month - dt.timedelta(days=1)),
            )
        )
        cursor = next_month
    return ranges


def update_all(
    source: DDBData,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    batch_mode: str = "days",
    batch_size: int = BATCH_SIZE,
) -> int:
    """按配置批次更新指定范围内的日线和复权因子。"""
    trading_days = source.get_trading_days(start_date, end_date)
    if not trading_days:
        logging.warning("未获取到交易日列表")
        return 0

    rq_codes = source.get_stock_universe()
    if not rq_codes:
        logging.warning("股票池为空")
        return 0

    batch_ranges = build_batch_ranges(
        trading_days,
        batch_mode=batch_mode,
        batch_size=batch_size,
    )
    total_written = 0
    for index, (batch_start, batch_end) in enumerate(batch_ranges, start=1):
        logging.info(
            "[%s/%s] 批次 %s ~ %s",
            index,
            len(batch_ranges),
            to_date_str(batch_start),
            to_date_str(batch_end),
        )
        allowed_dates = [
            value for value in trading_days if batch_start <= value <= batch_end
        ]
        total_written += update_day_range(
            source,
            batch_start,
            batch_end,
            mode,
            rq_codes=rq_codes,
            allowed_dates=allowed_dates,
        )

    logging.info(
        "更新完成: 共处理 %s 个交易日, 写入 %s 行",
        len(trading_days),
        total_written,
    )
    return total_written


def update_minute_all(
    source: DDBData,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
) -> int:
    """按自然月更新指定范围内的右对齐 15 分钟数据。"""
    trading_days = source.get_trading_days(start_date, end_date)
    if not trading_days:
        logging.warning("分钟线未获取到交易日列表")
        return 0

    rq_codes = source.get_stock_universe()
    if not rq_codes:
        logging.warning("分钟线股票池为空")
        return 0

    total_written = 0
    month_ranges = build_month_ranges(start_date, end_date)
    for index, (month_start, month_end) in enumerate(month_ranges, start=1):
        allowed_dates = [
            value for value in trading_days if month_start <= value <= month_end
        ]
        logging.info(
            "[分钟 %s/%s] 月度批次 %s ~ %s",
            index,
            len(month_ranges),
            to_date_str(month_start),
            to_date_str(month_end),
        )
        try:
            total_written += update_minute_range(
                source,
                month_start,
                month_end,
                mode,
                allowed_dates=allowed_dates,
                rq_codes=rq_codes,
            )
        except RuntimeError as exc:
            logging.warning(
                "[分钟 %s/%s] 月度批次失败，改为按交易日重试: %s",
                index,
                len(month_ranges),
                exc,
            )
            for trade_date in allowed_dates:
                logging.info(
                    "[分钟 %s/%s] 单日重试 %s",
                    index,
                    len(month_ranges),
                    to_date_str(trade_date),
                )
                total_written += update_minute_range(
                    source,
                    trade_date,
                    trade_date,
                    mode,
                    allowed_dates=[trade_date],
                    rq_codes=rq_codes,
                )

    logging.info(
        "分钟线更新完成: 共处理 %s 个交易日, 写入 %s 行",
        len(trading_days),
        total_written,
    )
    return total_written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析更新范围、模式、数据类型和质量门参数。"""
    parser = argparse.ArgumentParser(description="米筐本地数据更新脚本（DDB 版 v4）")
    parser.add_argument("--start-date", default="2021-01-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument(
        "--end-date",
        default=dt.date.today().strftime("%Y-%m-%d"),
        help="结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--mode",
        choices=["insert", "update"],
        default="insert",
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
        default="all",
        help="day=只更新日线/复权；min=只更新右对齐15分钟；all=全部更新",
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
    """执行用户选择的质量检查或数据更新流程。"""
    args = parse_args(argv)
    get_logger(log_file="log/米筐数据更新.log", inherit=False)

    start_date = to_date(args.start_date)
    end_date = to_date(args.end_date)
    if start_date > end_date:
        logging.info("起始日期大于结束日期，无需更新: %s > %s", start_date, end_date)
        return 0

    day_start_date = start_date if args.data_type in ("day", "all") else None
    minute_start_date = start_date if args.data_type in ("min", "all") else None
    if args.mode == "insert":
        if day_start_date is not None:
            day_start_date = infer_start_date(
                start_date,
                RQ_DAY_DIR,
                args.mode,
                end_date=end_date,
            )
        if minute_start_date is not None:
            minute_start_date = infer_start_date(
                start_date,
                RQ_MIN_DIR,
                args.mode,
                end_date=end_date,
            )
        if day_start_date is None and minute_start_date is None:
            return 0

    active_starts = [
        value for value in (day_start_date, minute_start_date) if value is not None
    ]
    log_start_date = min(active_starts) if active_starts else start_date
    logging.info(
        "米筐更新开始（DDB）— 类型: %s | 模式: %s | %s ~ %s",
        args.data_type,
        args.mode,
        to_date_str(log_start_date),
        to_date_str(end_date),
    )

    with DDBData() as source:
        if args.minute_quality_check_only:
            return 0 if run_minute_quality_gate(source) else 2
        if args.quality_check_only:
            return 0 if run_ddb_quality_gate(source) else 2

        if not args.skip_quality_check and args.mode == "update":
            if args.data_type in ("day", "all") and not run_ddb_quality_gate(source):
                logging.error("DDB 日线质量门失败，停止 update 重写；请改用官方米筐接口 fallback")
                return 2
            if args.data_type in ("min", "all") and not run_minute_quality_gate(source):
                logging.error("DDB 分钟质量门失败，停止分钟线 update 重写")
                return 2

        if args.data_type in ("day", "all"):
            if day_start_date is None:
                logging.info("%s 数据已跳过：已是最新", RQ_DAY_DIR)
            else:
                update_all(
                    source,
                    day_start_date,
                    end_date,
                    args.mode,
                    batch_mode=args.batch_mode,
                    batch_size=args.batch_size,
                )

        if args.data_type in ("min", "all"):
            if minute_start_date is None:
                logging.info("%s 数据已跳过：已是最新", RQ_MIN_DIR)
            else:
                update_minute_all(
                    source,
                    minute_start_date,
                    end_date,
                    args.mode,
                )

    logging.info("米筐数据更新结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
