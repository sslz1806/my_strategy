"""使用米筐官方接口更新全部历史 ETF 日线与原始 1 分钟行情。"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from rqdatac.share.errors import (
    AuthenticationFailed,
    GatewayError,
    PermissionDenied,
    QuotaExceeded,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_utils.fun import get_logger  # noqa: E402
from my_utils.rqdata import RqData  # noqa: E402
from my_utils.rq_fun import (  # noqa: E402
    RQ_ETF_DAY_SCHEMA,
    RQ_ETF_MIN_SCHEMA,
    build_etf_minute_batches,
    cleanup_new_failed_partitions,
    drop_incomplete_current_etf_minute_date,
    filter_etf_codes_for_range,
    get_existing_dates,
    infer_start_date,
    measure_bytes_per_row,
    normalize_etf_day_data,
    normalize_etf_instruments,
    normalize_etf_minute_data,
    quota_remaining_bytes,
    select_minute_days_for_quota,
    to_date,
    validate_etf_day_batch,
    validate_etf_minute_batch,
    write_partitioned,
)

RQ_ETF_DAY_DIR = "rq_etf_day_data"
RQ_ETF_MIN_DIR = "rq_1min_etf_data_dir"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_MAX_MINUTE_ROWS = 3_000_000
DEFAULT_QUOTA_RESERVE_MB = 128
# 实盘观测中，后续大批次的字节/行曾比校准批次高约 42%；1.75 倍既覆盖
# 这类波动，也能在 128 MiB 固定余量之外保留额外缓冲。
DEFAULT_QUOTA_SAFETY_FACTOR = 1.75
DEFAULT_QUOTA_POLL_ATTEMPTS = 4
DEFAULT_QUOTA_POLL_SECONDS = 2.0
DEFAULT_CALIBRATION_BYTES_PER_ROW = 128.0
FATAL_RQ_ERRORS = (AuthenticationFailed, PermissionDenied, QuotaExceeded)
SPLITTABLE_RQ_ERRORS = (GatewayError, TimeoutError)


def fetch_and_save(
    fetch_batch: Callable[[list[dt.date]], pd.DataFrame],
    validate_and_save_batch: Callable[[pd.DataFrame, list[dt.date]], int],
    trading_days: list[dt.date],
    sleep_func: Callable[[float], None] = time.sleep,
) -> int:
    """整批取数；响应过大时只按交易日二分，永不拆分 ETF 代码。"""
    days = sorted(set(trading_days))
    if not days:
        return 0

    try:
        try:
            raw = fetch_batch(days)
        except ConnectionError as exc:
            # 网络瞬断只重试一次，避免在凭据、权限或额度错误时反复消耗调用次数。
            logging.warning(
                "米筐批次 %s ~ %s 网络异常，3 秒后重试一次: %s",
                days[0],
                days[-1],
                exc,
            )
            sleep_func(3.0)
            raw = fetch_batch(days)
    except FATAL_RQ_ERRORS:
        # 额度、认证和权限问题无法靠重试恢复，必须立即终止本次任务。
        raise
    except SPLITTABLE_RQ_ERRORS as exc:
        if len(days) == 1:
            logging.error("米筐单交易日请求仍失败: %s", exc)
            raise
        middle = len(days) // 2
        logging.warning(
            "米筐批次 %s ~ %s 失败，按交易日二分: %s",
            days[0],
            days[-1],
            exc,
        )
        return fetch_and_save(
            fetch_batch,
            validate_and_save_batch,
            days[:middle],
            sleep_func=sleep_func,
        ) + fetch_and_save(
            fetch_batch,
            validate_and_save_batch,
            days[middle:],
            sleep_func=sleep_func,
        )

    return validate_and_save_batch(raw, days)


def update_day(
    source: RqData,
    instruments: pd.DataFrame,
    trading_days: list[dt.date],
    mode: str,
) -> int:
    """按日期范围一次请求全部有效 ETF，并写入日线分区。"""

    def fetch_batch(days: list[dt.date]) -> pd.DataFrame:
        rq_codes = filter_etf_codes_for_range(instruments, days[0], days[-1])
        logging.info(
            "请求 ETF 日线: %s ~ %s，%s 只 ETF",
            days[0],
            days[-1],
            len(rq_codes),
        )
        return source.fetch_etf_day_range(rq_codes, days[0], days[-1])

    def validate_and_save_batch(raw: pd.DataFrame, days: list[dt.date]) -> int:
        rq_codes = filter_etf_codes_for_range(instruments, days[0], days[-1])
        data = normalize_etf_day_data(raw)
        validate_etf_day_batch(data, rq_codes, days)
        return write_partitioned(data, RQ_ETF_DAY_DIR, RQ_ETF_DAY_SCHEMA, mode)

    return fetch_and_save(fetch_batch, validate_and_save_batch, trading_days)


def update_minute(
    source: RqData,
    instruments: pd.DataFrame,
    trading_days: list[dt.date],
    mode: str,
    max_rows: int = DEFAULT_MAX_MINUTE_ROWS,
    quota_reserve_bytes: int = DEFAULT_QUOTA_RESERVE_MB * 1024 * 1024,
    quota_safety_factor: float = DEFAULT_QUOTA_SAFETY_FACTOR,
    quota_poll_attempts: int = DEFAULT_QUOTA_POLL_ATTEMPTS,
    quota_poll_seconds: float = DEFAULT_QUOTA_POLL_SECONDS,
    quota_sleep_func: Callable[[float], None] = time.sleep,
    calibration_bytes_per_row: float = DEFAULT_CALIBRATION_BYTES_PER_ROW,
) -> int:
    """按当日实测流量规划并写入能够完整容纳的分钟交易日。"""
    days = sorted(set(trading_days))
    if not days:
        return 0
    if quota_reserve_bytes < 0:
        raise ValueError("quota_reserve_bytes must not be negative")
    if calibration_bytes_per_row <= 0:
        raise ValueError("calibration_bytes_per_row must be positive")
    if quota_poll_attempts <= 0:
        raise ValueError("quota_poll_attempts must be positive")
    if quota_poll_seconds < 0:
        raise ValueError("quota_poll_seconds must not be negative")

    # 记录任务启动前的完整分区。异常清理时这些日期始终受保护，避免 update
    # 或重复运行时误删用户原本已经拥有的可靠数据。
    dates_before_run = set(get_existing_dates(RQ_ETF_MIN_DIR))
    committed_dates: set[dt.date] = set()

    def download_days(selected_days: list[dt.date]) -> int:
        """下载一组已通过额度规划的日期，并逐批校验、原子落盘。"""
        batches = build_etf_minute_batches(
            selected_days,
            instruments,
            max_rows=max_rows,
        )
        written_rows = 0

        for index, batch in enumerate(batches, start=1):
            batch_days = list(batch.trading_days)
            logging.info(
                "ETF 1min 批次 %s/%s: %s ~ %s，%s 只 ETF，预计 %s 行",
                index,
                len(batches),
                batch.start_date,
                batch.end_date,
                len(batch.rq_codes),
                batch.estimated_rows,
            )

            def fetch_batch(request_days: list[dt.date]) -> pd.DataFrame:
                # 日期二分后重新计算交集代码，但每次请求仍携带该日期范围内
                # 全部有效 ETF，绝不通过拆代码来换取更多调用次数。
                rq_codes = filter_etf_codes_for_range(
                    instruments,
                    request_days[0],
                    request_days[-1],
                )
                return source.fetch_etf_minute_range(
                    rq_codes,
                    request_days[0],
                    request_days[-1],
                )

            def validate_and_save_batch(
                raw: pd.DataFrame,
                request_days: list[dt.date],
            ) -> int:
                rq_codes = filter_etf_codes_for_range(
                    instruments,
                    request_days[0],
                    request_days[-1],
                )
                data = normalize_etf_minute_data(raw)
                data = drop_incomplete_current_etf_minute_date(data)
                validate_etf_minute_batch(data, rq_codes, request_days)
                rows = write_partitioned(
                    data,
                    RQ_ETF_MIN_DIR,
                    RQ_ETF_MIN_SCHEMA,
                    mode,
                )
                if rows:
                    committed_dates.update(data["trading_date"].unique().to_list())
                return rows

            try:
                written_rows += fetch_and_save(
                    fetch_batch,
                    validate_and_save_batch,
                    batch_days,
                )
            except Exception:
                # 请求失败通常发生在写盘前，因此不会有坏分区；这里仍执行
                # 第二道清理，只处理本轮新建且尚未确认提交的精确日期目录。
                failed_dates = [
                    day for day in batch_days if day not in committed_dates
                ]
                cleanup_new_failed_partitions(
                    RQ_ETF_MIN_DIR,
                    failed_dates,
                    dates_before_run,
                )
                raise

        return written_rows

    quota_before = source.get_quota()
    available_before = quota_remaining_bytes(quota_before, quota_reserve_bytes)
    if available_before == 0:
        logging.warning(
            "ETF 1min 今日安全额度已用完，未发送行情请求；请下一额度日继续运行"
        )
        return 0

    # 上限为 0 表示不限流量，此时无需校准，直接沿用行数批次规划全部日期。
    if available_before is None:
        logging.info("米筐账户不限流量，本轮计划更新全部 %s 个分钟交易日", len(days))
        return download_days(days)

    # 用尽量接近正常请求大小的首个批次校准。单日响应中的固定协议开销占比
    # 过高，会严重高估大批量传输的字节/行；最多 300 万行的真实批次既能
    # 摊薄固定开销，也直接作为正式数据落盘，不增加纯采样行情调用。
    safe_calibration_rows = int(
        available_before
        / (calibration_bytes_per_row * quota_safety_factor)
    )
    if safe_calibration_rows <= 0:
        logging.warning("今日剩余额度不足以安全执行分钟校准，请下一额度日继续")
        return 0
    calibration_row_budget = min(max_rows, safe_calibration_rows)
    calibration_batches = build_etf_minute_batches(
        days,
        instruments,
        max_rows=calibration_row_budget,
    )
    if not calibration_batches:
        return 0
    calibration_batch = calibration_batches[0]
    if calibration_batch.estimated_rows > calibration_row_budget:
        logging.warning(
            "今日剩余额度不足以安全下载首个分钟交易日: %s",
            calibration_batch.start_date,
        )
        return 0
    calibration_days = list(calibration_batch.trading_days)
    calibration_rows = download_days(calibration_days)
    if calibration_rows == 0:
        logging.warning(
            "分钟校准批次没有完整数据，本轮停止: %s ~ %s",
            calibration_days[0],
            calibration_days[-1],
        )
        return 0

    # 官方 bytes_used 可能在行情响应返回后延迟数秒刷新。轮询只调用轻量的
    # 额度接口，不重复请求价格；这样不会把延迟到账误判为校准失败。
    quota_after = quota_before
    before_used = int(quota_before["bytes_used"])
    for attempt in range(quota_poll_attempts):
        quota_after = source.get_quota()
        if int(quota_after["bytes_used"]) > before_used:
            break
        if attempt < quota_poll_attempts - 1:
            logging.info(
                "米筐额度计数尚未刷新，%.1f 秒后再次查询（%s/%s）",
                quota_poll_seconds,
                attempt + 1,
                quota_poll_attempts - 1,
            )
            quota_sleep_func(quota_poll_seconds)
    bytes_per_row = measure_bytes_per_row(
        quota_before,
        quota_after,
        calibration_rows,
    )
    available_after = quota_remaining_bytes(quota_after, quota_reserve_bytes)
    planned_days = select_minute_days_for_quota(
        days[len(calibration_days):],
        instruments,
        available_after,
        bytes_per_row,
        safety_factor=quota_safety_factor,
    )
    logging.info(
        "ETF 1min 额度校准：%.2f 字节/行，安全系数 %.2f；今日计划 %s 个交易日（%s ~ %s）",
        bytes_per_row,
        quota_safety_factor,
        len(calibration_days) + len(planned_days),
        calibration_days[0],
        planned_days[-1] if planned_days else calibration_days[-1],
    )
    return calibration_rows + download_days(planned_days)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 ETF 更新范围、写入模式和分钟批次预算。"""
    parser = argparse.ArgumentParser(description="米筐官方 ETF 日线/1min 数据更新")
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="起始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        default=dt.date.today().isoformat(),
        help="结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--mode",
        choices=["insert", "update"],
        default="insert",
        help="insert=从各自最新分区增量更新；update=重写指定范围",
    )
    parser.add_argument(
        "--data-type",
        choices=["day", "min", "all"],
        default="all",
        help="day=日线；min=原始 1 分钟线；all=两者",
    )
    parser.add_argument(
        "--max-minute-rows",
        type=_positive_int,
        default=DEFAULT_MAX_MINUTE_ROWS,
        help="单次 1min 请求的预计最大行数；不会据此拆分 ETF 代码",
    )
    parser.add_argument(
        "--quota-reserve-mb",
        type=_positive_int,
        default=DEFAULT_QUOTA_RESERVE_MB,
        help="每日流量安全余量（MiB），默认 128",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行 ETF 日线、原始 1 分钟线或两者的官方增量更新。"""
    args = parse_args(argv)
    log_path = Path(__file__).resolve().parent / "log" / "米筐ETF数据更新.log"
    get_logger(log_file=str(log_path), inherit=False)

    requested_start = to_date(args.start_date)
    end_date = to_date(args.end_date)
    if requested_start > end_date:
        logging.info("起始日期大于结束日期，无需更新: %s > %s", requested_start, end_date)
        return 0

    day_start = requested_start if args.data_type in ("day", "all") else None
    minute_start = requested_start if args.data_type in ("min", "all") else None
    if args.mode == "insert":
        if day_start is not None:
            day_start = infer_start_date(
                requested_start,
                RQ_ETF_DAY_DIR,
                args.mode,
                end_date=end_date,
            )
        if minute_start is not None:
            minute_start = infer_start_date(
                requested_start,
                RQ_ETF_MIN_DIR,
                args.mode,
                end_date=end_date,
            )

    active_starts = [
        value for value in (day_start, minute_start) if value is not None
    ]
    if not active_starts:
        logging.info("所选 ETF 数据均已更新至 %s", end_date)
        return 0

    calendar_start = min(active_starts)
    logging.info(
        "米筐官方 ETF 更新开始 — 类型: %s | 模式: %s | %s ~ %s",
        args.data_type,
        args.mode,
        calendar_start,
        end_date,
    )

    # 基础池和交易日历每次运行各请求一次，后续日线与分钟线共同复用。
    source = RqData()
    instruments = normalize_etf_instruments(source.get_etf_instruments())
    trading_days = source.get_trading_days(calendar_start, end_date)
    logging.info(
        "ETF 历史池共 %s 只；本轮交易日 %s 天",
        len(instruments),
        len(trading_days),
    )

    day_written = 0
    minute_written = 0
    if day_start is None and args.data_type in ("day", "all"):
        logging.info("%s 已是最新，跳过日线", RQ_ETF_DAY_DIR)
    elif day_start is not None:
        day_days = [day for day in trading_days if day >= day_start]
        day_written = update_day(
            source,
            instruments,
            day_days,
            mode=args.mode,
        )

    if minute_start is None and args.data_type in ("min", "all"):
        logging.info("%s 已是最新，跳过 1min", RQ_ETF_MIN_DIR)
    elif minute_start is not None:
        minute_days = [day for day in trading_days if day >= minute_start]
        minute_written = update_minute(
            source,
            instruments,
            minute_days,
            mode=args.mode,
            max_rows=args.max_minute_rows,
            quota_reserve_bytes=args.quota_reserve_mb * 1024 * 1024,
        )

    logging.info(
        "米筐官方 ETF 更新结束：日线 %s 行，1min %s 行",
        day_written,
        minute_written,
    )
    source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
