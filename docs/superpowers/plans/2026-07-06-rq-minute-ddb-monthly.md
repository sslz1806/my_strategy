# RQ Minute DDB Monthly Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe monthly DDB 1-minute to right-aligned 15-minute backfill path for `rq_15min_stock_data_dir`.

**Architecture:** Keep `任务/米筐数据更新.py` as the orchestration script and add focused pure helpers for month ranges and right-aligned aggregation. The DDB-facing function fetches one natural month per call, then Polars performs all 15-minute aggregation and snapshot generation locally before partitioned writes. Existing day-line update behavior stays the default.

**Tech Stack:** Python 3.9 in `E:\working\anaconda3\envs\quant\python.exe`, Polars, DolphinDB Python client, pytest, existing `my_utils.rq_fun` schema/write helpers.

---

## File Structure

- Modify `任务/米筐数据更新.py`
  - Add pure helpers:
    - `build_month_ranges(start_date, end_date)`
    - `aggregate_right_aligned_15min(raw, rq_codes=None, allowed_dates=None)`
    - `_to_gm_code_expr()`
  - Replace the old day-only `fetch_minute_full` path with monthly-capable:
    - `fetch_minute_range(session, start_date, end_date, rq_codes=None)`
    - `update_minute_range(session, start_date, end_date, mode, rq_codes=None)`
    - `update_minute_all(session, start_date, end_date, mode)`
    - `run_minute_quality_gate(session)`
  - Add CLI:
    - `--data-type day|min|all`, default `day`
    - `--minute-quality-check-only`
  - Route `main()` by `data_type`.
- Modify `tests/test_historical_backfill_scripts.py`
  - Add unit tests for right-aligned aggregation and monthly ranges.
  - Add parse-args regression for the new default `data_type`.

## Task 1: Add Failing Pure Aggregation Tests

**Files:**
- Modify: `tests/test_historical_backfill_scripts.py`

- [ ] **Step 1: Add tests for right-aligned 15-minute aggregation**

Append these helpers and tests near the existing RQ tests:

```python
def _full_a_share_minute_rows(order_book_id="000001.XSHE", trade_date=dt.date(2021, 1, 4)):
    rows = []
    seq = 1
    for hour, minute_start, minute_end in [
        (9, 31, 59),
        (10, 0, 59),
        (11, 0, 30),
        (13, 1, 59),
        (14, 0, 59),
        (15, 0, 0),
    ]:
        for minute in range(minute_start, minute_end + 1):
            rows.append(
                {
                    "order_book_id": order_book_id,
                    "trade_time": dt.datetime.combine(trade_date, dt.time(hour, minute)),
                    "open": float(seq),
                    "high": float(seq) + 0.5,
                    "low": float(seq) - 0.5,
                    "close": float(seq) + 0.25,
                    "volume": float(seq),
                    "total_turnover": float(seq) * 100.0,
                }
            )
            seq += 1
    return rows


def test_rq_aggregate_right_aligned_15min_adds_gm_style_snapshots():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_agg_test")
    raw = pl.DataFrame(_full_a_share_minute_rows())

    result = rq_update.aggregate_right_aligned_15min(raw)

    assert result.schema == rq_update.RQ_MIN_SCHEMA
    assert result.height == 18
    assert result["code"].unique().to_list() == ["SZSE.000001"]
    assert result["trading_date"].unique().to_list() == [dt.date(2021, 1, 4)]
    assert result["datetime"].dt.strftime("%H:%M:%S").to_list() == [
        "09:30:00",
        "09:45:00",
        "10:00:00",
        "10:15:00",
        "10:30:00",
        "10:45:00",
        "11:00:00",
        "11:15:00",
        "11:30:00",
        "13:00:00",
        "13:15:00",
        "13:30:00",
        "13:45:00",
        "14:00:00",
        "14:15:00",
        "14:30:00",
        "14:45:00",
        "15:00:00",
    ]

    morning_snapshot = result.filter(pl.col("datetime") == dt.datetime(2021, 1, 4, 9, 30)).row(0, named=True)
    morning_first_bar = result.filter(pl.col("datetime") == dt.datetime(2021, 1, 4, 9, 45)).row(0, named=True)
    assert morning_snapshot["open"] == morning_first_bar["open"] == 1.0
    assert morning_snapshot["high"] == 1.0
    assert morning_snapshot["low"] == 1.0
    assert morning_snapshot["close"] == 1.0
    assert morning_snapshot["volume"] == morning_first_bar["volume"] == sum(float(i) for i in range(1, 16))

    afternoon_snapshot = result.filter(pl.col("datetime") == dt.datetime(2021, 1, 4, 13, 0)).row(0, named=True)
    afternoon_first_bar = result.filter(pl.col("datetime") == dt.datetime(2021, 1, 4, 13, 15)).row(0, named=True)
    assert afternoon_snapshot["open"] == afternoon_first_bar["open"] == 121.0
    assert afternoon_snapshot["high"] == 121.0
    assert afternoon_snapshot["low"] == 121.0
    assert afternoon_snapshot["close"] == 121.0
    assert afternoon_snapshot["volume"] == afternoon_first_bar["volume"] == sum(float(i) for i in range(121, 136))


def test_rq_aggregate_right_aligned_15min_filters_universe_and_dates():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_filter_test")
    rows = _full_a_share_minute_rows("000001.XSHE", dt.date(2021, 1, 4))
    rows += _full_a_share_minute_rows("000001.XSHG", dt.date(2021, 1, 4))
    rows += _full_a_share_minute_rows("000001.XSHE", dt.date(2021, 1, 5))
    raw = pl.DataFrame(rows)

    result = rq_update.aggregate_right_aligned_15min(
        raw,
        rq_codes=["000001.XSHE"],
        allowed_dates={dt.date(2021, 1, 4)},
    )

    assert result.height == 18
    assert result["code"].unique().to_list() == ["SZSE.000001"]
    assert result["trading_date"].unique().to_list() == [dt.date(2021, 1, 4)]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py::test_rq_aggregate_right_aligned_15min_adds_gm_style_snapshots tests/test_historical_backfill_scripts.py::test_rq_aggregate_right_aligned_15min_filters_universe_and_dates -q
```

Expected: both tests fail with `AttributeError: module ... has no attribute 'aggregate_right_aligned_15min'`.

- [ ] **Step 3: Commit only the failing tests is not allowed**

Do not commit red tests. Continue to Task 2.

## Task 2: Implement Pure Aggregation Helpers

**Files:**
- Modify: `任务/米筐数据更新.py`
- Test: `tests/test_historical_backfill_scripts.py`

- [ ] **Step 1: Add helper imports and code constants**

In `任务/米筐数据更新.py`, keep existing imports and add `Iterable`:

```python
from collections.abc import Iterable
```

Add constants under `DDB_QUERY_TIMEOUT`:

```python
MORNING_START_MINUTE = 9 * 60 + 30
MORNING_FIRST_TRADE_MINUTE = 9 * 60 + 31
MORNING_END_MINUTE = 11 * 60 + 30
AFTERNOON_START_MINUTE = 13 * 60
AFTERNOON_FIRST_TRADE_MINUTE = 13 * 60 + 1
AFTERNOON_END_MINUTE = 15 * 60
MINUTE_BAR_SIZE = 15
```

- [ ] **Step 2: Add `_to_gm_code_expr`**

Place after `filter_to_stock_universe`:

```python
def _to_gm_code_expr(source_col: str = "order_book_id") -> pl.Expr:
    """把米筐后缀代码转换为项目本地 GM 前缀代码。"""
    return (
        pl.when(pl.col(source_col).str.ends_with(".XSHE"))
        .then(pl.lit("SZSE.") + pl.col(source_col).str.replace(".XSHE", ""))
        .otherwise(pl.lit("SHSE.") + pl.col(source_col).str.replace(".XSHG", ""))
    )
```

- [ ] **Step 3: Add `aggregate_right_aligned_15min`**

Replace the old `fetch_minute_full` aggregation body later, but first add this pure function above it:

```python
def aggregate_right_aligned_15min(
    raw: pl.DataFrame,
    rq_codes: Iterable[str] | None = None,
    allowed_dates: set[dt.date] | None = None,
) -> pl.DataFrame:
    """
    将 DDB 1 分钟线合成为米筐右对齐 15 分钟线。

    DDB 的 trade_time 是 1 分钟 Bar 结束时间；完整 15 分钟 Bar 使用结束时间
    作为 datetime。09:30 和 13:00 为 GM 右对齐兼容 snapshot，复制对应
    第一根完整 15 分钟 Bar 的 volume，OHLC 全部设为该 Bar 的 open。
    """
    if raw.is_empty():
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    df = raw
    if rq_codes is not None:
        df = filter_to_stock_universe(df, list(rq_codes))
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
                pl.col("trade_time_dt").dt.hour() * 60
                + pl.col("trade_time_dt").dt.minute()
            ).alias("minute_of_day"),
        ]
    )

    if allowed_dates is not None:
        df = df.filter(pl.col("trading_date").is_in(list(allowed_dates)))
        if df.is_empty():
            return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    in_morning = pl.col("minute_of_day").is_between(
        MORNING_FIRST_TRADE_MINUTE, MORNING_END_MINUTE
    )
    in_afternoon = pl.col("minute_of_day").is_between(
        AFTERNOON_FIRST_TRADE_MINUTE, AFTERNOON_END_MINUTE
    )

    df = df.filter(in_morning | in_afternoon).with_columns(
        [
            pl.when(in_morning)
            .then(
                MORNING_START_MINUTE
                + (((pl.col("minute_of_day") - MORNING_FIRST_TRADE_MINUTE) // MINUTE_BAR_SIZE) + 1)
                * MINUTE_BAR_SIZE
            )
            .otherwise(
                AFTERNOON_START_MINUTE
                + (((pl.col("minute_of_day") - AFTERNOON_FIRST_TRADE_MINUTE) // MINUTE_BAR_SIZE) + 1)
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
    ).sort(["order_book_id", "trade_time_dt"])

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
        .sort(["order_book_id", "datetime"])
    )

    first_session_bars = bars.with_columns(
        (
            pl.col("datetime").dt.hour() * 60
            + pl.col("datetime").dt.minute()
        ).alias("bar_minute")
    ).filter(pl.col("bar_minute").is_in([9 * 60 + 45, 13 * 60 + 15]))

    snapshots = first_session_bars.with_columns(
        [
            (pl.col("datetime") - pl.duration(minutes=15)).alias("datetime"),
            pl.col("open").alias("high"),
            pl.col("open").alias("low"),
            pl.col("open").alias("close"),
        ]
    ).select(bars.columns)

    result = pl.concat([bars, snapshots], how="vertical").unique(
        subset=["order_book_id", "datetime"],
        keep="first",
    )

    result = result.with_columns(
        [
            _to_gm_code_expr().alias("code"),
        ]
    ).select(
        [
            "code",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trading_date",
        ]
    ).sort(["trading_date", "code", "datetime"])

    return result.cast(RQ_MIN_SCHEMA)
```

- [ ] **Step 4: Run aggregation tests**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py::test_rq_aggregate_right_aligned_15min_adds_gm_style_snapshots tests/test_historical_backfill_scripts.py::test_rq_aggregate_right_aligned_15min_filters_universe_and_dates -q
```

Expected: 2 passed.

- [ ] **Step 5: Run full regression file**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

Commit only the script and tests:

```powershell
git commit --only -m "feat: aggregate RQ minute bars right aligned" -- "任务/米筐数据更新.py" tests/test_historical_backfill_scripts.py
```

## Task 3: Add Monthly Range and CLI Tests

**Files:**
- Modify: `tests/test_historical_backfill_scripts.py`
- Modify: `任务/米筐数据更新.py`

- [ ] **Step 1: Add failing monthly range and CLI tests**

Append:

```python
def test_rq_build_month_ranges_splits_by_natural_month():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_month_ranges_test")

    ranges = rq_update.build_month_ranges(dt.date(2021, 1, 15), dt.date(2021, 3, 2))

    assert ranges == [
        (dt.date(2021, 1, 15), dt.date(2021, 1, 31)),
        (dt.date(2021, 2, 1), dt.date(2021, 2, 28)),
        (dt.date(2021, 3, 1), dt.date(2021, 3, 2)),
    ]


def test_rq_parse_args_defaults_keep_day_update_and_support_minute_type():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_cli_data_type_test")

    args = rq_update.parse_args([])
    assert args.data_type == "day"
    assert args.minute_quality_check_only is False

    min_args = rq_update.parse_args(["--data-type", "min", "--minute-quality-check-only"])
    assert min_args.data_type == "min"
    assert min_args.minute_quality_check_only is True
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py::test_rq_build_month_ranges_splits_by_natural_month tests/test_historical_backfill_scripts.py::test_rq_parse_args_defaults_keep_day_update_and_support_minute_type -q
```

Expected: first fails with missing `build_month_ranges`, second fails with missing `data_type` argument.

- [ ] **Step 3: Implement `build_month_ranges`**

Add below `build_batch_ranges`:

```python
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
```

- [ ] **Step 4: Add CLI parameters**

In `parse_args`, add:

```python
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
```

- [ ] **Step 5: Run the new tests**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py::test_rq_build_month_ranges_splits_by_natural_month tests/test_historical_backfill_scripts.py::test_rq_parse_args_defaults_keep_day_update_and_support_minute_type -q
```

Expected: 2 passed.

- [ ] **Step 6: Run full regression file**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```powershell
git commit --only -m "feat: add RQ minute monthly controls" -- "任务/米筐数据更新.py" tests/test_historical_backfill_scripts.py
```

## Task 4: Add Monthly DDB Fetch and Update Flow

**Files:**
- Modify: `任务/米筐数据更新.py`

- [ ] **Step 1: Replace old day-only minute fetch with monthly range fetch**

Replace the existing `fetch_minute_full(session, trade_date)` function with:

```python
def fetch_minute_range(
    session: ddb.session,
    start_date: dt.date,
    end_date: dt.date,
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
        """)
    except Exception as exc:
        logging.warning("  DDB 分钟线查询失败: %s", exc)
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    if raw.empty:
        logging.info("  DDB 分钟线 %s ~ %s 返回空", to_date_str(start_date), to_date_str(end_date))
        return pl.DataFrame(schema=RQ_MIN_SCHEMA)

    raw_pl = pl.from_pandas(raw)
    logging.info("    one_min_kline 返回 %d 行", len(raw_pl))
    allowed_dates = set(get_trading_days(session, start_date, end_date))
    result = aggregate_right_aligned_15min(raw_pl, rq_codes=rq_codes, allowed_dates=allowed_dates)
    logging.info("    => 合成右对齐 15min %d 行", len(result))
    return result
```

- [ ] **Step 2: Add minute update functions**

Add below `update_day_range`:

```python
def update_minute_range(
    session: ddb.session,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    rq_codes: list[str] | None = None,
) -> int:
    """
    更新一个月度范围内的米筐右对齐 15 分钟线。

    只有在 DDB 查询和本地合成结果非空后，才清理旧分区，避免失败时破坏已有数据。
    """
    minute_data = fetch_minute_range(session, start_date, end_date, rq_codes=rq_codes)
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
            rq_codes=rq_codes,
        )

    logging.info(
        "分钟线更新完成: 共处理 %s 个交易日, 写入 %s 行",
        len(trading_days),
        total_written,
    )
    return total_written
```

- [ ] **Step 3: Run tests**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit Task 4**

```powershell
git commit --only -m "feat: fetch RQ minute data monthly from DDB" -- "任务/米筐数据更新.py"
```

## Task 5: Add Minute Quality Gate and Main Routing

**Files:**
- Modify: `任务/米筐数据更新.py`

- [ ] **Step 1: Add minute quality gate**

Add below `run_ddb_quality_gate`:

```python
def run_minute_quality_gate(session: ddb.session) -> bool:
    """
    用一只股票一天的 DDB 1min 合成结果，对齐本地 GM 右对齐 15min 样本。

    该检查只查询 000001.XSHE 在 2021-01-04 的 1min 数据，查询量很小。
    """
    sample_date = dt.date(2021, 1, 4)
    sample_rq_code = "000001.XSHE"
    sample_gm_code = "SZSE.000001"

    minute_sample = fetch_minute_range(
        session,
        sample_date,
        sample_date,
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
        pl.read_parquet(gm_files)
        .filter(pl.col("code") == sample_gm_code)
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .sort("datetime")
    )

    compare_cols = ["open", "high", "low", "close", "volume"]
    ddb_sample = minute_sample.select(["datetime", *compare_cols]).sort("datetime")
    joined = gm_sample.rename({col: f"gm_{col}" for col in compare_cols}).join(
        ddb_sample,
        on="datetime",
        how="inner",
    )

    if joined.height != 18:
        logging.error("分钟质量门失败: 时间戳交集行数为 %s，期望 18", joined.height)
        return False

    max_diff = joined.select(
        [
            (pl.col(f"gm_{col}") - pl.col(col)).abs().max().alias(col)
            for col in compare_cols
        ]
    ).to_dicts()[0]
    bad_diff = {col: value for col, value in max_diff.items() if value is not None and value > 1e-6}
    if bad_diff:
        logging.error("分钟质量门失败: DDB 合成结果与 GM 右对齐样本不一致 %s", bad_diff)
        return False

    logging.info("分钟质量门通过: DDB 1min 合成结果与 GM 右对齐样本一致")
    return True
```

- [ ] **Step 2: Route `main()` by `--data-type`**

In `main()`, replace the insert start-date inference block with:

```python
    if args.mode == "insert":
        infer_dir = RQ_MIN_DIR if args.data_type == "min" else RQ_DAY_DIR
        existing = get_existing_dates(infer_dir)
        if existing:
            start_date = max(existing) + dt.timedelta(days=1)
            if start_date > end_date:
                logging.info("%s 数据已是最新，无需更新", infer_dir)
                return 0
```

Then replace the quality gate and `update_all(...)` block inside `try` with:

```python
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
```

- [ ] **Step 3: Run unit tests**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run no-write CLI smoke**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\米筐数据更新.py --start-date 2021-01-04 --end-date 2021-01-04 --data-type min --minute-quality-check-only
```

Expected:

```text
分钟质量门通过
```

Exit code: `0`.

- [ ] **Step 5: Commit Task 5**

```powershell
git commit --only -m "feat: route RQ minute monthly updates" -- "任务/米筐数据更新.py"
```

## Task 6: One-Day Safe Write Validation

**Files:**
- No code edits expected.
- Data write target: `E:\working\stock_data\rq_15min_stock_data_dir\trading_date=2021-01-04`

- [ ] **Step 1: Back up current one-day RQ minute partition if it exists**

Run:

```powershell
$src = "E:\working\stock_data\rq_15min_stock_data_dir\trading_date=2021-01-04"
$dst = "E:\working\stock_data\rq_15min_stock_data_dir_backup_20210104_before_ddb_monthly"
if (Test-Path $src) {
  if (Test-Path $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
  Copy-Item -LiteralPath $src -Destination $dst -Recurse
}
```

Expected: backup directory exists if source existed; no output is acceptable.

- [ ] **Step 2: Run one-day minute update**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\米筐数据更新.py --start-date 2021-01-04 --end-date 2021-01-04 --mode update --data-type min
```

Expected:

```text
分钟质量门通过
[分钟 1/1] 月度批次 2021-01-04 ~ 2021-01-04
分钟线更新完成
```

- [ ] **Step 3: Verify the written one-day partition**

Run:

```powershell
@'
from pathlib import Path
import polars as pl

path = Path(r"E:/working/stock_data/rq_15min_stock_data_dir/trading_date=2021-01-04")
files = sorted(path.glob("*.parquet"))
assert files, path
df = pl.read_parquet(files)
print(df.select([
    pl.len().alias("rows"),
    pl.col("code").n_unique().alias("n_codes"),
    pl.col("datetime").n_unique().alias("n_datetimes"),
    pl.col("datetime").min().alias("min_datetime"),
    pl.col("datetime").max().alias("max_datetime"),
]).to_dicts()[0])
print(df.filter(pl.col("code") == "SZSE.000001").sort("datetime").select(["datetime", "open", "high", "low", "close", "volume"]).to_dicts())
'@ | E:\working\anaconda3\envs\quant\python.exe -
```

Expected:

```text
n_datetimes: 18
min_datetime: 2021-01-04 09:30:00
max_datetime: 2021-01-04 15:00:00
```

`SZSE.000001` should have 18 rows.

- [ ] **Step 4: If validation fails, restore backup before continuing**

Run only if Step 3 fails:

```powershell
$src = "E:\working\stock_data\rq_15min_stock_data_dir_backup_20210104_before_ddb_monthly"
$dst = "E:\working\stock_data\rq_15min_stock_data_dir\trading_date=2021-01-04"
if (Test-Path $src) {
  if (Test-Path $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
  Copy-Item -LiteralPath $src -Destination $dst -Recurse
}
```

- [ ] **Step 5: Commit if code changed during validation**

If validation required code edits, commit them with:

```powershell
git commit --only -m "fix: validate RQ minute one-day write" -- "任务/米筐数据更新.py" tests/test_historical_backfill_scripts.py
```

## Task 7: Historical Minute Backfill Execution

**Files:**
- No code edits expected.
- Data write target: `E:\working\stock_data\rq_15min_stock_data_dir`

- [ ] **Step 1: Run 2018-2020 minute backfill**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\米筐数据更新.py --start-date 2018-01-01 --end-date 2020-12-31 --mode update --data-type min
```

Expected:

```text
分钟质量门通过
[分钟 1/36] 月度批次 2018-01-01 ~ 2018-01-31
...
分钟线更新完成
```

This command should issue one DDB minute query per natural month.

- [ ] **Step 2: Audit 2018-2021 RQ minute coverage**

Run:

```powershell
@'
from pathlib import Path
import datetime as dt
import polars as pl

ROOT = Path("E:/working/stock_data/rq_15min_stock_data_dir")

def parse_date(path):
    return dt.date.fromisoformat(path.name.split("=", 1)[1])

dirs = [
    item for item in ROOT.iterdir()
    if item.is_dir()
    and item.name.startswith("trading_date=")
    and dt.date(2018, 1, 1) <= parse_date(item) <= dt.date(2021, 12, 31)
]
files = [file for folder in dirs for file in folder.glob("*.parquet")]
lf = pl.scan_parquet([str(file).replace("\\", "/") for file in files], hive_partitioning=True)
summary = lf.select([
    pl.len().alias("rows"),
    pl.col("trading_date").min().alias("min_date"),
    pl.col("trading_date").max().alias("max_date"),
    pl.col("trading_date").n_unique().alias("n_dates"),
    pl.col("code").n_unique().alias("n_codes"),
    pl.col("datetime").n_unique().alias("n_datetimes"),
]).collect().to_dicts()[0]
print(summary)
per_code_day_max = (
    lf.group_by(["trading_date", "code"])
    .agg(pl.len().alias("bars"))
    .select(pl.col("bars").max().alias("max_bars"))
    .collect()
    .item()
)
print({"max_bars_per_code_day": per_code_day_max})
'@ | E:\working\anaconda3\envs\quant\python.exe -
```

Expected:

```text
min_date: 2018-01-02
max_date: 2021-12-31
max_bars_per_code_day: 18
```

`n_dates` should match the available RQ trading-day coverage for 2018-2021 after backfill.

- [ ] **Step 3: Final regression tests**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Record final status**

Run:

```powershell
git status --short -- "任务/米筐数据更新.py" tests/test_historical_backfill_scripts.py docs/superpowers/specs/2026-07-06-rq-minute-ddb-monthly-design.md docs/superpowers/plans/2026-07-06-rq-minute-ddb-monthly.md
```

Expected: no output for these tracked task files after all intended commits.

## Self-Review

Spec coverage:

- Monthly DDB query limit: Task 3 adds month ranges; Task 4 uses them in `update_minute_all`.
- Right-aligned 15-minute口径: Task 1 tests it; Task 2 implements it.
- Snapshot `09:30` and `13:00`: Task 1 asserts OHLC and volume; Task 2 implements it by cloning first session bars.
- Write to `rq_15min_stock_data_dir` and overwrite old data: Task 4 writes `RQ_MIN_DIR`; Task 6 validates one-day update.
- One-day口径 test before full run: Task 5 quality gate and Task 6 one-day write validation cover it.
- DDB call protection: Task 4 uses one query per `fetch_minute_range`; Task 7 runs 36 natural-month batches for 2018-2020.

Placeholder scan:

- This plan contains no open placeholder markers or deferred implementation steps.

Type consistency:

- New helper names are consistent across tasks:
  - `aggregate_right_aligned_15min`
  - `build_month_ranges`
  - `fetch_minute_range`
  - `update_minute_range`
  - `update_minute_all`
  - `run_minute_quality_gate`
