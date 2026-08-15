# RQData ETF Quota-Aware Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the official RQData ETF updater start at 2018, measure the current account's actual traffic cost, download only the continuous trading-day prefix that safely fits today's quota, and resume from the next complete partition on later runs.

**Architecture:** Keep official API calls in `my_utils/rqdata.py`, keep quota arithmetic and safe partition cleanup in `my_utils/rq_fun.py`, and leave `任务/米筐ETF数据更新.py` as orchestration. Calibrate minute traffic with the first safe batch of up to 3,000,000 estimated rows, reuse that batch as real output, reserve 128 MiB plus a 75% estimate margin, and preserve the existing atomic partition writer.

**Tech Stack:** Python 3.9, rqdatac 3.4.7.7, Pandas, Polars, pytest, unittest.mock, existing partition helpers.

## Global Constraints

- Use `E:\working\anaconda3\envs\quant\python.exe` for Python and pytest.
- Default start date is exactly `2018-01-01`.
- Use `adjust_type='none'`, `skip_suspended=False`, and original 1-minute end timestamps.
- Every price request contains all ETFs active in its date range; never split by code.
- Default minute response budget remains `3_000_000` estimated rows.
- Default quota reserve is `128 MiB`; observed bytes per row use a `1.75` safety multiplier.
- Fatal quota/auth/permission errors receive no retry.
- Never delete a partition that existed before the current run or a partition confirmed successfully written.
- Do not stage or commit implementation files because the shared files contain unrelated user changes. Only this plan document may be committed separately.

---

### Task 1: Clear Names and 2018 Default

**Files:**
- Modify: `任务/米筐ETF数据更新.py`
- Test: `tests/test_rq_etf_update.py`

**Interfaces:**
- Produces: `fetch_and_save(fetch_batch, validate_and_save_batch, trading_days, sleep_func=time.sleep) -> int`
- Produces: `update_day(source, instruments, trading_days, mode) -> int`
- Produces: `update_minute(source, instruments, trading_days, mode, max_rows=DEFAULT_MAX_MINUTE_ROWS) -> int`
- Renames callback parameter to `validate_and_save_batch` and local request closures to `fetch_batch`.

- [ ] **Step 1: Write failing tests for the new public names and default date**

```python
def test_etf_update_uses_clear_public_names_and_2018_default():
    module = load_etf_update_module("rq_etf_clear_names_test")
    args = module.parse_args([])

    assert args.start_date == "2018-01-01"
    assert callable(module.fetch_and_save)
    assert callable(module.update_day)
    assert callable(module.update_minute)
    assert not hasattr(module, "fetch_with_date_fallback")
    assert not hasattr(module, "update_etf_day_all")
    assert not hasattr(module, "update_etf_minute_all")
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py::test_etf_update_uses_clear_public_names_and_2018_default -q
```

Expected: FAIL because the default is 2021 and the new names do not exist.

- [ ] **Step 3: Rename the functions and callbacks, then change the constant**

Use these exact names in `任务/米筐ETF数据更新.py`:

```python
DEFAULT_START_DATE = "2018-01-01"


def fetch_and_save(
    fetch_batch: Callable[[list[dt.date]], pd.DataFrame],
    validate_and_save_batch: Callable[[pd.DataFrame, list[dt.date]], int],
    trading_days: list[dt.date],
    sleep_func: Callable[[float], None] = time.sleep,
) -> int:
    days = sorted(set(trading_days))
    if not days:
        return 0
    # Keep the existing one-network-retry and date-bisection implementation,
    # replacing every callback invocation with validate_and_save_batch(raw, days).


-def update_etf_day_all(
+def update_day(

-def update_etf_minute_all(
+def update_minute(
```

Update all tests and internal call sites to the new names. Add Chinese comments only around non-obvious behavior: all-code batching, one network retry, date bisection, validation-before-write, and independent cursors.

- [ ] **Step 4: Run the renamed behavior tests**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -q
```

Expected: all ETF tests pass.

---

### Task 2: Official Quota Adapter and Pure Quota Planning

**Files:**
- Modify: `my_utils/rqdata.py`
- Modify: `my_utils/rq_fun.py`
- Test: `tests/test_rq_etf_update.py`

**Interfaces:**
- Produces: `RqData.get_quota() -> dict`
- Produces: `quota_remaining_bytes(quota: dict, reserve_bytes: int) -> int | None`
- Produces: `measure_bytes_per_row(before: dict, after: dict, row_count: int) -> float`
- Produces: `select_minute_days_for_quota(trading_days, instruments, available_bytes, bytes_per_row, safety_factor=1.75) -> list[dt.date]`

- [ ] **Step 1: Write failing adapter and calculator tests**

```python
def test_rqdata_get_quota_delegates_once_to_official_api():
    from my_utils import rqdata

    expected = {"bytes_limit": 1024, "bytes_used": 100, "remaining_days": 14}
    with patch.object(rqdata.rq, "init"), patch.object(
        rqdata.rq.user, "get_quota", return_value=expected
    ) as get_quota:
        result = rqdata.RqData().get_quota()

    assert result is expected
    get_quota.assert_called_once_with()


def test_quota_helpers_measure_usage_and_keep_only_fitting_day_prefix():
    instruments = rq_fun.normalize_etf_instruments(sample_etf_instruments())
    days = [dt.date(2026, 8, 6), dt.date(2026, 8, 7)]
    bytes_per_row = rq_fun.measure_bytes_per_row(
        {"bytes_used": 100}, {"bytes_used": 580}, row_count=240
    )
    selected = rq_fun.select_minute_days_for_quota(
        days,
        instruments,
        available_bytes=1680,
        bytes_per_row=bytes_per_row,
        safety_factor=1.75,
    )

    assert bytes_per_row == 2.0
    assert selected == [days[0]]
```

Also test unlimited quota (`bytes_limit == 0`), decreasing/unchanged counters, zero rows, no fitting day, and strictly continuous prefixes.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k "quota" -q
```

Expected: FAIL because the adapter and pure helpers are missing.

- [ ] **Step 3: Add the official adapter**

Add to `RqData`:

```python
def get_quota(self):
    """读取官方账户的当日流量上限、已用量和剩余有效期。"""
    return rq.user.get_quota()
```

- [ ] **Step 4: Add pure quota helpers**

Implement in `my_utils/rq_fun.py`:

```python
def quota_remaining_bytes(quota: dict, reserve_bytes: int) -> int | None:
    """返回扣除安全余量后的可用字节；官方上限为 0 时返回 None 表示不限流量。"""
    limit = int(quota.get("bytes_limit", 0))
    used = int(quota.get("bytes_used", 0))
    if limit == 0:
        return None
    return max(0, limit - used - reserve_bytes)


def measure_bytes_per_row(before: dict, after: dict, row_count: int) -> float:
    """用一次真实行情调用前后的官方计数差，计算传输字节/实际返回行。"""
    if row_count <= 0:
        raise ValueError("row_count must be positive")
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
    """按交易日顺序返回当前额度可承受的最长连续前缀。"""
    if bytes_per_row <= 0 or safety_factor < 1:
        raise ValueError("quota estimates must be positive and conservative")
    days = sorted(set(trading_days))
    if available_bytes is None:
        return days

    selected = []
    estimated_bytes = 0.0
    for trade_date in days:
        active_count = len(
            filter_etf_codes_for_range(instruments, trade_date, trade_date)
        )
        day_bytes = active_count * 240 * bytes_per_row * safety_factor
        if estimated_bytes + day_bytes > available_bytes:
            break
        selected.append(trade_date)
        estimated_bytes += day_bytes
    return selected
```

For every day, estimate rows as `active_etf_count * 240`. If `available_bytes is None`, return every day. Stop before the first day that would exceed the available byte budget.

- [ ] **Step 5: Run quota and existing helper regressions**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py tests/test_rq_update_data.py -q
```

Expected: all tests pass.

---

### Task 3: Calibration, Safe Stop, and Failed-Day Cleanup

**Files:**
- Modify: `my_utils/rq_fun.py`
- Modify: `任务/米筐ETF数据更新.py`
- Test: `tests/test_rq_etf_update.py`

**Interfaces:**
- Produces: `cleanup_new_failed_partitions(save_dir, failed_dates, dates_before_run) -> list[dt.date]`
- Extends: `update_minute(source, instruments, trading_days, mode, max_rows=DEFAULT_MAX_MINUTE_ROWS, quota_reserve_bytes=128 * 1024 * 1024, quota_safety_factor=1.75) -> int`

- [ ] **Step 1: Write failing tests for safe cleanup**

```python
def test_cleanup_failed_partitions_never_deletes_preexisting_dates(tmp_path):
    old_day = dt.date(2026, 8, 6)
    new_day = dt.date(2026, 8, 7)
    old_partition = tmp_path / "minute" / f"trading_date={old_day}"
    new_partition = tmp_path / "minute" / f"trading_date={new_day}"
    old_partition.mkdir(parents=True)
    new_partition.mkdir(parents=True)

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        removed = rq_fun.cleanup_new_failed_partitions(
            "minute", [old_day, new_day], dates_before_run={old_day}
        )

    assert removed == [new_day]
    assert old_partition.exists()
    assert not new_partition.exists()
```

Add a test proving paths outside the requested save directory are never considered and a missing partition is a no-op.

- [ ] **Step 2: Run the cleanup tests and confirm RED**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k "cleanup_failed" -q
```

Expected: FAIL because `cleanup_new_failed_partitions` is missing.

- [ ] **Step 3: Implement exact, bounded cleanup**

Implement cleanup using exact `trading_date=YYYY-MM-DD` child paths under `data_dir(save_dir)`. Resolve the target root and each partition path before deletion, require the parent to equal the target root, skip any date in `dates_before_run`, and return the dates actually removed. Use `shutil.rmtree` only on those validated exact directories.

- [ ] **Step 4: Write failing orchestration tests**

Cover these behaviors with mocked RQData:

1. The first quota-safe batch of up to 3,000,000 rows is fetched once, validated, and written as calibration output.
2. Quota is read before and after calibration.
3. The observed byte cost selects a continuous suffix that fits after reserving 128 MiB.
4. If no second day fits, no second minute request is made.
5. `QuotaExceeded` propagates immediately, and cleanup receives only the failed batch dates plus the set of dates that existed before the run.
6. A second run starts from the partition after the previous run's last complete date.

- [ ] **Step 5: Run orchestration tests and confirm RED**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k "calibrat or quota_stop or failed_batch or independent" -q
```

Expected: FAIL because `update_minute` does not yet calibrate or stop by quota.

- [ ] **Step 6: Implement quota-aware minute orchestration**

In `update_minute`:

1. Snapshot `dates_before_run = set(get_existing_dates(RQ_ETF_MIN_DIR))`.
2. If no trading days, return 0.
3. Read quota and ensure at least the configured reserve remains.
4. Build the largest safe initial batch up to `max_rows`, then fetch, validate, and atomically save it as calibration.
5. Read quota again and call `measure_bytes_per_row`.
6. Call `quota_remaining_bytes` and `select_minute_days_for_quota` for the remaining days.
7. Feed only the selected days into `build_etf_minute_batches` and existing date-split handling.
8. Catch `FATAL_RQ_ERRORS`, call `cleanup_new_failed_partitions` only for the currently attempted dates, log the exact result, and re-raise.

The validation/save closure remains outside the request exception block so data-quality or disk errors are never mistaken for response-size errors.

- [ ] **Step 7: Add CLI reserve option and run routing tests**

Add:

```python
DEFAULT_QUOTA_RESERVE_MB = 128

parser.add_argument(
    "--quota-reserve-mb",
    type=_positive_int,
    default=DEFAULT_QUOTA_RESERVE_MB,
    help="每日流量安全余量（MiB），默认 128",
)
```

Pass `args.quota_reserve_mb * 1024 * 1024` into `update_minute`.

- [ ] **Step 8: Run the complete relevant regression suite**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py tests/test_rq_update_data.py -q
& 'E:\working\anaconda3\envs\quant\python.exe' -m py_compile my_utils\rq_fun.py my_utils\rqdata.py 任务\米筐ETF数据更新.py
```

Expected: all tests pass and compilation exits 0.

---

### Task 4: Execute Today's Incremental Update and Audit Output

**Files:**
- Write production data under: `E:\working\stock_data\rq_etf_day_data`
- Write production data under: `E:\working\stock_data\rq_1min_etf_data_dir`
- Write log: `任务/log/米筐ETF数据更新.log`

**Interfaces:**
- Executes: `main([])` through the task script.
- Consumes: official quota, ETF universe, trading calendar, daily prices, minute prices.

- [ ] **Step 1: Record pre-run quota and verify target directories**

Run a read-only script that prints `RqData.get_quota()`, existing partition ranges, and E-drive free space. Do not start if remaining quota is at or below the 128 MiB reserve.

- [ ] **Step 2: Run the default incremental updater**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' '任务\米筐ETF数据更新.py'
```

Allow it to complete its quota-selected range or stop on a fatal quota error. Do not retry the whole command on the same day after `QuotaExceeded`.

- [ ] **Step 3: Audit actual partitions and data quality**

Read every newly written partition's metadata and verify:

- dates form a continuous prefix of official trading days;
- daily schema equals `RQ_ETF_DAY_SCHEMA`;
- minute schema equals `RQ_ETF_MIN_SCHEMA`;
- no duplicate `(code, trading_date)` daily keys;
- no duplicate `(code, datetime)` minute keys;
- minute timestamps remain between 09:31 and 15:00 with no lunch-session bars;
- no `.rq-staging-*` or `.backup-*` directories remain;
- the logged rows and partition dates match the filesystem.

- [ ] **Step 4: Record post-run quota and report continuation command**

Print today's actual completed daily/minute trading-day counts, first/last dates, rows, bytes used, bytes remaining, and whether any failed partition was removed. Confirm that the next-day command is the same default invocation and that independent cursors resume automatically.
