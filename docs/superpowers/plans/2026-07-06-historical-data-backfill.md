# Historical Data Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe historical backfill controls, validate the 米筐 DDB source, then backfill 2018-2021 data according to方案 A.

**Architecture:** Keep daily incremental scripts compatible with `任务/run_update_data.bat`. Add CLI-driven historical paths to `任务/米筐数据更新.py` and `任务/数据更新v2.py`, with reusable small helpers for date partition cleanup and validation. Prefer DDB wide-range queries with local stock filtering; use official 米筐 API only if DDB quality checks fail.

**Tech Stack:** Python 3.9, Polars, Pandas, DolphinDB Python SDK, 掘金 API via `my_utils.stock_api`, pytest/unittest-style script checks, local Parquet partitions under `E:\working\stock_data`.

---

## File Structure

- Modify: `任务/米筐数据更新.py`
  - Add DDB quality gate.
  - Fix `rq_codes` filtering by applying local `instrument_base.type='CS'` stock filtering after wide DDB queries.
  - Clear existing date partitions inside each successful `mode='update'` batch before writing, so old 2021 holiday/non-stock partitions cannot remain.
  - Add `--quality-check-only`, `--batch-mode all|year|days`, and `--batch-size` CLI controls.
  - Keep existing insert defaults compatible with daily runs.
- Modify: `任务/数据更新v2.py`
  - Convert top-level execution into `main(argv=None)`.
  - Add historical CLI controls: `--start-date`, `--end-date`, `--mode`, `--skip-day`, `--skip-min`, `--min-align`.
  - Make day and minute update modes clear only the targeted date partitions before writing.
  - Preserve default daily behavior used by `run_update_data.bat`.
- Create: `tests/test_historical_backfill_scripts.py`
  - Unit tests for date partition cleanup, DDB stock filtering behavior, batch range generation, and CLI argument defaults.
- Use: `E:\working\anaconda3\envs\quant\python.exe`
  - Run all Python validation through the project `quant` environment.

---

### Task 1: Add Regression Tests For Backfill Helpers

**Files:**
- Create: `tests/test_historical_backfill_scripts.py`
- Test target: `任务/米筐数据更新.py`, `任务/数据更新v2.py`, `my_utils/rq_fun.py`

- [ ] **Step 1: Write import helper and tests**

Create `tests/test_historical_backfill_scripts.py` with this content:

```python
import datetime as dt
import importlib.util
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str, module_name: str):
    script_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rq_filter_to_stock_universe_removes_non_stock_codes():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_filter_test")
    raw = pl.DataFrame(
        {
            "order_book_id": ["000001.XSHE", "000001.XSHG", "H50066.XSHG", "600000.XSHG"],
            "trading_date": [dt.date(2021, 1, 4)] * 4,
            "close": [18.6, 3502.95, 137.73, 9.69],
        }
    )
    filtered = rq_update.filter_to_stock_universe(raw, ["000001.XSHE", "600000.XSHG"])
    assert filtered["order_book_id"].to_list() == ["000001.XSHE", "600000.XSHG"]


def test_rq_remove_existing_partitions_in_range_clears_only_target_dates(tmp_path):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_partition_test")
    rq_update.DATA_ROOT_DIR = str(tmp_path)
    target = tmp_path / "rq_stock_all_data"
    for date_text in ["2021-02-10", "2021-02-11", "2021-02-12"]:
        partition = target / f"trading_date={date_text}"
        partition.mkdir(parents=True)
        (partition / "part.parquet").write_text("partition marker", encoding="utf-8")

    rq_update.remove_existing_partitions_in_range(
        "rq_stock_all_data",
        dt.date(2021, 2, 11),
        dt.date(2021, 2, 12),
    )

    assert (target / "trading_date=2021-02-10").exists()
    assert not (target / "trading_date=2021-02-11").exists()
    assert not (target / "trading_date=2021-02-12").exists()


def test_rq_batch_ranges_all_mode_uses_single_range():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_batch_test")
    trading_days = [dt.date(2021, 1, 4), dt.date(2021, 1, 5), dt.date(2021, 1, 6)]
    ranges = rq_update.build_batch_ranges(trading_days, batch_mode="all", batch_size=60)
    assert ranges == [(dt.date(2021, 1, 4), dt.date(2021, 1, 6))]


def test_rq_batch_ranges_year_mode_splits_by_year():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_year_batch_test")
    trading_days = [dt.date(2020, 12, 31), dt.date(2021, 1, 4), dt.date(2021, 1, 5)]
    ranges = rq_update.build_batch_ranges(trading_days, batch_mode="year", batch_size=60)
    assert ranges == [
        (dt.date(2020, 12, 31), dt.date(2020, 12, 31)),
        (dt.date(2021, 1, 4), dt.date(2021, 1, 5)),
    ]


def test_gm_parse_args_defaults_preserve_daily_run():
    source = (PROJECT_ROOT / "任务/数据更新v2.py").read_text(encoding="utf-8")
    assert "def parse_args" in source
    assert 'if __name__ == "__main__"' in source
    gm_update = load_script("任务/数据更新v2.py", "gm_update_for_args_test")
    args = gm_update.parse_args([])
    assert args.mode == "insert"
    assert args.skip_day is False
    assert args.skip_min is False
    assert args.min_align == "both"
```

- [ ] **Step 2: Run tests to verify they fail before implementation**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py -q
```

Expected: FAIL because `filter_to_stock_universe`, `remove_existing_partitions_in_range`, `build_batch_ranges`, and `数据更新v2.py::parse_args` do not exist yet. The GM test checks source text before import, so it does not trigger the current top-level script.

- [ ] **Step 3: Commit failing tests**

Run:

```powershell
git add -- tests/test_historical_backfill_scripts.py
git commit --only -m "test: add historical backfill script regressions" -- tests/test_historical_backfill_scripts.py
```

Expected: commit succeeds without staging unrelated dirty files.

---

### Task 2: Implement 米筐 DDB Filtering, Batch Ranges, And Quality Gate

**Files:**
- Modify: `任务/米筐数据更新.py`
- Test: `tests/test_historical_backfill_scripts.py`

- [ ] **Step 1: Add stock filtering helper**

In `任务/米筐数据更新.py`, after `get_stock_universe`, add:

```python
def filter_to_stock_universe(data: pl.DataFrame, rq_codes: list[str]) -> pl.DataFrame:
    """
    按米筐股票池过滤 DDB 返回结果。

    DDB 历史补数优先采用少次数大范围查询，因此行情表允许先全量拉取
    `.XSHE/.XSHG` 后缀数据，再在本地用 instrument_base(type='CS') 股票池
    剔除指数、基金、债券等非股票代码。这样可以减少 DDB 请求次数，代价是
    单次传输会包含少量非股票数据。
    """
    if data.is_empty() or not rq_codes or "order_book_id" not in data.columns:
        return data
    return data.filter(pl.col("order_book_id").is_in(rq_codes))
```

- [ ] **Step 2: Apply filter inside `fetch_day_range`**

In `fetch_day_range`, immediately after `df = pl.from_pandas(kline)`, add:

```python
    df = filter_to_stock_universe(df, rq_codes)
    if df.is_empty():
        logging.info("  %s 无股票行情数据（全量查询后股票池过滤为空）", date_label)
        return pl.DataFrame(schema=RQ_DAY_SCHEMA)
```

Keep the later joins unchanged; the base `df` now only contains stock rows.

- [ ] **Step 3: Add batch range helper**

Before `update_all`, add:

```python
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
    return [(batch[0], batch[-1]) for batch in (days[i : i + batch_size] for i in range(0, len(days), batch_size))]
```

- [ ] **Step 4: Add update-range partition cleanup helper**

Before `update_day_range`, add:

```python
def remove_existing_partitions_in_range(
    save_dir: str,
    start_date: dt.date,
    end_date: dt.date,
) -> int:
    """
    删除指定日期范围内已有分区。

    update 重写时，新的股票过滤结果可能不再包含旧分区中的指数/节假日数据。
    因此不能只依赖 write_partitioned 清理“新数据里出现的日期”，还要在每个
    成功拉取到数据的批次写入前清理整个批次范围内的旧分区。
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
        logging.info("%s 清理旧分区: %s 个（%s ~ %s）", save_dir, removed, to_date_str(start_date), to_date_str(end_date))
    return removed
```

In `update_day_range`, after `all_data = fetch_day_range(...)` and after confirming `all_data` is not empty, add:

```python
    if mode == "update":
        remove_existing_partitions_in_range(RQ_DAY_DIR, start_date, end_date)
        remove_existing_partitions_in_range(RQ_ADJ_DIR, start_date, end_date)
```

This clears old holiday/non-stock partitions only after the new batch has been fetched successfully.

- [ ] **Step 5: Add DDB quality gate**

Before `parse_args`, add:

```python
def run_ddb_quality_gate(session: ddb.session) -> bool:
    """
    小样本检查 DDB 数据是否适合历史重写。

    检查重点不是要求米筐和掘金股票数量完全一致，而是确认 DDB 结果没有
    非股票污染、节假日分区污染和明显字段缺失。若失败，调用方应停止 DDB
    全量重写，改走官方米筐接口 fallback。
    """
    rq_codes = get_stock_universe(session)
    if not rq_codes:
        logging.error("质量门失败: 股票池为空")
        return False

    trade_sample = fetch_day_range(session, dt.date(2021, 1, 4), dt.date(2021, 1, 4), rq_codes=rq_codes)
    if trade_sample.is_empty():
        logging.error("质量门失败: 2021-01-04 股票行情为空")
        return False

    sample_codes = set(trade_sample["code"].to_list())
    if {"SHSE.000001", "SHSE.H50066", "SHSE.H50069", "SZSE.980001"} & sample_codes:
        logging.error("质量门失败: 样本交易日仍含非股票代码")
        return False

    holiday_sample = fetch_day_range(session, dt.date(2021, 2, 11), dt.date(2021, 2, 11), rq_codes=rq_codes)
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
```

- [ ] **Step 6: Add CLI args and update `update_all` signature**

Change:

```python
def update_all(session: ddb.session, start_date: dt.date, end_date: dt.date, mode: str) -> int:
```

to:

```python
def update_all(
    session: ddb.session,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    batch_mode: str = "days",
    batch_size: int = BATCH_SIZE,
) -> int:
```

Replace the `for i in range(0, n_days, BATCH_SIZE)` loop with:

```python
    batch_ranges = build_batch_ranges(trading_days, batch_mode=batch_mode, batch_size=batch_size)
    for idx, (batch_start, batch_end) in enumerate(batch_ranges, start=1):
        logging.info(
            "[%s/%s] 批次 %s ~ %s",
            idx, len(batch_ranges), to_date_str(batch_start), to_date_str(batch_end),
        )
        written = update_day_range(session, batch_start, batch_end, mode, rq_codes=rq_codes)
        total_written += written
```

In `parse_args`, add:

```python
    parser.add_argument("--quality-check-only", action="store_true", help="只运行 DDB 质量门，不写入本地数据")
    parser.add_argument("--skip-quality-check", action="store_true", help="跳过 DDB 质量门")
    parser.add_argument("--batch-mode", choices=["all", "year", "days"], default="days", help="DDB 批量查询模式")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch-mode=days 时每批交易日数量")
```

In `main`, after `session = create_ddb_session()`, add:

```python
        if args.quality_check_only:
            return 0 if run_ddb_quality_gate(session) else 2
        if not args.skip_quality_check and args.mode == "update":
            if not run_ddb_quality_gate(session):
                logging.error("DDB 质量门失败，停止 update 重写；请改用官方米筐接口 fallback")
                return 2
        update_all(session, start_date, end_date, args.mode, args.batch_mode, args.batch_size)
```

Remove the old single `update_all(session, start_date, end_date, args.mode)` call.

- [ ] **Step 7: Run targeted tests**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py -q
```

Expected: tests for 米筐 helpers pass; GM parse args may still fail until Task 3.

- [ ] **Step 8: Run DDB quality gate without writing**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\米筐数据更新.py --quality-check-only
```

Expected: exit code `0`, log contains `DDB 质量门通过`. If exit code `2`, stop DDB update and implement official API fallback before writing data.

- [ ] **Step 9: Commit 米筐 script changes**

Run:

```powershell
git add -- 任务\米筐数据更新.py
git commit --only -m "fix: filter 米筐 DDB backfill to stock universe" -- 任务\米筐数据更新.py
```

Expected: commit succeeds without unrelated files.

---

### Task 3: Implement 掘金 Historical CLI Without Breaking Daily Batch

**Files:**
- Modify: `任务/数据更新v2.py`
- Test: `tests/test_historical_backfill_scripts.py`

- [ ] **Step 1: Add imports and argument parser**

At the top of `任务/数据更新v2.py`, add:

```python
import argparse
```

After `api = stock_api()`, add:

```python
def parse_args(argv=None):
    """解析数据更新 v2 参数；默认值保持 run_update_data.bat 的日常增量行为。"""
    parser = argparse.ArgumentParser(description="掘金数据源数据更新脚本 v2")
    parser.add_argument("--start-date", default="2025-01-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default=datetime.date.today().strftime("%Y-%m-%d"), help="结束日期 YYYY-MM-DD")
    parser.add_argument("--mode", choices=["insert", "update"], default="insert", help="insert=增量; update=按指定范围覆盖")
    parser.add_argument("--skip-day", action="store_true", help="跳过日线更新")
    parser.add_argument("--skip-min", action="store_true", help="跳过 15 分钟更新")
    parser.add_argument("--min-align", choices=["left", "right", "both"], default="both", help="分钟线对齐方式")
    return parser.parse_args(argv)
```

- [ ] **Step 2: Add partition cleanup helper**

Before `update_day_data_gm`, add:

```python
def remove_date_partitions(save_dir: str, dates: list[datetime.date]) -> None:
    """只删除指定日期分区，用于历史 update 覆盖，避免影响范围外数据。"""
    import shutil

    target_dir = os.path.join(DATA_ROOT_DIR, save_dir)
    for date in dates:
        partition_dir = os.path.join(target_dir, f"trading_date={date:%Y-%m-%d}")
        if os.path.exists(partition_dir):
            shutil.rmtree(partition_dir)
            print(f"已清理旧分区: {partition_dir}")
```

- [ ] **Step 3: Make `update_day_data_gm` support update range**

At the start of `update_day_data_gm`, preserve the directory name before converting it to an absolute path:

```python
    save_dir_name = save_dir
    save_dir = os.path.join(DATA_ROOT_DIR, save_dir_name)
```

Inside the non-insert branch, replace `new_data = day_data` with:

```python
        new_data = day_data
        if mode == "update":
            dates_to_update = (
                new_data.select(pl.col("trading_date").unique().sort()).to_series().to_list()
            )
            remove_date_partitions(save_dir_name, dates_to_update)
```

This makes update mode overwrite only the partitions that are present in the fetched historical data.

- [ ] **Step 4: Wrap top-level flow in `main`**

Move the current top-level execution from `print("\n" + "=" * 70)` through the final completion print into:

```python
def main(argv=None) -> int:
    args = parse_args(argv)
    start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if start_date > end_date:
        print(f"起始日期大于结束日期，无需更新: {start_date} > {end_date}")
        return 0
    # 将当前顶层“步骤1/步骤2/步骤3/验证更新结果”逻辑整体缩进到 main 中，
    # 并把所有硬编码日期、模式和分钟线开关替换为 args 对应字段。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Replace hard-coded `start_date = datetime.date(2025, 1, 1)` behavior with `args.start_date`. Preserve insert mode inference for daily runs:

```python
    if args.mode == "insert" and exsist_data is not None:
        start_date = latest_date + datetime.timedelta(days=1)
```

For historical update mode, do not override the user-specified start date.

- [ ] **Step 5: Parameterize minute update**

In step 3 of the existing script, replace the hard-coded:

```python
exsist_data = read_day_data(
    start_date=datetime.date(2026, 1, 1),
    end_date=end_date,
    file_path='gm_stock_all_data'
)
```

with:

```python
min_start_date = start_date
exsist_data = read_day_data(
    start_date=min_start_date,
    end_date=end_date,
    file_path='gm_stock_all_data'
)
```

Guard minute updates:

```python
if not args.skip_min and exsist_data is not None and not exsist_data.is_empty():
    if args.min_align in ("left", "both"):
        update_min_data_by_day_data_gm(exsist_data, min_data_dir="15min_stock_data_dir", n=15, align="left")
    if args.min_align in ("right", "both"):
        update_min_data_by_day_data_gm(exsist_data, min_data_dir="15min_stock_data_right_dir", n=15, align="right")
```

- [ ] **Step 6: Make `update_min_data_by_day_data_gm` support update mode**

Change the function signature:

```python
def update_min_data_by_day_data_gm(
    day_data,
    min_data_dir='15min_stock_data_dir',
    n=15,
    align='left',
    mode='insert',
):
```

At the start of the function, preserve the directory name:

```python
    min_data_dir_name = min_data_dir
    min_data_dir = os.path.join(DATA_ROOT_DIR, min_data_dir_name)
```

Replace the current `dates_to_update` calculation with:

```python
    trading_dates = day_data.select(pl.col("trading_date").unique()).to_series().to_list()
    if mode == "update":
        dates_to_update = trading_dates
        remove_date_partitions(min_data_dir_name, dates_to_update)
    else:
        dates_to_update = [date for date in trading_dates if date not in existing_dates]
```

When calling the function from `main`, pass `mode=args.mode` for both left and right directories:

```python
update_min_data_by_day_data_gm(
    exsist_data,
    min_data_dir="15min_stock_data_dir",
    n=15,
    align="left",
    mode=args.mode,
)
```

- [ ] **Step 7: Run tests**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_historical_backfill_scripts.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Run CLI smoke tests without large writes**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py --start-date 2020-01-01 --end-date 2019-12-31
```

Expected: exit code `0`, output says start date is greater than end date.

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py --start-date 2021-01-04 --end-date 2021-01-04 --mode insert --skip-day --skip-min
```

Expected: exit code `0`, no data writes.

- [ ] **Step 9: Commit 掘金 CLI changes**

Run:

```powershell
git add -- 任务\数据更新v2.py tests/test_historical_backfill_scripts.py
git commit --only -m "feat: add historical controls to 数据更新v2" -- 任务\数据更新v2.py tests/test_historical_backfill_scripts.py
```

Expected: commit succeeds.

---

### Task 4: Execute 米筐 DDB Quality Gate And Historical Rewrite

**Files:**
- Runtime data: `E:\working\stock_data\rq_stock_all_data`
- Runtime data: `E:\working\stock_data\rq_adj`
- Logs: `log/米筐数据更新.log`

- [ ] **Step 1: Run quality gate**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\米筐数据更新.py --quality-check-only
```

Expected: exit code `0`.

- [ ] **Step 2: Rewrite 米筐 daily and adj data with low-query mode**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\米筐数据更新.py --start-date 2018-01-01 --end-date 2021-12-31 --mode update --batch-mode all
```

Expected: either succeeds, or fails due DDB memory/timeout.

- [ ] **Step 3: If all-mode fails, retry by year**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\米筐数据更新.py --start-date 2018-01-01 --end-date 2021-12-31 --mode update --batch-mode year
```

Expected: succeeds with lower per-query memory pressure. Do not retry with daily batches unless the user explicitly approves.

- [ ] **Step 4: Verify 米筐 output**

Run:

```powershell
@'
import polars as pl
from pathlib import Path

root = Path(r"E:\working\stock_data")
for dirname in ["rq_stock_all_data", "rq_adj"]:
    lf = pl.scan_parquet(str(root / dirname))
    stats = lf.select(
        pl.col("trading_date").min().alias("min_date"),
        pl.col("trading_date").max().alias("max_date"),
        pl.col("trading_date").n_unique().alias("n_dates"),
    ).collect()
    print(dirname, stats.to_dicts())

bad_codes = ["SHSE.000001", "SHSE.H50066", "SHSE.H50069", "SZSE.980001"]
bad = (
    pl.scan_parquet(str(root / "rq_stock_all_data"))
    .filter(pl.col("trading_date").is_between(pl.date(2018, 1, 1), pl.date(2021, 12, 31)))
    .filter(pl.col("code").is_in(bad_codes))
    .select("code", "trading_date")
    .limit(20)
    .collect()
)
print("bad_codes", bad)

for y, m, d in [(2021, 2, 11), (2021, 10, 4)]:
    cnt = (
        pl.scan_parquet(str(root / "rq_stock_all_data"))
        .filter(pl.col("trading_date") == pl.date(y, m, d))
        .select(pl.len().alias("rows"))
        .collect()
    )
    print(f"{y}-{m:02d}-{d:02d}", cnt.to_dicts())
'@ | E:\working\anaconda3\envs\quant\python.exe -
```

Expected:
- `rq_stock_all_data` and `rq_adj` min date no later than `2018-01-02`.
- `bad_codes` is empty.
- `2021-02-11` and `2021-10-04` rows are `0`.

---

### Task 5: Execute 掘金 Historical Daily And Minute Backfill

**Files:**
- Runtime data: `E:\working\stock_data\gm_stock_all_data`
- Runtime data: `E:\working\stock_data\15min_stock_data_dir`
- Runtime data: `E:\working\stock_data\15min_stock_data_right_dir`
- Logs: `log/数据更新v2.log`

- [ ] **Step 1: Backfill 掘金 daily only**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py --start-date 2018-01-01 --end-date 2020-12-31 --mode update --skip-min
```

Expected: `gm_stock_all_data` gets 2018-2020 date partitions.

- [ ] **Step 2: Backfill left-aligned 15 minute data by year**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py --start-date 2018-01-01 --end-date 2018-12-31 --mode update --skip-day --min-align left
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py --start-date 2019-01-01 --end-date 2019-12-31 --mode update --skip-day --min-align left
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py --start-date 2020-01-01 --end-date 2020-12-31 --mode update --skip-day --min-align left
```

Expected: `15min_stock_data_dir` gets 2018-2020 partitions. If a year run fails from API throttling, retry that year only.

- [ ] **Step 3: Backfill right-aligned 15 minute data by year**

Run:

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py --start-date 2018-01-01 --end-date 2018-12-31 --mode update --skip-day --min-align right
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py --start-date 2019-01-01 --end-date 2019-12-31 --mode update --skip-day --min-align right
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py --start-date 2020-01-01 --end-date 2020-12-31 --mode update --skip-day --min-align right
```

Expected: `15min_stock_data_right_dir` gets 2018-2020 partitions.

- [ ] **Step 4: Verify 掘金 output**

Run:

```powershell
@'
import polars as pl
from pathlib import Path

root = Path(r"E:\working\stock_data")
for dirname in ["gm_stock_all_data", "15min_stock_data_dir", "15min_stock_data_right_dir"]:
    lf = pl.scan_parquet(str(root / dirname))
    stats = lf.filter(
        pl.col("trading_date").is_between(pl.date(2018, 1, 1), pl.date(2020, 12, 31))
    ).select(
        pl.col("trading_date").min().alias("min_date"),
        pl.col("trading_date").max().alias("max_date"),
        pl.col("trading_date").n_unique().alias("n_dates"),
        pl.col("code").n_unique().alias("n_codes"),
    ).collect()
    print(dirname, stats.to_dicts())
'@ | E:\working\anaconda3\envs\quant\python.exe -
```

Expected:
- Each directory min date no later than `2018-01-02`.
- Each directory max date at least `2020-12-31` or the final 2020 trading date.
- `n_dates` is close to A-share trading day count for 2018-2020.

---

### Task 6: Final Audit And Status

**Files:**
- Inspect: `log/米筐数据更新.log`
- Inspect: `log/数据更新v2.log`
- Inspect: `E:\working\stock_data`

- [ ] **Step 1: Check logs for errors**

Run:

```powershell
Select-String -Path 'log\米筐数据更新.log','log\数据更新v2.log' -Pattern 'ERROR|失败|Traceback|timeout|超时|schema' -SimpleMatch
```

Expected: no fatal errors for the completed runs. Warnings about empty non-trading days are acceptable only if final partition verification passes.

- [ ] **Step 2: Run combined final verification**

Run:

```powershell
@'
import polars as pl
from pathlib import Path

root = Path(r"E:\working\stock_data")
checks = {
    "gm_stock_all_data": (2018, 2020),
    "15min_stock_data_dir": (2018, 2020),
    "15min_stock_data_right_dir": (2018, 2020),
    "rq_stock_all_data": (2018, 2021),
    "rq_adj": (2018, 2021),
}
for dirname, (start_year, end_year) in checks.items():
    lf = pl.scan_parquet(str(root / dirname))
    stats = lf.filter(
        pl.col("trading_date").is_between(pl.date(start_year, 1, 1), pl.date(end_year, 12, 31))
    ).select(
        pl.col("trading_date").min().alias("min_date"),
        pl.col("trading_date").max().alias("max_date"),
        pl.col("trading_date").n_unique().alias("n_dates"),
        pl.len().alias("rows"),
    ).collect()
    print(dirname, stats.to_dicts())

bad_codes = ["SHSE.000001", "SHSE.H50066", "SHSE.H50069", "SZSE.980001"]
bad_rows = (
    pl.scan_parquet(str(root / "rq_stock_all_data"))
    .filter(pl.col("trading_date").is_between(pl.date(2018, 1, 1), pl.date(2021, 12, 31)))
    .filter(pl.col("code").is_in(bad_codes))
    .select(pl.len().alias("bad_rows"))
    .collect()
)
print("rq_bad_rows", bad_rows.to_dicts())
'@ | E:\working\anaconda3\envs\quant\python.exe -
```

Expected:
- All target directories have non-empty coverage for the planned years.
- `rq_bad_rows` is `0`.

- [ ] **Step 3: Summarize completion**

Report:
- Code changes made.
- Which commands were run.
- Which ranges were backfilled.
- Final verification counts.
- Any fallback or retry used.
