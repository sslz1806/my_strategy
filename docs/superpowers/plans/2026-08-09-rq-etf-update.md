# RQData Official ETF Updater Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `任务/米筐ETF数据更新.py` to incrementally store all historical RQData ETFs' unadjusted daily bars and original end-timestamp 1-minute bars while minimizing official API calls.

**Architecture:** Extend the existing `RqData` adapter for one-call-per-date-batch official requests, keep ETF universe cleaning, normalization, validation, and row-budget batching as pure helpers in `my_utils/rq_fun.py`, and keep the new task script as thin orchestration. All applicable ETF codes travel in every request; only the trading-date range may split, and existing atomic partition writers remain the sole persistence path.

**Tech Stack:** Python 3.9, `rqdatac 3.4.7.7`, Pandas, Polars, pytest, `unittest.mock`, existing `my_utils.mapping` and `my_utils.rq_fun` partition helpers.

## Global Constraints

- Run Python and pytest with `E:\working\anaconda3\envs\quant\python.exe`.
- Add no new dependency and do not change the embedded RQData authentication mechanism in `RqData.__init__`.
- Preserve all current uncommitted user changes in `my_utils/rqdata.py`, `my_utils/rq_fun.py`, `tests/test_rq_update_data.py`, and `任务/米筐数据更新.py`; use localized patches only.
- Do not stage or commit implementation files during execution: `my_utils/rqdata.py` and `my_utils/rq_fun.py` were already dirty before this feature, so whole-file staging would capture unrelated user work and partial staging would create commits against a baseline that does not contain the authoritative current refactor. Use test-plus-diff review checkpoints after every task and leave implementation changes uncommitted unless the user separately authorizes a commit strategy.
- Include all `type='ETF'` instruments whose listing interval overlaps the requested dates, including delisted ETFs; exclude invalid exchange codes and future instruments outside the batch.
- Send all applicable ETF codes in one `get_price` call per date batch; never split by code.
- Use `adjust_type='none'`, `skip_suspended=False`, and `expect_df=True` for both daily and 1-minute requests.
- Preserve RQData's 1-minute Bar end timestamps (`09:31–11:30`, `13:01–15:00`) without shifting or aggregation.
- Default start date is `2021-01-01`; default minute request budget is `3_000_000` estimated rows.
- Write only to `rq_etf_day_data` and `rq_1min_etf_data_dir`, partitioned by `trading_date=YYYY-MM-DD`.
- Write logs to `任务/log/米筐ETF数据更新.log`.
- Automated tests must mock RQData. The final live smoke test uses one ETF and one completed historical trading day, makes exactly two price calls, and writes no production directory.

---

## File Structure

- Modify `my_utils/rq_fun.py`
  - Add `RQ_ETF_DAY_SCHEMA` and `RQ_ETF_MIN_SCHEMA`.
  - Add historical ETF-universe cleaning and date-overlap filtering.
  - Add daily and original 1-minute normalization.
  - Add `EtfRequestBatch`, row-budget batching, batch validation, and current-day minute completeness filtering.
  - Reuse existing `expand_rq_multiindex`, `align_schema`, `infer_start_date`, and `write_partitioned`.
- Modify `my_utils/rqdata.py`
  - Extend existing `RqData.get_price` without breaking its callers.
  - Add one-call ETF instrument, calendar, daily, and minute access methods.
- Create `任务/米筐ETF数据更新.py`
  - Add failure classification, one network retry, date bisection, day/minute pipelines, CLI, logging, and independent insert cursors.
- Create `tests/test_rq_etf_update.py`
  - Test pure helpers, adapter call shapes, fallback behavior, validation, CLI, and main routing without external requests.

---

### Task 1: ETF Universe, Schemas, and Raw-Bar Normalization

**Files:**
- Modify: `my_utils/rq_fun.py:31-75,346-392,1176-1190`
- Create: `tests/test_rq_etf_update.py`

**Interfaces:**
- Consumes: existing `expand_rq_multiindex(df, timestamp_col, shift_minutes=0)`, `align_schema(df, schema)`, and `convert_code_format`.
- Produces:
  - `RQ_ETF_DAY_SCHEMA: pl.Schema`
  - `RQ_ETF_MIN_SCHEMA: pl.Schema`
  - `normalize_etf_instruments(instruments: pd.DataFrame) -> pd.DataFrame`
  - `filter_etf_codes_for_range(instruments: pd.DataFrame, start_date: dt.date, end_date: dt.date) -> list[str]`
  - `normalize_etf_day_data(price_data: pd.DataFrame) -> pl.DataFrame`
  - `normalize_etf_minute_data(price_data: pd.DataFrame) -> pl.DataFrame`

- [ ] **Step 1: Create focused failing tests for the universe and schemas**

Create `tests/test_rq_etf_update.py` with these imports and tests:

```python
import datetime as dt
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import polars as pl
import pytest

from my_utils import rq_fun


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_etf_update_module(module_name: str):
    path = PROJECT_ROOT / "任务" / "米筐ETF数据更新.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_etf_instruments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_book_id": [
                "510300.XSHG",
                "159901.XSHE",
                "510010.XSHG",
                "159077.XSHE",
                "BAD.CODE",
            ],
            "type": ["ETF", "ETF", "ETF", "ETF", "ETF"],
            "listed_date": [
                "2012-05-28",
                "2006-04-24",
                "2013-03-25",
                "2026-08-12",
                "2020-01-01",
            ],
            "de_listed_date": [
                "0000-00-00",
                "0000-00-00",
                "2020-12-31",
                "0000-00-00",
                "0000-00-00",
            ],
            "status": ["Active", "Active", "Delisted", "Unknown", "Active"],
        }
    )


def test_etf_universe_keeps_delisted_history_and_excludes_future_and_bad_codes():
    instruments = rq_fun.normalize_etf_instruments(sample_etf_instruments())

    historical = rq_fun.filter_etf_codes_for_range(
        instruments, dt.date(2020, 1, 1), dt.date(2020, 12, 31)
    )
    current = rq_fun.filter_etf_codes_for_range(
        instruments, dt.date(2026, 8, 7), dt.date(2026, 8, 7)
    )

    assert historical == ["159901.XSHE", "510010.XSHG", "510300.XSHG"]
    assert current == ["159901.XSHE", "510300.XSHG"]
    assert instruments["listed_date"].map(type).eq(dt.date).all()
    assert instruments["de_listed_date"].map(type).eq(dt.date).all()


def test_etf_schemas_are_separate_from_stock_schemas():
    assert list(rq_fun.RQ_ETF_DAY_SCHEMA) == [
        "code",
        "trading_date",
        "pre_close",
        "open",
        "high",
        "low",
        "close",
        "change",
        "pct",
        "volume",
        "amount",
    ]
    assert list(rq_fun.RQ_ETF_MIN_SCHEMA) == [
        "code",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "trading_date",
    ]
    assert rq_fun.RQ_ETF_MIN_SCHEMA["datetime"] == pl.Datetime("us")
```

- [ ] **Step 2: Run the universe tests and confirm the missing-interface failure**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py::test_etf_universe_keeps_delisted_history_and_excludes_future_and_bad_codes tests/test_rq_etf_update.py::test_etf_schemas_are_separate_from_stock_schemas -v
```

Expected: FAIL because `RQ_ETF_DAY_SCHEMA`, `RQ_ETF_MIN_SCHEMA`, and the ETF universe helpers do not exist.

- [ ] **Step 3: Add the schemas and historical-universe helpers**

Add after `RQ_ADJ_SCHEMA` in `my_utils/rq_fun.py`:

```python
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
```

Add near the existing RQ parsing helpers:

```python
def _parse_etf_instrument_date(value, default: dt.date) -> dt.date:
    if pd.isna(value) or str(value) in {"", "0000-00-00", "NaT", "None"}:
        return default
    parsed = pd.to_datetime(value, errors="coerce")
    return default if pd.isna(parsed) else parsed.date()


def normalize_etf_instruments(instruments: pd.DataFrame) -> pd.DataFrame:
    """Normalize the complete RQData ETF history without applying a survivor filter."""
    required = {"order_book_id", "listed_date", "de_listed_date"}
    missing = required - set(instruments.columns)
    if missing:
        raise ValueError(f"ETF instruments missing columns: {sorted(missing)}")

    result = instruments.copy()
    if "type" in result.columns:
        result = result[result["type"].eq("ETF")]
    result = result[
        result["order_book_id"].astype(str).str.fullmatch(r"\d{6}\.(XSHG|XSHE)")
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
    """Return every ETF whose listing interval overlaps the requested date range."""
    if start_date > end_date:
        return []
    active = instruments[
        instruments["listed_date"].le(end_date)
        & instruments["de_listed_date"].ge(start_date)
    ]
    return sorted(active["order_book_id"].dropna().astype(str).unique().tolist())
```

- [ ] **Step 4: Add failing normalization tests**

Append:

```python
def test_normalize_etf_day_data_uses_reference_pre_close_and_rq_units():
    raw = pd.DataFrame(
        {
            "open": [11.0],
            "high": [11.25],
            "low": [10.88],
            "close": [11.24],
            "prev_close": [10.94],
            "volume": [203235546.0],
            "total_turnover": [2263042930.0],
        },
        index=pd.MultiIndex.from_arrays(
            [["000001.XSHE"], pd.to_datetime(["2026-06-12"])],
            names=["order_book_id", "date"],
        ),
    )

    result = rq_fun.normalize_etf_day_data(raw)

    assert result.schema == rq_fun.RQ_ETF_DAY_SCHEMA
    row = result.row(0, named=True)
    assert row["code"] == "SZSE.000001"
    assert row["pre_close"] == 10.94
    assert row["change"] == pytest.approx(0.30)
    assert row["pct"] == pytest.approx((11.24 / 10.94 - 1) * 100)
    assert row["volume"] == 203235546.0
    assert row["amount"] == 2263042930.0


def test_normalize_etf_minute_data_preserves_rq_end_timestamp_and_amount():
    raw = pd.DataFrame(
        {
            "open": [4.706, 4.713],
            "high": [4.716, 4.717],
            "low": [4.705, 4.711],
            "close": [4.713, 4.716],
            "volume": [20580149.0, 9736100.0],
            "total_turnover": [96958684.0, 45901975.0],
        },
        index=pd.MultiIndex.from_arrays(
            [
                ["510300.XSHG", "510300.XSHG"],
                pd.to_datetime(["2026-08-07 09:31:00", "2026-08-07 09:32:00"]),
            ],
            names=["order_book_id", "datetime"],
        ),
    )

    result = rq_fun.normalize_etf_minute_data(raw)

    assert result.schema == rq_fun.RQ_ETF_MIN_SCHEMA
    assert result["code"].to_list() == ["SHSE.510300", "SHSE.510300"]
    assert result["datetime"].to_list() == [
        dt.datetime(2026, 8, 7, 9, 31),
        dt.datetime(2026, 8, 7, 9, 32),
    ]
    assert result["amount"].to_list() == [96958684.0, 45901975.0]
    assert result["trading_date"].to_list() == [
        dt.date(2026, 8, 7),
        dt.date(2026, 8, 7),
    ]


def test_normalize_etf_data_rejects_missing_source_fields():
    raw = pd.DataFrame(
        {"close": [1.0]},
        index=pd.MultiIndex.from_arrays(
            [["510300.XSHG"], pd.to_datetime(["2026-08-07"])],
            names=["order_book_id", "date"],
        ),
    )

    with pytest.raises(ValueError, match="ETF day data missing columns"):
        rq_fun.normalize_etf_day_data(raw)
```

- [ ] **Step 5: Run the normalization tests and confirm they fail**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k 'normalize_etf' -v
```

Expected: FAIL because the normalization helpers do not exist.

- [ ] **Step 6: Implement exact day and minute normalization**

Add:

```python
def _require_rq_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def normalize_etf_day_data(price_data: pd.DataFrame) -> pl.DataFrame:
    """Convert unadjusted official RQData ETF day bars to the local ETF schema."""
    if price_data is None or price_data.empty:
        return pl.DataFrame(schema=RQ_ETF_DAY_SCHEMA)
    data = expand_rq_multiindex(price_data, timestamp_col="trading_date")
    _require_rq_columns(
        data,
        {"open", "high", "low", "close", "prev_close", "volume", "total_turnover"},
        "ETF day data",
    )
    result = pl.from_pandas(
        data.rename(columns={"prev_close": "pre_close", "total_turnover": "amount"})
    )
    result = result.with_columns(
        [
            pl.when(pl.col("pre_close").is_not_null() & (pl.col("pre_close") != 0))
            .then(pl.col("close") - pl.col("pre_close"))
            .otherwise(None)
            .alias("change"),
            pl.when(pl.col("pre_close").is_not_null() & (pl.col("pre_close") != 0))
            .then((pl.col("close") / pl.col("pre_close") - 1) * 100)
            .otherwise(None)
            .alias("pct"),
        ]
    )
    return align_schema(result, RQ_ETF_DAY_SCHEMA).sort(["trading_date", "code"])


def normalize_etf_minute_data(price_data: pd.DataFrame) -> pl.DataFrame:
    """Convert official 1-minute ETF bars without shifting RQData end timestamps."""
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
```

- [ ] **Step 7: Run Task 1 tests and shared RQ regressions**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py tests/test_rq_update_data.py -v
```

Expected: PASS.

- [ ] **Step 8: Record the Task 1 review checkpoint without staging dirty shared files**

```powershell
git diff --check -- my_utils/rq_fun.py tests/test_rq_etf_update.py
git status --short -- my_utils/rq_fun.py tests/test_rq_etf_update.py
```

Expected: diff check exits 0; `rq_fun.py` remains unstaged and the new test file remains untracked or unstaged.

---

### Task 2: Dynamic Minute Batches and Pre-Write Validation

**Files:**
- Modify: `my_utils/rq_fun.py` after the ETF normalization helpers
- Modify: `tests/test_rq_etf_update.py`

**Interfaces:**
- Consumes: normalized ETF instruments and `filter_etf_codes_for_range` from Task 1.
- Produces:
  - `EtfRequestBatch`
  - `build_etf_minute_batches(trading_days, instruments, max_rows=3_000_000) -> list[EtfRequestBatch]`
  - `validate_etf_day_batch(data, rq_codes, expected_dates, today=None) -> None`
  - `validate_etf_minute_batch(data, rq_codes, expected_dates, today=None) -> None`
  - `drop_incomplete_current_etf_minute_date(data, today=None) -> pl.DataFrame`

- [ ] **Step 1: Add failing row-budget batch tests**

Append:

```python
def test_build_etf_minute_batches_maximizes_days_within_row_budget():
    instruments = rq_fun.normalize_etf_instruments(
        pd.DataFrame(
            {
                "order_book_id": ["510300.XSHG", "159901.XSHE"],
                "type": ["ETF", "ETF"],
                "listed_date": ["2020-01-01", "2021-01-05"],
                "de_listed_date": ["0000-00-00", "0000-00-00"],
                "status": ["Active", "Active"],
            }
        )
    )
    days = [
        dt.date(2021, 1, 4),
        dt.date(2021, 1, 5),
        dt.date(2021, 1, 6),
        dt.date(2021, 1, 7),
    ]

    batches = rq_fun.build_etf_minute_batches(days, instruments, max_rows=720)

    assert [(batch.start_date, batch.end_date, batch.estimated_rows) for batch in batches] == [
        (dt.date(2021, 1, 4), dt.date(2021, 1, 5), 720),
        (dt.date(2021, 1, 6), dt.date(2021, 1, 6), 480),
        (dt.date(2021, 1, 7), dt.date(2021, 1, 7), 480),
    ]
    assert batches[0].rq_codes == ("159901.XSHE", "510300.XSHG")


def test_build_etf_minute_batches_keeps_an_oversized_day_whole():
    instruments = rq_fun.normalize_etf_instruments(sample_etf_instruments())
    day = dt.date(2020, 6, 1)

    batches = rq_fun.build_etf_minute_batches([day], instruments, max_rows=100)

    assert len(batches) == 1
    assert batches[0].trading_days == (day,)
    assert batches[0].estimated_rows == 3 * 240
```

- [ ] **Step 2: Run the batch tests and confirm they fail**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k 'build_etf_minute_batches' -v
```

Expected: FAIL because the batch interface does not exist.

- [ ] **Step 3: Implement the immutable request batch and greedy row-budget planner**

Add `from dataclasses import dataclass` to `rq_fun.py`, then add:

```python
@dataclass(frozen=True)
class EtfRequestBatch:
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
    """Greedily pack consecutive trading days without splitting the ETF universe."""
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
```

- [ ] **Step 4: Add failing validation and incomplete-current-day tests**

Append:

```python
def etf_day_frame(code="SHSE.510300", trade_date=dt.date(2026, 8, 7)):
    return pl.DataFrame(
        {
            "code": [code],
            "trading_date": [trade_date],
            "pre_close": [4.70],
            "open": [4.71],
            "high": [4.76],
            "low": [4.69],
            "close": [4.75],
            "change": [0.05],
            "pct": [0.05 / 4.70 * 100],
            "volume": [943535585.0],
            "amount": [4474693310.0],
        },
        schema=rq_fun.RQ_ETF_DAY_SCHEMA,
    )


def test_validate_etf_day_batch_rejects_unknown_code_duplicate_and_missing_history():
    day = dt.date(2026, 8, 7)
    with pytest.raises(RuntimeError, match="unexpected ETF codes"):
        rq_fun.validate_etf_day_batch(
            etf_day_frame(code="SHSE.999999"), ["510300.XSHG"], [day], today=dt.date(2026, 8, 9)
        )
    with pytest.raises(RuntimeError, match="duplicate keys"):
        rq_fun.validate_etf_day_batch(
            pl.concat([etf_day_frame(), etf_day_frame()]),
            ["510300.XSHG"],
            [day],
            today=dt.date(2026, 8, 9),
        )
    with pytest.raises(RuntimeError, match="missing trading days"):
        rq_fun.validate_etf_day_batch(
            pl.DataFrame(schema=rq_fun.RQ_ETF_DAY_SCHEMA),
            ["510300.XSHG"],
            [day],
            today=dt.date(2026, 8, 9),
        )


def test_drop_incomplete_current_minute_date_keeps_history_and_drops_partial_today():
    historical = dt.date(2026, 8, 7)
    today = dt.date(2026, 8, 10)
    rows = []
    for trade_date, count in [(historical, 240), (today, 30)]:
        for index in range(count):
            rows.append(
                {
                    "code": "SHSE.510300",
                    "datetime": dt.datetime.combine(trade_date, dt.time(9, 31))
                    + dt.timedelta(minutes=index),
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 0.0,
                    "amount": 0.0,
                    "trading_date": trade_date,
                }
            )
    data = pl.DataFrame(rows, schema=rq_fun.RQ_ETF_MIN_SCHEMA)

    result = rq_fun.drop_incomplete_current_etf_minute_date(data, today=today)

    assert result["trading_date"].unique().to_list() == [historical]
    rq_fun.validate_etf_minute_batch(
        result,
        ["510300.XSHG"],
        [historical, today],
        today=today,
    )


def test_validate_etf_minute_batch_rejects_date_overflow_and_datetime_mismatch():
    requested = dt.date(2026, 8, 7)
    outside = dt.date(2026, 8, 8)
    outside_data = pl.DataFrame(
        {
            "code": ["SHSE.510300"],
            "datetime": [dt.datetime(2026, 8, 8, 9, 31)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [0.0],
            "amount": [0.0],
            "trading_date": [outside],
        },
        schema=rq_fun.RQ_ETF_MIN_SCHEMA,
    )
    with pytest.raises(RuntimeError, match="outside requested dates"):
        rq_fun.validate_etf_minute_batch(
            outside_data,
            ["510300.XSHG"],
            [requested],
            today=dt.date(2026, 8, 9),
        )

    mismatched = outside_data.with_columns(
        pl.lit(requested).cast(pl.Date).alias("trading_date")
    )
    with pytest.raises(RuntimeError, match="does not match trading_date"):
        rq_fun.validate_etf_minute_batch(
            mismatched,
            ["510300.XSHG"],
            [requested],
            today=dt.date(2026, 8, 9),
        )
```

- [ ] **Step 5: Run validation tests and confirm they fail**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k 'validate_etf or incomplete_current' -v
```

Expected: FAIL because the validation interfaces do not exist.

- [ ] **Step 6: Implement shared validation and current-day minute filtering**

Add:

```python
def _validate_etf_batch(
    data: pl.DataFrame,
    rq_codes: list[str],
    expected_dates: list[dt.date],
    key_cols: list[str],
    today: dt.date | None = None,
) -> None:
    check_date = today or dt.date.today()
    allowed_codes = set(convert_code_format(rq_codes, format="gm"))
    returned_codes = set(data["code"].drop_nulls().unique().to_list())
    unexpected_codes = sorted(returned_codes - allowed_codes)
    if unexpected_codes:
        raise RuntimeError(f"unexpected ETF codes: {unexpected_codes}")

    expected = {to_date(value) for value in expected_dates}
    present = set(data["trading_date"].unique().to_list()) if data.height else set()
    unexpected_dates = sorted(present - expected)
    if unexpected_dates:
        raise RuntimeError(f"ETF data outside requested dates: {unexpected_dates}")

    missing_history = sorted(value for value in expected - present if value < check_date)
    if missing_history:
        raise RuntimeError(f"ETF data missing trading days: {missing_history}")

    if data.height and data.select(key_cols).is_duplicated().any():
        raise RuntimeError(f"ETF data duplicate keys: {key_cols}")


def validate_etf_day_batch(
    data: pl.DataFrame,
    rq_codes: list[str],
    expected_dates: list[dt.date],
    today: dt.date | None = None,
) -> None:
    _validate_etf_batch(
        data, rq_codes, expected_dates, ["code", "trading_date"], today=today
    )


def validate_etf_minute_batch(
    data: pl.DataFrame,
    rq_codes: list[str],
    expected_dates: list[dt.date],
    today: dt.date | None = None,
) -> None:
    _validate_etf_batch(
        data, rq_codes, expected_dates, ["code", "datetime"], today=today
    )
    if data.height and data.filter(
        pl.col("datetime").dt.date() != pl.col("trading_date")
    ).height:
        raise RuntimeError("ETF minute datetime does not match trading_date")


def drop_incomplete_current_etf_minute_date(
    data: pl.DataFrame,
    today: dt.date | None = None,
) -> pl.DataFrame:
    check_date = today or dt.date.today()
    current = data.filter(pl.col("trading_date") == check_date)
    if current.is_empty():
        return data
    max_bars = current.group_by("code").len()["len"].max()
    if max_bars is not None and max_bars >= 240:
        return data
    logging.warning("ETF 1min current trading day is incomplete and will not be written: %s", check_date)
    return data.filter(pl.col("trading_date") != check_date)
```

- [ ] **Step 7: Run Task 2 and shared tests**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py tests/test_rq_update_data.py -v
```

Expected: PASS.

- [ ] **Step 8: Record the Task 2 review checkpoint without staging dirty shared files**

```powershell
git diff --check -- my_utils/rq_fun.py tests/test_rq_etf_update.py
git status --short -- my_utils/rq_fun.py tests/test_rq_etf_update.py
```

Expected: diff check exits 0 and neither implementation file is staged.

---

### Task 3: Official RQData ETF Access Adapter

**Files:**
- Modify: `my_utils/rqdata.py:138-152`
- Modify: `tests/test_rq_etf_update.py`

**Interfaces:**
- Consumes: official module `rqdatac as rq` and the existing `RqData` initialization.
- Produces:
  - `ETF_DAY_FIELDS`
  - `ETF_MINUTE_FIELDS`
  - backward-compatible `RqData.get_price(..., adjust_type='pre', skip_suspended=False, expect_df=True)`
  - `RqData.get_etf_instruments() -> pd.DataFrame`
  - `RqData.get_trading_days(start_date, end_date) -> list[dt.date]`
  - `RqData.fetch_etf_day_range(rq_codes, start_date, end_date) -> pd.DataFrame`
  - `RqData.fetch_etf_minute_range(rq_codes, start_date, end_date) -> pd.DataFrame`

- [ ] **Step 1: Add failing adapter call-shape tests**

Append:

```python
def test_rqdata_etf_adapter_uses_one_unadjusted_call_for_all_codes():
    from my_utils import rqdata

    day_result = pd.DataFrame({"close": [1.0]})
    minute_result = pd.DataFrame({"close": [1.0]})
    with patch.object(rqdata.rq, "init"), patch.object(
        rqdata.rq, "get_price", side_effect=[day_result, minute_result]
    ) as get_price:
        source = rqdata.RqData()
        codes = ["510300.XSHG", "159915.XSHE"]
        day = source.fetch_etf_day_range(
            codes, dt.date(2026, 8, 7), dt.date(2026, 8, 7)
        )
        minute = source.fetch_etf_minute_range(
            codes, dt.date(2026, 8, 7), dt.date(2026, 8, 7)
        )

    assert day is day_result
    assert minute is minute_result
    assert get_price.call_args_list == [
        call(
            codes,
            start_date=dt.date(2026, 8, 7),
            end_date=dt.date(2026, 8, 7),
            frequency="1d",
            fields=rqdata.ETF_DAY_FIELDS,
            adjust_type="none",
            skip_suspended=False,
            expect_df=True,
        ),
        call(
            codes,
            start_date=dt.date(2026, 8, 7),
            end_date=dt.date(2026, 8, 7),
            frequency="1m",
            fields=rqdata.ETF_MINUTE_FIELDS,
            adjust_type="none",
            skip_suspended=False,
            expect_df=True,
        ),
    ]


def test_rqdata_etf_metadata_and_calendar_are_single_calls():
    from my_utils import rqdata

    instruments = sample_etf_instruments()
    with patch.object(rqdata.rq, "init"), patch.object(
        rqdata.rq, "all_instruments", return_value=instruments
    ) as all_instruments, patch.object(
        rqdata.rq,
        "get_trading_dates",
        return_value=[pd.Timestamp("2026-08-06"), pd.Timestamp("2026-08-07")],
    ) as get_trading_dates:
        source = rqdata.RqData()
        result_instruments = source.get_etf_instruments()
        days = source.get_trading_days(dt.date(2026, 8, 6), dt.date(2026, 8, 7))

    assert result_instruments is instruments
    assert days == [dt.date(2026, 8, 6), dt.date(2026, 8, 7)]
    all_instruments.assert_called_once_with(type="ETF", market="cn")
    get_trading_dates.assert_called_once_with(
        dt.date(2026, 8, 6), dt.date(2026, 8, 7), market="cn"
    )
```

- [ ] **Step 2: Run adapter tests and confirm they fail**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k 'rqdata_etf' -v
```

Expected: FAIL because the ETF constants and methods do not exist.

- [ ] **Step 3: Extend `RqData` without changing existing default behavior**

Add above `class RqData`:

```python
ETF_DAY_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "total_turnover",
]
ETF_MINUTE_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "total_turnover",
]
```

Replace only the existing `RqData.get_price` method and add the focused methods below it:

```python
    def get_price(
        self,
        symbol,
        start_date,
        end_date,
        frequency="1d",
        fields=None,
        adjust_type="pre",
        skip_suspended=False,
        expect_df=True,
    ):
        """Call official RQData while preserving the historical default pre-adjust behavior."""
        return rq.get_price(
            symbol,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            fields=fields,
            adjust_type=adjust_type,
            skip_suspended=skip_suspended,
            expect_df=expect_df,
        )

    def get_etf_instruments(self):
        """Fetch the complete ETF instrument history, including delisted records."""
        return rq.all_instruments(type="ETF", market="cn")

    def get_trading_days(self, start_date, end_date):
        """Fetch one reusable official China-market trading calendar range."""
        values = rq.get_trading_dates(start_date, end_date, market="cn")
        return [value.date() if hasattr(value, "date") else value for value in values]

    def fetch_etf_day_range(self, rq_codes, start_date, end_date):
        """Fetch all applicable ETF day bars in one unadjusted request."""
        return self.get_price(
            rq_codes,
            start_date,
            end_date,
            frequency="1d",
            fields=ETF_DAY_FIELDS,
            adjust_type="none",
            skip_suspended=False,
            expect_df=True,
        )

    def fetch_etf_minute_range(self, rq_codes, start_date, end_date):
        """Fetch all applicable original ETF 1-minute bars in one unadjusted request."""
        return self.get_price(
            rq_codes,
            start_date,
            end_date,
            frequency="1m",
            fields=ETF_MINUTE_FIELDS,
            adjust_type="none",
            skip_suspended=False,
            expect_df=True,
        )
```

Do not catch RQData exceptions here; the task script needs their exact types to decide between fatal stop, one retry, and date bisection.

- [ ] **Step 4: Run adapter and existing multi-source compatibility tests**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k 'rqdata_etf' -v
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_update_data.py -v
```

Expected: PASS. Existing `RqData.get_return` continues receiving the original default `adjust_type='pre'`.

- [ ] **Step 5: Record the Task 3 review checkpoint without staging dirty shared files**

```powershell
git diff --check -- my_utils/rqdata.py tests/test_rq_etf_update.py
git status --short -- my_utils/rqdata.py tests/test_rq_etf_update.py
```

Expected: diff check exits 0 and the pre-existing `rqdata.py` work remains unstaged.

---

### Task 4: Date Bisection and Quota-Aware Batch Pipelines

**Files:**
- Create: `任务/米筐ETF数据更新.py`
- Modify: `tests/test_rq_etf_update.py`

**Interfaces:**
- Consumes: Task 1–3 helpers, `RqData`, existing `write_partitioned`, and RQData exception classes.
- Produces:
  - `fetch_with_date_fallback(fetch_once, consume, trading_days, sleep_func=time.sleep) -> int`
  - `update_etf_day_all(source, instruments, trading_days, mode) -> int`
  - `update_etf_minute_all(source, instruments, trading_days, mode, max_rows) -> int`

- [ ] **Step 1: Add failing fallback behavior tests**

Append:

```python
def test_fetch_with_date_fallback_bisects_gateway_error_without_code_splitting():
    module = load_etf_update_module("rq_etf_fallback_split_test")
    days = [dt.date(2026, 8, 6), dt.date(2026, 8, 7)]
    fetch_calls = []
    consumed = []

    def fetch_once(batch_days):
        fetch_calls.append(tuple(batch_days))
        if len(batch_days) == 2:
            raise module.GatewayError("response too large")
        return pd.DataFrame({"day": batch_days})

    def consume(raw, batch_days):
        consumed.append(tuple(batch_days))
        return len(raw)

    written = module.fetch_with_date_fallback(fetch_once, consume, days)

    assert written == 2
    assert fetch_calls == [(days[0], days[1]), (days[0],), (days[1],)]
    assert consumed == [(days[0],), (days[1],)]


def test_fetch_with_date_fallback_retries_network_once_and_never_retries_quota():
    module = load_etf_update_module("rq_etf_fallback_retry_test")
    day = dt.date(2026, 8, 7)
    sleeps = []
    network_fetch = MagicMock(side_effect=[ConnectionError("reset"), pd.DataFrame({"x": [1]})])

    written = module.fetch_with_date_fallback(
        network_fetch,
        lambda raw, batch_days: len(raw),
        [day],
        sleep_func=sleeps.append,
    )

    assert written == 1
    assert network_fetch.call_count == 2
    assert sleeps == [3.0]

    quota_fetch = MagicMock(side_effect=module.QuotaExceeded("quota exhausted"))
    with pytest.raises(module.QuotaExceeded):
        module.fetch_with_date_fallback(
            quota_fetch,
            lambda raw, batch_days: len(raw),
            [day],
            sleep_func=sleeps.append,
        )
    assert quota_fetch.call_count == 1


@pytest.mark.parametrize("error_name", ["AuthenticationFailed", "PermissionDenied", "QuotaExceeded"])
def test_fetch_with_date_fallback_never_retries_fatal_rq_errors(error_name):
    module = load_etf_update_module(f"rq_etf_fatal_{error_name}_test")
    error_type = getattr(module, error_name)
    fetch = MagicMock(side_effect=error_type("fatal account error"))

    with pytest.raises(error_type):
        module.fetch_with_date_fallback(
            fetch,
            lambda raw, batch_days: 0,
            [dt.date(2026, 8, 7)],
        )
    assert fetch.call_count == 1


def test_fetch_with_date_fallback_stops_when_single_day_is_still_too_large():
    module = load_etf_update_module("rq_etf_fallback_single_day_test")
    fetch = MagicMock(side_effect=module.GatewayError("still too large"))

    with pytest.raises(module.GatewayError):
        module.fetch_with_date_fallback(
            fetch,
            lambda raw, batch_days: 0,
            [dt.date(2026, 8, 7)],
        )
    assert fetch.call_count == 1


def test_fetch_with_date_fallback_does_not_bisect_validation_or_write_failures():
    module = load_etf_update_module("rq_etf_fallback_consume_failure_test")
    fetch = MagicMock(return_value=pd.DataFrame({"x": [1]}))
    consume = MagicMock(side_effect=RuntimeError("duplicate keys"))
    days = [dt.date(2026, 8, 6), dt.date(2026, 8, 7)]

    with pytest.raises(RuntimeError, match="duplicate keys"):
        module.fetch_with_date_fallback(fetch, consume, days)
    fetch.assert_called_once_with(days)
    consume.assert_called_once()
```

- [ ] **Step 2: Run fallback tests and confirm the missing-script failure**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k 'fetch_with_date_fallback' -v
```

Expected: FAIL because `任务/米筐ETF数据更新.py` does not exist.

- [ ] **Step 3: Create the script foundation and exact fallback function**

Create `任务/米筐ETF数据更新.py` with imports, constants, and fallback logic:

```python
"""Use official RQData to update all historical ETF day and original 1-minute bars."""
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
    drop_incomplete_current_etf_minute_date,
    filter_etf_codes_for_range,
    infer_start_date,
    normalize_etf_day_data,
    normalize_etf_instruments,
    normalize_etf_minute_data,
    to_date,
    validate_etf_day_batch,
    validate_etf_minute_batch,
    write_partitioned,
)

RQ_ETF_DAY_DIR = "rq_etf_day_data"
RQ_ETF_MIN_DIR = "rq_1min_etf_data_dir"
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_MAX_MINUTE_ROWS = 3_000_000
FATAL_RQ_ERRORS = (AuthenticationFailed, PermissionDenied, QuotaExceeded)
SPLITTABLE_RQ_ERRORS = (GatewayError, TimeoutError)


def fetch_with_date_fallback(
    fetch_once: Callable[[list[dt.date]], pd.DataFrame],
    consume: Callable[[pd.DataFrame, list[dt.date]], int],
    trading_days: list[dt.date],
    sleep_func: Callable[[float], None] = time.sleep,
) -> int:
    """Fetch one all-code date batch, retry one network reset, or bisect by date."""
    days = sorted(set(trading_days))
    if not days:
        return 0

    try:
        try:
            raw = fetch_once(days)
        except FATAL_RQ_ERRORS:
            raise
        except SPLITTABLE_RQ_ERRORS:
            raise
        except ConnectionError as exc:
            logging.warning("RQData network error; retrying this batch once: %s", exc)
            sleep_func(3.0)
            raw = fetch_once(days)
    except FATAL_RQ_ERRORS:
        raise
    except SPLITTABLE_RQ_ERRORS as exc:
        if len(days) == 1:
            logging.error("RQData single-day batch still failed: %s", exc)
            raise
        middle = len(days) // 2
        logging.warning(
            "RQData batch %s ~ %s failed; bisecting at the trading-day boundary: %s",
            days[0],
            days[-1],
            exc,
        )
        return fetch_with_date_fallback(
            fetch_once, consume, days[:middle], sleep_func=sleep_func
        ) + fetch_with_date_fallback(
            fetch_once, consume, days[middle:], sleep_func=sleep_func
        )

    return consume(raw, days)
```

Keep normalization, validation, and writes outside the request `try` block so a data or disk error aborts instead of wasting quota on date bisection.

- [ ] **Step 4: Run fallback tests**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k 'fetch_with_date_fallback' -v
```

Expected: PASS.

- [ ] **Step 5: Add failing end-to-end batch pipeline tests with temporary storage**

Append:

```python
def test_day_pipeline_passes_all_overlapping_codes_in_one_call_and_writes_partition(tmp_path):
    module = load_etf_update_module("rq_etf_day_pipeline_test")
    instruments = rq_fun.normalize_etf_instruments(sample_etf_instruments())
    day = dt.date(2020, 6, 1)
    raw = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "prev_close": [1.0, 2.0, 3.0],
            "volume": [100.0, 200.0, 300.0],
            "total_turnover": [105.0, 410.0, 915.0],
        },
        index=pd.MultiIndex.from_arrays(
            [
                ["159901.XSHE", "510010.XSHG", "510300.XSHG"],
                pd.to_datetime([day, day, day]),
            ],
            names=["order_book_id", "date"],
        ),
    )
    source = MagicMock()
    source.fetch_etf_day_range.return_value = raw

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        written = module.update_etf_day_all(
            source, instruments, [day], mode="insert"
        )

    source.fetch_etf_day_range.assert_called_once_with(
        ["159901.XSHE", "510010.XSHG", "510300.XSHG"], day, day
    )
    assert written == 3
    saved = pl.read_parquet(
        str(tmp_path / "rq_etf_day_data" / f"trading_date={day}" / "*.parquet")
    )
    assert saved.schema == rq_fun.RQ_ETF_DAY_SCHEMA


def test_minute_pipeline_uses_dynamic_batches_and_preserves_0931(tmp_path):
    module = load_etf_update_module("rq_etf_minute_pipeline_test")
    instruments = rq_fun.normalize_etf_instruments(
        pd.DataFrame(
            {
                "order_book_id": ["510300.XSHG"],
                "type": ["ETF"],
                "listed_date": ["2012-05-28"],
                "de_listed_date": ["0000-00-00"],
                "status": ["Active"],
            }
        )
    )
    day = dt.date(2026, 8, 7)
    datetimes = [
        dt.datetime.combine(day, dt.time(9, 31)) + dt.timedelta(minutes=index)
        for index in range(120)
    ] + [
        dt.datetime.combine(day, dt.time(13, 1)) + dt.timedelta(minutes=index)
        for index in range(120)
    ]
    raw = pd.DataFrame(
        {
            "open": [1.0] * 240,
            "high": [1.0] * 240,
            "low": [1.0] * 240,
            "close": [1.0] * 240,
            "volume": [0.0] * 240,
            "total_turnover": [0.0] * 240,
        },
        index=pd.MultiIndex.from_arrays(
            [["510300.XSHG"] * 240, pd.to_datetime(datetimes)],
            names=["order_book_id", "datetime"],
        ),
    )
    source = MagicMock()
    source.fetch_etf_minute_range.return_value = raw

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        written = module.update_etf_minute_all(
            source, instruments, [day], mode="insert", max_rows=3_000_000
        )

    source.fetch_etf_minute_range.assert_called_once_with(["510300.XSHG"], day, day)
    assert written == 240
    saved = pl.read_parquet(
        str(tmp_path / "rq_1min_etf_data_dir" / f"trading_date={day}" / "*.parquet")
    ).sort("datetime")
    assert saved["datetime"][0] == dt.datetime(2026, 8, 7, 9, 31)
    assert saved["datetime"][-1] == dt.datetime(2026, 8, 7, 15, 0)
```

- [ ] **Step 6: Run pipeline tests and confirm the missing-pipeline failure**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k 'pipeline' -v
```

Expected: FAIL because `update_etf_day_all` and `update_etf_minute_all` do not exist.

- [ ] **Step 7: Implement the day and minute pipelines using the shared writer**

Add to the task script:

```python
def update_etf_day_all(source, instruments, trading_days, mode: str) -> int:
    """Fetch the full pending day range, bisect only on request-size failures."""
    def fetch(days: list[dt.date]) -> pd.DataFrame:
        rq_codes = filter_etf_codes_for_range(instruments, days[0], days[-1])
        logging.info("ETF day request: %s ~ %s | %s codes", days[0], days[-1], len(rq_codes))
        return source.fetch_etf_day_range(rq_codes, days[0], days[-1])

    def consume(raw: pd.DataFrame, days: list[dt.date]) -> int:
        rq_codes = filter_etf_codes_for_range(instruments, days[0], days[-1])
        data = normalize_etf_day_data(raw)
        validate_etf_day_batch(data, rq_codes, days)
        logging.info(
            "ETF day response: %s ~ %s | %s rows", days[0], days[-1], data.height
        )
        return write_partitioned(data, RQ_ETF_DAY_DIR, RQ_ETF_DAY_SCHEMA, mode)

    return fetch_with_date_fallback(fetch, consume, trading_days)


def update_etf_minute_all(
    source,
    instruments,
    trading_days,
    mode: str,
    max_rows: int,
) -> int:
    """Process maximal row-budget batches and preserve every successful checkpoint."""
    total_written = 0
    batches = build_etf_minute_batches(trading_days, instruments, max_rows=max_rows)
    for index, batch in enumerate(batches, start=1):
        logging.info(
            "ETF 1min batch %s/%s: %s ~ %s | %s codes | estimated %s rows",
            index,
            len(batches),
            batch.start_date,
            batch.end_date,
            len(batch.rq_codes),
            batch.estimated_rows,
        )

        def fetch(days: list[dt.date]) -> pd.DataFrame:
            rq_codes = filter_etf_codes_for_range(instruments, days[0], days[-1])
            return source.fetch_etf_minute_range(rq_codes, days[0], days[-1])

        def consume(raw: pd.DataFrame, days: list[dt.date]) -> int:
            rq_codes = filter_etf_codes_for_range(instruments, days[0], days[-1])
            data = normalize_etf_minute_data(raw)
            data = drop_incomplete_current_etf_minute_date(data)
            validate_etf_minute_batch(data, rq_codes, days)
            logging.info(
                "ETF 1min response: %s ~ %s | %s rows",
                days[0],
                days[-1],
                data.height,
            )
            return write_partitioned(data, RQ_ETF_MIN_DIR, RQ_ETF_MIN_SCHEMA, mode)

        total_written += fetch_with_date_fallback(
            fetch, consume, list(batch.trading_days)
        )
    return total_written
```

- [ ] **Step 8: Run Task 4 and shared writer regression tests**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -v
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_update_data.py -k 'write_partitioned or partition' -v
```

Expected: PASS.

- [ ] **Step 9: Record the Task 4 review checkpoint without staging implementation files**

```powershell
git diff --check -- '任务/米筐ETF数据更新.py' tests/test_rq_etf_update.py
git status --short -- '任务/米筐ETF数据更新.py' tests/test_rq_etf_update.py
```

Expected: diff check exits 0; the new script and tests remain unstaged for final review together with the shared-file changes they depend on.

---

### Task 5: CLI, Independent Cursors, Full Verification, and Live Smoke

**Files:**
- Modify: `任务/米筐ETF数据更新.py`
- Modify: `tests/test_rq_etf_update.py`

**Interfaces:**
- Consumes: all Task 1–4 interfaces.
- Produces:
  - `parse_args(argv=None) -> argparse.Namespace`
  - `main(argv=None) -> int`
  - executable `day|min|all` production entry point.

- [ ] **Step 1: Add failing CLI and independent-cursor tests**

Append:

```python
def test_etf_cli_defaults_match_the_approved_design():
    module = load_etf_update_module("rq_etf_cli_defaults_test")

    args = module.parse_args([])

    assert args.start_date == "2021-01-01"
    assert args.mode == "insert"
    assert args.data_type == "all"
    assert args.max_minute_rows == 3_000_000


@pytest.mark.parametrize("data_type", ["day", "min", "all"])
def test_etf_cli_accepts_every_supported_data_type(data_type):
    module = load_etf_update_module(f"rq_etf_cli_{data_type}_test")

    args = module.parse_args(["--data-type", data_type])

    assert args.data_type == data_type


def test_etf_main_keeps_day_and_minute_insert_cursors_independent():
    module = load_etf_update_module("rq_etf_main_cursor_test")
    minute_start = dt.date(2026, 8, 7)
    end = dt.date(2026, 8, 7)
    fake_source = MagicMock()
    fake_source.get_etf_instruments.return_value = sample_etf_instruments()
    fake_source.get_trading_days.return_value = [minute_start]

    with patch.object(module, "get_logger"), patch.object(
        module, "infer_start_date", side_effect=[None, minute_start]
    ) as infer_start, patch.object(
        module, "RqData", return_value=fake_source
    ), patch.object(
        module, "update_etf_day_all"
    ) as update_day, patch.object(
        module, "update_etf_minute_all", return_value=240
    ) as update_minute:
        result = module.main(
            [
                "--start-date", "2021-01-01",
                "--end-date", "2026-08-07",
                "--mode", "insert",
                "--data-type", "all",
            ]
        )

    assert result == 0
    assert infer_start.call_args_list == [
        call(dt.date(2021, 1, 1), "rq_etf_day_data", "insert", end_date=end),
        call(dt.date(2021, 1, 1), "rq_1min_etf_data_dir", "insert", end_date=end),
    ]
    update_day.assert_not_called()
    update_minute.assert_called_once()
    fake_source.get_etf_instruments.assert_called_once_with()
    fake_source.get_trading_days.assert_called_once_with(minute_start, end)
```

- [ ] **Step 2: Run CLI tests and confirm they fail**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -k 'etf_cli or etf_main' -v
```

Expected: FAIL because `parse_args` and `main` do not exist.

- [ ] **Step 3: Implement the exact CLI and one-call metadata/calendar main flow**

Add to the task script:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="米筐官方 ETF 日线与 1 分钟数据更新")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="起始日期 YYYY-MM-DD")
    parser.add_argument(
        "--end-date",
        default=dt.date.today().strftime("%Y-%m-%d"),
        help="结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--mode",
        choices=["insert", "update"],
        default="insert",
        help="insert=从最新分区续更；update=重写指定日期范围",
    )
    parser.add_argument(
        "--data-type",
        choices=["day", "min", "all"],
        default="all",
        help="day=日线；min=原始1分钟；all=两者",
    )
    parser.add_argument(
        "--max-minute-rows",
        type=int,
        default=DEFAULT_MAX_MINUTE_ROWS,
        help="每个1分钟请求的预计最大行数",
    )
    args = parser.parse_args(argv)
    if args.max_minute_rows <= 0:
        parser.error("--max-minute-rows 必须为正整数")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    get_logger(log_file="任务/log/米筐ETF数据更新.log", inherit=False)

    start_date = to_date(args.start_date)
    end_date = to_date(args.end_date)
    if start_date > end_date:
        logging.info("起始日期大于结束日期，无需更新: %s > %s", start_date, end_date)
        return 0

    day_start = start_date if args.data_type in ("day", "all") else None
    minute_start = start_date if args.data_type in ("min", "all") else None
    if args.mode == "insert":
        if day_start is not None:
            day_start = infer_start_date(
                start_date, RQ_ETF_DAY_DIR, args.mode, end_date=end_date
            )
        if minute_start is not None:
            minute_start = infer_start_date(
                start_date, RQ_ETF_MIN_DIR, args.mode, end_date=end_date
            )
    active_starts = [value for value in (day_start, minute_start) if value is not None]
    if not active_starts:
        logging.info("ETF 日线和 1 分钟目录均已是最新")
        return 0

    source = RqData()
    raw_instruments = source.get_etf_instruments()
    instruments = normalize_etf_instruments(raw_instruments)
    if instruments.empty:
        raise RuntimeError("米筐 ETF 历史池为空")
    status_counts = raw_instruments.get("status", pd.Series(dtype=str)).value_counts().to_dict()
    logging.info("ETF 历史池: %s | 状态: %s", len(instruments), status_counts)

    calendar_start = min(active_starts)
    trading_days = source.get_trading_days(calendar_start, end_date)
    if not trading_days:
        logging.info("指定范围没有交易日: %s ~ %s", calendar_start, end_date)
        return 0

    if day_start is not None:
        day_days = [value for value in trading_days if value >= day_start]
        day_written = update_etf_day_all(source, instruments, day_days, args.mode)
        logging.info("ETF 日线本次写入: %s 行", day_written)
    else:
        logging.info("ETF 日线已跳过：目录已是最新或未选择 day")

    if minute_start is not None:
        minute_days = [value for value in trading_days if value >= minute_start]
        minute_written = update_etf_minute_all(
            source,
            instruments,
            minute_days,
            args.mode,
            max_rows=args.max_minute_rows,
        )
        logging.info("ETF 1分钟本次写入: %s 行", minute_written)
    else:
        logging.info("ETF 1分钟已跳过：目录已是最新或未选择 min")

    logging.info("米筐 ETF 数据更新完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run all new tests and fix only observed failures**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_etf_update.py -v
```

Expected: PASS.

- [ ] **Step 5: Run shared RQ and syntax regression checks**

Run:

```powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest tests/test_rq_update_data.py tests/test_rq_etf_update.py -v
& 'E:\working\anaconda3\envs\quant\python.exe' -m py_compile my_utils/rqdata.py my_utils/rq_fun.py '任务/米筐ETF数据更新.py'
```

Expected: all tests PASS and `py_compile` exits 0.

- [ ] **Step 6: Run the two-call official smoke test without writing any directory**

Run this from the repository root:

```powershell
@'
import datetime as dt
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl

from my_utils import rq_fun
from my_utils.rqdata import RqData
from my_utils.rq_fun import (
    RQ_ETF_DAY_SCHEMA,
    RQ_ETF_MIN_SCHEMA,
    normalize_etf_day_data,
    normalize_etf_minute_data,
    write_partitioned,
)

source = RqData()
code = "510300.XSHG"
trade_date = dt.date(2026, 8, 7)
day = normalize_etf_day_data(source.fetch_etf_day_range([code], trade_date, trade_date))
minute = normalize_etf_minute_data(source.fetch_etf_minute_range([code], trade_date, trade_date))

assert day.schema == RQ_ETF_DAY_SCHEMA
assert day.height == 1
assert day["code"].to_list() == ["SHSE.510300"]
assert minute.schema == RQ_ETF_MIN_SCHEMA
assert minute.height == 240
assert minute["datetime"].min() == dt.datetime(2026, 8, 7, 9, 31)
assert minute["datetime"].max() == dt.datetime(2026, 8, 7, 15, 0)
with tempfile.TemporaryDirectory() as temp_dir, patch.object(
    rq_fun, "DATA_ROOT_DIR", temp_dir
):
    assert write_partitioned(day, "rq_etf_day_data", RQ_ETF_DAY_SCHEMA, "insert") == 1
    assert write_partitioned(minute, "rq_1min_etf_data_dir", RQ_ETF_MIN_SCHEMA, "insert") == 240
    saved_day = pl.read_parquet(
        str(Path(temp_dir) / "rq_etf_day_data" / "trading_date=2026-08-07" / "*.parquet")
    )
    saved_minute = pl.read_parquet(
        str(Path(temp_dir) / "rq_1min_etf_data_dir" / "trading_date=2026-08-07" / "*.parquet")
    )
    assert saved_day.schema == RQ_ETF_DAY_SCHEMA
    assert saved_minute.schema == RQ_ETF_MIN_SCHEMA
print("RQData ETF smoke passed: day=1, minute=240, timestamps=09:31~15:00")
'@ | & 'E:\working\anaconda3\envs\quant\python.exe' -
```

Expected: exactly two `get_price` calls occur and the final line is `RQData ETF smoke passed: day=1, minute=240, timestamps=09:31~15:00`.

- [ ] **Step 7: Audit the implementation against every acceptance criterion**

Run:

```powershell
git diff --check
git status --short
git diff -- my_utils/rqdata.py my_utils/rq_fun.py '任务/米筐ETF数据更新.py' tests/test_rq_etf_update.py
```

Verify from the diff and test evidence:

- No production directory was created by tests or smoke verification.
- No code path uses `adjust_type='pre'` for ETF storage.
- No minute normalizer subtracts time or aggregates Bars.
- No request loop iterates ETF codes.
- Day and minute cursors are independent.
- Fatal quota/auth/permission errors make one attempt and stop.
- Date bisection never catches validation or write errors.
- Existing stock RQ schemas, DDB access, and partition tests still pass.

- [ ] **Step 8: Leave the verified implementation unstaged and report its exact scope**

```powershell
git status --short -- my_utils/rqdata.py my_utils/rq_fun.py '任务/米筐ETF数据更新.py' tests/test_rq_etf_update.py
git diff --check -- my_utils/rqdata.py my_utils/rq_fun.py '任务/米筐ETF数据更新.py' tests/test_rq_etf_update.py
```

Expected: only the four planned implementation paths are reported for this feature review, diff check exits 0, and no implementation file is staged. If the repository status contains other pre-existing paths, leave them untouched and list them separately in the handoff.
