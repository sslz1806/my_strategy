import datetime as dt
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import polars as pl
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str, module_name: str):
    script_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_rq_filter_to_stock_universe_removes_non_stock_codes():
    from my_utils import rq_fun

    raw = pl.DataFrame(
        {
            "order_book_id": [
                "000001.XSHE",
                "000001.XSHG",
                "H50066.XSHG",
                "302132.XSHE",
                "600000.XSHG",
            ],
            "trading_date": [dt.date(2021, 1, 4)] * 5,
            "close": [18.6, 3502.95, 137.73, 12.34, 9.69],
        }
    )
    filtered = rq_fun.filter_to_stock_universe(
        raw,
        ["000001.XSHE", "000001.XSHG", "H50066.XSHG", "302132.XSHE", "600000.XSHG"],
    )
    assert filtered["order_book_id"].to_list() == ["000001.XSHE", "600000.XSHG"]


def test_rq_fetch_day_range_keeps_only_allowed_trading_dates_and_a_shares():
    from my_utils.rqdata import DDBData

    trade_date = dt.date(2021, 1, 4)
    holiday = dt.date(2021, 1, 3)

    class FakeSession:
        def run(self, script):
            if "from loadTable('dfs://common_years_olap', 'day_kline')" in script:
                return pd.DataFrame(
                    {
                        "order_book_id": [
                            "000001.XSHE",
                            "600000.XSHG",
                            "000001.XSHG",
                            "H50066.XSHG",
                            "302132.XSHE",
                        ],
                        "trading_date": [trade_date, trade_date, trade_date, holiday, trade_date],
                        "open": [10.0, 8.0, 3500.0, 100.0, 12.0],
                        "close": [10.5, 8.2, 3510.0, 101.0, 12.2],
                        "high": [10.8, 8.4, 3520.0, 102.0, 12.4],
                        "low": [9.9, 7.9, 3490.0, 99.0, 11.8],
                        "volume": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
                        "amount": [10500.0, 16400.0, 10530000.0, 404000.0, 61000.0],
                        "pre_close": [10.0, 8.0, 3500.0, 100.0, 12.0],
                        "limit_up": [11.0, 8.8, 3850.0, 110.0, 13.2],
                        "limit_down": [9.0, 7.2, 3150.0, 90.0, 10.8],
                    }
                )
            if "from loadTable('dfs://stock_years_tsdb', 'is_st_stock')" in script:
                return pd.DataFrame()
            if "from loadTable('dfs://stock_years_tsdb', 'stock_shares')" in script:
                return pd.DataFrame()
            if "from loadTable('dfs://stock_years_tsdb', 'ex_factor')" in script:
                return pd.DataFrame()
            if "from loadTable('dfs://common_years_tsdb', 'instrument_base')" in script:
                return pd.DataFrame(
                    {
                        "order_book_id": ["000001.XSHE", "600000.XSHG"],
                        "name": ["平安银行", "浦发银行"],
                    }
                )
            raise AssertionError(f"unexpected query: {script}")

    result = DDBData(session=FakeSession()).fetch_day_range(
        holiday,
        trade_date,
        rq_codes=["000001.XSHE", "600000.XSHG", "000001.XSHG", "H50066.XSHG", "302132.XSHE"],
        allowed_dates={trade_date},
    )

    assert result["trading_date"].unique().to_list() == [trade_date]
    assert result["code"].to_list() == ["SZSE.000001", "SHSE.600000"]


def test_rq_remove_existing_partitions_in_range_clears_only_target_dates(tmp_path):
    from my_utils import rq_fun

    original_data_root = rq_fun.DATA_ROOT_DIR
    rq_fun.DATA_ROOT_DIR = str(tmp_path)
    target = tmp_path / "rq_stock_all_data"
    for date_text in ["2021-02-10", "2021-02-11", "2021-02-12"]:
        partition = target / f"trading_date={date_text}"
        partition.mkdir(parents=True)
        (partition / "part.parquet").write_text("partition marker", encoding="utf-8")

    try:
        rq_fun.remove_existing_partitions_in_range(
            "rq_stock_all_data",
            dt.date(2021, 2, 11),
            dt.date(2021, 2, 12),
        )
    finally:
        rq_fun.DATA_ROOT_DIR = original_data_root

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


def test_rq_aggregate_right_aligned_15min_adds_gm_style_snapshots():
    from my_utils import rq_fun

    raw = pl.DataFrame(_full_a_share_minute_rows())

    result = rq_fun.aggregate_right_aligned_15min(raw)

    assert result.schema == rq_fun.RQ_MIN_SCHEMA
    assert result.height == 18
    assert result["code"].unique().to_list() == ["SZSE.000001"]
    assert result["trading_date"].unique().to_list() == [dt.date(2021, 1, 4)]

    empty_universe = rq_fun.aggregate_right_aligned_15min(raw, rq_codes=[])
    assert empty_universe.is_empty()
    assert empty_universe.schema == rq_fun.RQ_MIN_SCHEMA
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
    assert morning_first_bar["high"] == 15.5
    assert morning_first_bar["low"] == 0.5
    assert morning_first_bar["close"] == 15.25
    assert morning_first_bar["volume"] == sum(float(i) for i in range(1, 16))
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
    from my_utils import rq_fun

    rows = _full_a_share_minute_rows("000001.XSHE", dt.date(2021, 1, 4))
    rows += _full_a_share_minute_rows("000001.XSHG", dt.date(2021, 1, 4))
    rows += _full_a_share_minute_rows("000001.XSHE", dt.date(2021, 1, 5))
    raw = pl.DataFrame(rows)

    result = rq_fun.aggregate_right_aligned_15min(
        raw,
        rq_codes=["000001.XSHE"],
        allowed_dates={dt.date(2021, 1, 4)},
    )

    assert result.height == 18
    assert result["code"].unique().to_list() == ["SZSE.000001"]
    assert result["trading_date"].unique().to_list() == [dt.date(2021, 1, 4)]


def _full_market_right_aligned_sample(stock_count=100):
    from my_utils import rq_fun

    one_stock = rq_fun.aggregate_right_aligned_15min(
        pl.DataFrame(_full_a_share_minute_rows())
    )
    return pl.concat(
        [
            one_stock.with_columns(pl.lit(f"SZSE.{index:06d}").alias("code"))
            for index in range(stock_count)
        ]
    )


def test_rq_minute_market_validation_accepts_full_right_aligned_schedule():
    from my_utils import rq_fun

    minute_data = _full_market_right_aligned_sample()

    rq_fun.validate_minute_market_coverage(
        minute_data,
        rq_codes=[f"{index:06d}.XSHE" for index in range(100)],
    )


def test_rq_minute_market_validation_rejects_corrupted_snapshot():
    from my_utils import rq_fun

    snapshot_time = dt.datetime(2021, 1, 4, 9, 30)
    minute_data = _full_market_right_aligned_sample().with_columns(
        pl.when(
            (pl.col("code") == "SZSE.000000")
            & (pl.col("datetime") == snapshot_time)
        )
        .then(pl.col("high") + 1.0)
        .otherwise(pl.col("high"))
        .alias("high")
    )

    with pytest.raises(RuntimeError, match="minute snapshot invalid"):
        rq_fun.validate_minute_market_coverage(
            minute_data,
            rq_codes=[f"{index:06d}.XSHE" for index in range(100)],
        )


def test_rq_day_market_validation_rejects_small_fraction_of_stock_universe():
    from my_utils import rq_fun

    trade_date = dt.date(2021, 1, 4)
    day_data = rq_fun.align_schema(
        pl.DataFrame(
            {
                "code": [f"SZSE.{index:06d}" for index in range(100)],
                "trading_date": [trade_date] * 100,
            }
        ),
        rq_fun.RQ_DAY_SCHEMA,
    )

    with pytest.raises(RuntimeError, match="day coverage too low"):
        rq_fun.validate_day_market_coverage(
            day_data,
            rq_codes=[f"{index:06d}.XSHE" for index in range(5_000)],
        )


def test_rq_minute_market_validation_rejects_small_fraction_of_stock_universe():
    from my_utils import rq_fun

    minute_data = _full_market_right_aligned_sample()

    with pytest.raises(RuntimeError, match="minute coverage too low"):
        rq_fun.validate_minute_market_coverage(
            minute_data,
            rq_codes=[f"{index:06d}.XSHE" for index in range(5_000)],
        )

def test_rq_build_month_ranges_splits_by_natural_month():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_month_ranges_test")

    ranges = rq_update.build_month_ranges(dt.date(2021, 1, 15), dt.date(2021, 3, 2))

    assert ranges == [
        (dt.date(2021, 1, 15), dt.date(2021, 1, 31)),
        (dt.date(2021, 2, 1), dt.date(2021, 2, 28)),
        (dt.date(2021, 3, 1), dt.date(2021, 3, 2)),
    ]


def test_rq_update_minute_all_reuses_trading_days_for_month_batches(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_allowed_dates_test")
    trading_days = [
        dt.date(2021, 1, 15),
        dt.date(2021, 1, 29),
        dt.date(2021, 2, 1),
        dt.date(2021, 2, 26),
        dt.date(2021, 3, 2),
    ]
    get_trading_days_calls = []
    month_calls = []

    class FakeSource:
        def get_trading_days(self, start_date, end_date):
            get_trading_days_calls.append((self, start_date, end_date))
            return trading_days

        def get_stock_universe(self):
            return ["000001.XSHE"]

    source = FakeSource()

    def fake_update_minute_range(
        session_arg,
        start_date,
        end_date,
        mode,
        allowed_dates=None,
        rq_codes=None,
    ):
        month_calls.append(
            (
                session_arg,
                start_date,
                end_date,
                mode,
                set(allowed_dates or []),
                list(rq_codes or []),
            )
        )
        return len(allowed_dates or [])

    monkeypatch.setattr(rq_update, "update_minute_range", fake_update_minute_range)

    written = rq_update.update_minute_all(
        source,
        dt.date(2021, 1, 15),
        dt.date(2021, 3, 2),
        "update",
    )

    assert written == len(trading_days)
    assert get_trading_days_calls == [(source, dt.date(2021, 1, 15), dt.date(2021, 3, 2))]
    assert month_calls == [
        (
            source,
            dt.date(2021, 1, 15),
            dt.date(2021, 1, 31),
            "update",
            {dt.date(2021, 1, 15), dt.date(2021, 1, 29)},
            ["000001.XSHE"],
        ),
        (
            source,
            dt.date(2021, 2, 1),
            dt.date(2021, 2, 28),
            "update",
            {dt.date(2021, 2, 1), dt.date(2021, 2, 26)},
            ["000001.XSHE"],
        ),
        (
            source,
            dt.date(2021, 3, 1),
            dt.date(2021, 3, 2),
            "update",
            {dt.date(2021, 3, 2)},
            ["000001.XSHE"],
        ),
    ]


def test_rq_update_all_passes_trading_day_filter_to_day_batches(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_day_allowed_dates_test")
    calls = []
    trading_days = [
        dt.date(2021, 1, 4),
        dt.date(2021, 1, 5),
        dt.date(2021, 2, 1),
    ]

    def fake_update_day_range(
        session_arg,
        start_date,
        end_date,
        mode,
        rq_codes=None,
        allowed_dates=None,
    ):
        calls.append(
            (
                session_arg,
                start_date,
                end_date,
                mode,
                list(rq_codes or []),
                sorted(allowed_dates or []),
            )
        )
        return len(allowed_dates or [])

    class FakeSource:
        def get_trading_days(self, start_date, end_date):
            return trading_days

        def get_stock_universe(self):
            return ["000001.XSHE"]

    source = FakeSource()
    monkeypatch.setattr(rq_update, "update_day_range", fake_update_day_range)

    written = rq_update.update_all(
        source,
        dt.date(2021, 1, 1),
        dt.date(2021, 2, 28),
        "update",
        batch_mode="days",
        batch_size=2,
    )

    assert written == 3
    assert calls == [
        (
            source,
            dt.date(2021, 1, 4),
            dt.date(2021, 1, 5),
            "update",
            ["000001.XSHE"],
            [dt.date(2021, 1, 4), dt.date(2021, 1, 5)],
        ),
        (
            source,
            dt.date(2021, 2, 1),
            dt.date(2021, 2, 1),
            "update",
            ["000001.XSHE"],
            [dt.date(2021, 2, 1)],
        ),
    ]


def test_rq_update_day_range_rejects_missing_trading_day_before_cleanup(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_day_missing_guard_test")
    present_date = dt.date(2026, 2, 24)
    missing_date = dt.date(2026, 2, 25)
    day_data = pl.DataFrame(
        {
            "code": [f"SZSE.{index:06d}" for index in range(150)],
            "trading_date": [present_date] * 150,
        }
    )

    class FakeSource:
        def fetch_day_range(self, *args, **kwargs):
            return day_data

    monkeypatch.setattr(
        rq_update,
        "write_rq_day_and_adj_partitioned",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("missing day data must not be written")
        ),
    )

    with pytest.raises(RuntimeError, match="day data missing trading days"):
        rq_update.update_day_range(
            FakeSource(),
            present_date,
            missing_date,
            "update",
            rq_codes=[f"{index:06d}.XSHE" for index in range(200)],
            allowed_dates={present_date, missing_date},
        )


def test_rq_update_day_range_rejects_low_market_coverage_before_cleanup(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_day_coverage_guard_test")
    trade_date = dt.date(2026, 2, 24)
    day_data = pl.DataFrame(
        {"code": ["SZSE.000001"], "trading_date": [trade_date]}
    )

    class FakeSource:
        def fetch_day_range(self, *args, **kwargs):
            return day_data

    monkeypatch.setattr(
        rq_update,
        "write_rq_day_and_adj_partitioned",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("low-coverage day data must not be written")
        ),
    )

    with pytest.raises(RuntimeError, match="day coverage too low"):
        rq_update.update_day_range(
            FakeSource(),
            trade_date,
            trade_date,
            "update",
            rq_codes=[f"{index:06d}.XSHE" for index in range(200)],
            allowed_dates={trade_date},
        )



def test_rq_update_minute_range_rejects_trading_day_with_too_few_codes(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_coverage_guard_test")
    trade_date = dt.date(2026, 2, 24)
    bar_times = [
        dt.datetime.combine(trade_date, dt.time(hour, minute))
        for hour, minute in [
            (9, 30),
            (9, 45),
            (10, 0),
            (10, 15),
            (10, 30),
            (10, 45),
            (11, 0),
            (11, 15),
            (11, 30),
            (13, 0),
            (13, 15),
            (13, 30),
            (13, 45),
            (14, 0),
            (14, 15),
            (14, 30),
            (14, 45),
            (15, 0),
        ]
    ]
    one_stock_day = pl.DataFrame(
        {
            "code": ["SZSE.000001"] * len(bar_times),
            "datetime": bar_times,
            "open": [1.0] * len(bar_times),
            "high": [1.0] * len(bar_times),
            "low": [1.0] * len(bar_times),
            "close": [1.0] * len(bar_times),
            "volume": [1.0] * len(bar_times),
            "trading_date": [trade_date] * len(bar_times),
        },
        schema=rq_update.RQ_MIN_SCHEMA,
    )

    class FakeSource:
        def fetch_minute_range(self, *args, **kwargs):
            return one_stock_day

    def fail_if_written(*args, **kwargs):
        raise AssertionError("bad minute data must not be written")

    monkeypatch.setattr(rq_update, "write_partitioned", fail_if_written)

    with pytest.raises(RuntimeError, match="minute coverage too low"):
        rq_update.update_minute_range(
            FakeSource(),
            trade_date,
            trade_date,
            "update",
            allowed_dates={trade_date},
            rq_codes=[f"{idx:06d}.XSHE" for idx in range(200)],
        )

def test_rq_update_minute_range_rejects_missing_historical_trading_day(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_missing_day_test")
    present_date = dt.date(2026, 2, 24)
    missing_date = dt.date(2026, 2, 25)
    bar_times = [
        dt.datetime.combine(present_date, dt.time(hour, minute))
        for hour, minute in [
            (9, 30),
            (9, 45),
            (10, 0),
            (10, 15),
            (10, 30),
            (10, 45),
            (11, 0),
            (11, 15),
            (11, 30),
            (13, 0),
            (13, 15),
            (13, 30),
            (13, 45),
            (14, 0),
            (14, 15),
            (14, 30),
            (14, 45),
            (15, 0),
        ]
    ]
    rows = []
    for idx in range(150):
        code = f"SZSE.{idx:06d}"
        for bar_time in bar_times:
            rows.append(
                {
                    "code": code,
                    "datetime": bar_time,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1.0,
                    "trading_date": present_date,
                }
            )
    one_day_only = pl.DataFrame(rows, schema=rq_update.RQ_MIN_SCHEMA)

    class FakeSource:
        def fetch_minute_range(self, *args, **kwargs):
            return one_day_only

    def fail_if_written(*args, **kwargs):
        raise AssertionError("missing historical minute data must not be written")

    monkeypatch.setattr(rq_update, "write_partitioned", fail_if_written)

    with pytest.raises(RuntimeError, match="minute data missing trading days"):
        rq_update.update_minute_range(
            FakeSource(),
            present_date,
            missing_date,
            "update",
            allowed_dates={present_date, missing_date},
            rq_codes=[f"{idx:06d}.XSHE" for idx in range(200)],
        )


def test_rq_update_minute_range_rejects_incomplete_bars_before_cleanup(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_bar_guard_test")
    trade_date = dt.date(2026, 2, 24)
    rows = [
        {
            "code": f"SZSE.{index:06d}",
            "datetime": dt.datetime.combine(trade_date, dt.time(9, 30)),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
            "trading_date": trade_date,
        }
        for index in range(150)
    ]
    incomplete_data = pl.DataFrame(rows, schema=rq_update.RQ_MIN_SCHEMA)

    class FakeSource:
        def fetch_minute_range(self, *args, **kwargs):
            return incomplete_data

    monkeypatch.setattr(
        rq_update,
        "write_partitioned",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("incomplete minute bars must not be written")
        ),
    )

    with pytest.raises(RuntimeError, match="minute bar count incomplete"):
        rq_update.update_minute_range(
            FakeSource(),
            trade_date,
            trade_date,
            "update",
            allowed_dates={trade_date},
            rq_codes=[f"{index:06d}.XSHE" for index in range(200)],
        )



def test_rq_update_minute_all_falls_back_to_daily_batches_after_month_failure(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_daily_fallback_test")
    calls = []
    trading_days = [dt.date(2026, 2, 24), dt.date(2026, 2, 25)]

    def fake_update_minute_range(
        session_arg,
        start_date,
        end_date,
        mode,
        allowed_dates=None,
        rq_codes=None,
    ):
        calls.append((start_date, end_date, sorted(allowed_dates or []), list(rq_codes or [])))
        if start_date != end_date:
            raise RuntimeError("DDB 分钟线查询失败: Out of memory")
        return 10

    class FakeSource:
        def get_trading_days(self, start_date, end_date):
            return trading_days

        def get_stock_universe(self):
            return ["000001.XSHE"]

    source = FakeSource()
    monkeypatch.setattr(rq_update, "update_minute_range", fake_update_minute_range)

    written = rq_update.update_minute_all(
        source,
        dt.date(2026, 2, 1),
        dt.date(2026, 2, 28),
        "update",
    )

    assert written == 20
    assert calls == [
        (
            dt.date(2026, 2, 1),
            dt.date(2026, 2, 28),
            [dt.date(2026, 2, 24), dt.date(2026, 2, 25)],
            ["000001.XSHE"],
        ),
        (dt.date(2026, 2, 24), dt.date(2026, 2, 24), [dt.date(2026, 2, 24)], ["000001.XSHE"]),
        (dt.date(2026, 2, 25), dt.date(2026, 2, 25), [dt.date(2026, 2, 25)], ["000001.XSHE"]),
    ]



def test_rq_fetch_minute_range_pushes_single_code_filter_into_ddb_query():
    from my_utils.rqdata import DDBData

    sample_date = dt.date(2021, 1, 4)

    class FakeSession:
        queries = []

        def run(self, script):
            self.queries.append(script)
            return pd.DataFrame(_full_a_share_minute_rows("000001.XSHE", sample_date))

    session = FakeSession()

    result = DDBData(session=session).fetch_minute_range(
        sample_date,
        sample_date,
        allowed_dates={sample_date},
        rq_codes=["000001.XSHE"],
    )

    assert result.height == 18
    assert "order_book_id = '000001.XSHE'" in session.queries[0]


def test_rq_fetch_minute_range_raises_on_ddb_query_failure():
    from my_utils.rqdata import DDBData

    sample_date = dt.date(2021, 1, 4)

    class FailingSession:
        def run(self, script):
            raise RuntimeError("Out of memory")

    with pytest.raises(RuntimeError, match="DDB 分钟线查询失败"):
        DDBData(session=FailingSession()).fetch_minute_range(
            sample_date,
            sample_date,
            allowed_dates={sample_date},
            rq_codes=["000001.XSHE"],
        )


def test_rq_parse_args_defaults_keep_day_update_and_support_minute_type():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_cli_data_type_test")

    args = rq_update.parse_args([])
    assert args.data_type == "all"
    assert args.minute_quality_check_only is False

    min_args = rq_update.parse_args(["--data-type", "min", "--minute-quality-check-only"])
    assert min_args.data_type == "min"
    assert min_args.minute_quality_check_only is True


def test_rq_main_minute_quality_check_only_uses_minute_gate(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_gate_route_test")
    calls = []

    class FakeSource:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.closed = True
            return False

    source = FakeSource()
    monkeypatch.setattr(rq_update, "DDBData", lambda: source)
    monkeypatch.setattr(
        rq_update,
        "run_minute_quality_gate",
        lambda session_arg: calls.append(("minute_gate", session_arg)) or True,
        raising=False,
    )
    monkeypatch.setattr(
        rq_update,
        "run_ddb_quality_gate",
        lambda session_arg: (_ for _ in ()).throw(AssertionError("日线质量门不应被调用")),
    )
    monkeypatch.setattr(
        rq_update,
        "update_all",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("日线更新不应被调用")),
    )
    monkeypatch.setattr(
        rq_update,
        "update_minute_all",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("质量门模式不应写分钟数据")),
    )

    result = rq_update.main(["--mode", "update", "--data-type", "min", "--minute-quality-check-only"])

    assert result == 0
    assert calls == [("minute_gate", source)]
    assert source.closed is True


def test_rq_minute_quality_gate_compares_gm_right_sample(tmp_path, monkeypatch):
    from my_utils import rq_fun

    sample_date = dt.date(2021, 1, 4)
    sample_times = [
        dt.datetime(2021, 1, 4, 9, 30),
        dt.datetime(2021, 1, 4, 9, 45),
        dt.datetime(2021, 1, 4, 10, 0),
        dt.datetime(2021, 1, 4, 10, 15),
        dt.datetime(2021, 1, 4, 10, 30),
        dt.datetime(2021, 1, 4, 10, 45),
        dt.datetime(2021, 1, 4, 11, 0),
        dt.datetime(2021, 1, 4, 11, 15),
        dt.datetime(2021, 1, 4, 11, 30),
        dt.datetime(2021, 1, 4, 13, 0),
        dt.datetime(2021, 1, 4, 13, 15),
        dt.datetime(2021, 1, 4, 13, 30),
        dt.datetime(2021, 1, 4, 13, 45),
        dt.datetime(2021, 1, 4, 14, 0),
        dt.datetime(2021, 1, 4, 14, 15),
        dt.datetime(2021, 1, 4, 14, 30),
        dt.datetime(2021, 1, 4, 14, 45),
        dt.datetime(2021, 1, 4, 15, 0),
    ]
    values = [float(i) for i in range(1, len(sample_times) + 1)]
    sample_df = pl.DataFrame(
        {
            "code": ["SZSE.000001"] * len(sample_times),
            "datetime": sample_times,
            "open": values,
            "high": [value + 0.1 for value in values],
            "low": [value - 0.1 for value in values],
            "close": [value + 0.05 for value in values],
            "volume": [value * 100 for value in values],
            "trading_date": [sample_date] * len(sample_times),
        }
    ).cast(rq_fun.RQ_MIN_SCHEMA)
    sample_df = sample_df.with_columns(
        [
            pl.when(pl.col("datetime") == dt.datetime(2021, 1, 4, 9, 30))
            .then(pl.lit(2.0))
            .when(pl.col("datetime") == dt.datetime(2021, 1, 4, 13, 0))
            .then(pl.lit(11.0))
            .otherwise(pl.col(col))
            .alias(col)
            for col in ["open", "high", "low", "close"]
        ]
        + [
            pl.when(pl.col("datetime") == dt.datetime(2021, 1, 4, 9, 30))
            .then(pl.lit(200.0))
            .when(pl.col("datetime") == dt.datetime(2021, 1, 4, 13, 0))
            .then(pl.lit(1100.0))
            .otherwise(pl.col("volume"))
            .alias("volume")
        ]
    )

    gm_partition = tmp_path / "15min_stock_data_right_dir" / "trading_date=2021-01-04"
    gm_partition.mkdir(parents=True)
    # GM 和 RQ/DDB 是不同数据源，真实样本里 OHLCV 可能并不逐值相等；
    # 质量门只要求右对齐时间戳口径一致，数值差异应记录为提示而不是失败。
    gm_sample_df = sample_df.with_columns(
        [
            (pl.col("open") + 0.5).alias("open"),
            (pl.col("high") + 0.5).alias("high"),
            (pl.col("low") + 0.5).alias("low"),
            (pl.col("close") + 0.5).alias("close"),
            (pl.col("volume") + 100.0).alias("volume"),
        ]
    )
    gm_sample_df.write_parquet(gm_partition / "part.parquet")

    calls = []

    class FakeSource:
        def fetch_minute_range(
            self,
            start_date,
            end_date,
            allowed_dates=None,
            rq_codes=None,
        ):
            calls.append(
                (self, start_date, end_date, set(allowed_dates or []), list(rq_codes or []))
            )
            return sample_df

    source = FakeSource()
    monkeypatch.setattr(rq_fun, "DATA_ROOT_DIR", str(tmp_path))

    assert rq_fun.run_minute_quality_gate(source) is True
    assert calls == [
        (
            source,
            sample_date,
            sample_date,
            {sample_date},
            ["000001.XSHE"],
        )
    ]


def test_rq_main_min_data_type_routes_to_minute_update(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_update_route_test")
    calls = []

    class FakeSource:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.closed = True
            return False

    source = FakeSource()
    monkeypatch.setattr(rq_update, "DDBData", lambda: source)
    monkeypatch.setattr(
        rq_update,
        "run_ddb_quality_gate",
        lambda session_arg: (_ for _ in ()).throw(AssertionError("日线质量门不应被调用")),
    )
    monkeypatch.setattr(
        rq_update,
        "run_minute_quality_gate",
        lambda session_arg: (_ for _ in ()).throw(AssertionError("已跳过质量门，不应被调用")),
        raising=False,
    )
    monkeypatch.setattr(
        rq_update,
        "update_all",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("data-type=min 不应更新日线")),
    )

    def fake_update_minute_all(session_arg, start_date, end_date, mode):
        calls.append((session_arg, start_date, end_date, mode))
        return 18

    monkeypatch.setattr(rq_update, "update_minute_all", fake_update_minute_all)

    result = rq_update.main(
        [
            "--start-date",
            "2021-01-04",
            "--end-date",
            "2021-01-04",
            "--mode",
            "update",
            "--data-type",
            "min",
            "--skip-quality-check",
        ]
    )

    assert result == 0
    assert calls == [(source, dt.date(2021, 1, 4), dt.date(2021, 1, 4), "update")]
    assert source.closed is True


def test_rq_main_all_insert_uses_independent_day_and_minute_cursors(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_all_insert_cursor_test")
    calls = []

    class FakeSource:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.closed = True
            return False

    source = FakeSource()
    monkeypatch.setattr(rq_update, "DDBData", lambda: source)
    monkeypatch.setattr(
        rq_update,
        "infer_start_date",
        lambda default_start, save_dir, mode, end_date=None: {
            rq_update.RQ_DAY_DIR: None,
            rq_update.RQ_MIN_DIR: dt.date(2026, 7, 7),
        }[save_dir],
    )
    monkeypatch.setattr(
        rq_update,
        "update_all",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("day data is already latest and should be skipped")),
    )

    def fake_update_minute_all(session_arg, start_date, end_date, mode):
        calls.append((session_arg, start_date, end_date, mode))
        return 18

    monkeypatch.setattr(rq_update, "update_minute_all", fake_update_minute_all)

    result = rq_update.main(["--end-date", "2026-07-07"])

    assert result == 0
    assert calls == [(source, dt.date(2026, 7, 7), dt.date(2026, 7, 7), "insert")]
    assert source.closed is True

def test_gm_parse_args_defaults_preserve_daily_run():
    source = (PROJECT_ROOT / "任务/数据更新v2.py").read_text(encoding="utf-8")
    assert "def parse_args" in source
    assert 'if __name__ == "__main__"' in source
    gm_update = load_script("任务/数据更新v2.py", "gm_update_for_args_test")
    args = gm_update.parse_args([])
    assert args.mode == "insert"
    assert args.skip_day is False
    assert args.skip_min is False
    assert args.allow_old_min is False
    assert args.min_align == "both"


def test_gm_minute_history_guard_blocks_old_dates():
    gm_update = load_script("任务/数据更新v2.py", "gm_update_for_min_guard_test")

    ok, floor = gm_update.check_gm_minute_history_window(
        dt.date(2018, 1, 2),
        today=dt.date(2026, 7, 6),
    )
    assert ok is False
    assert floor == dt.date(2026, 1, 7)

    ok, floor = gm_update.check_gm_minute_history_window(
        dt.date(2026, 1, 7),
        today=dt.date(2026, 7, 6),
    )
    assert ok is True
    assert floor == dt.date(2026, 1, 7)


def test_gm_cli_handles_gbk_stdout_without_unicodeencodeerror():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "gbk"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "任务/数据更新v2.py"),
            "--start-date",
            "2021-01-04",
            "--end-date",
            "2021-01-04",
            "--skip-day",
            "--skip-min",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    combined_output = result.stdout + result.stderr
    assert result.returncode == 0
    assert "UnicodeEncodeError" not in combined_output
