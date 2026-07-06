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

    empty_universe = rq_update.aggregate_right_aligned_15min(raw, rq_codes=[])
    assert empty_universe.is_empty()
    assert empty_universe.schema == rq_update.RQ_MIN_SCHEMA


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
    session = object()
    trading_days = [
        dt.date(2021, 1, 15),
        dt.date(2021, 1, 29),
        dt.date(2021, 2, 1),
        dt.date(2021, 2, 26),
        dt.date(2021, 3, 2),
    ]
    get_trading_days_calls = []
    month_calls = []

    def fake_get_trading_days(session_arg, start_date, end_date):
        get_trading_days_calls.append((session_arg, start_date, end_date))
        return trading_days

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

    monkeypatch.setattr(rq_update, "get_trading_days", fake_get_trading_days)
    monkeypatch.setattr(rq_update, "get_stock_universe", lambda session_arg: ["000001.XSHE"])
    monkeypatch.setattr(rq_update, "update_minute_range", fake_update_minute_range)

    written = rq_update.update_minute_all(
        session,
        dt.date(2021, 1, 15),
        dt.date(2021, 3, 2),
        "update",
    )

    assert written == len(trading_days)
    assert get_trading_days_calls == [(session, dt.date(2021, 1, 15), dt.date(2021, 3, 2))]
    assert month_calls == [
        (
            session,
            dt.date(2021, 1, 15),
            dt.date(2021, 1, 31),
            "update",
            {dt.date(2021, 1, 15), dt.date(2021, 1, 29)},
            ["000001.XSHE"],
        ),
        (
            session,
            dt.date(2021, 2, 1),
            dt.date(2021, 2, 28),
            "update",
            {dt.date(2021, 2, 1), dt.date(2021, 2, 26)},
            ["000001.XSHE"],
        ),
        (
            session,
            dt.date(2021, 3, 1),
            dt.date(2021, 3, 2),
            "update",
            {dt.date(2021, 3, 2)},
            ["000001.XSHE"],
        ),
    ]


def test_rq_fetch_minute_range_pushes_single_code_filter_into_ddb_query():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_single_code_sql_test")
    sample_date = dt.date(2021, 1, 4)

    class FakeSession:
        queries = []

        def run(self, script):
            self.queries.append(script)
            return pd.DataFrame(_full_a_share_minute_rows("000001.XSHE", sample_date))

    session = FakeSession()

    result = rq_update.fetch_minute_range(
        session,
        sample_date,
        sample_date,
        allowed_dates={sample_date},
        rq_codes=["000001.XSHE"],
    )

    assert result.height == 18
    assert "order_book_id = '000001.XSHE'" in session.queries[0]


def test_rq_fetch_minute_range_raises_on_ddb_query_failure():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_failure_test")
    sample_date = dt.date(2021, 1, 4)

    class FailingSession:
        def run(self, script):
            raise RuntimeError("Out of memory")

    with pytest.raises(RuntimeError, match="DDB 分钟线查询失败"):
        rq_update.fetch_minute_range(
            FailingSession(),
            sample_date,
            sample_date,
            allowed_dates={sample_date},
            rq_codes=["000001.XSHE"],
        )


def test_rq_parse_args_defaults_keep_day_update_and_support_minute_type():
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_cli_data_type_test")

    args = rq_update.parse_args([])
    assert args.data_type == "day"
    assert args.minute_quality_check_only is False

    min_args = rq_update.parse_args(["--data-type", "min", "--minute-quality-check-only"])
    assert min_args.data_type == "min"
    assert min_args.minute_quality_check_only is True


def test_rq_main_minute_quality_check_only_uses_minute_gate(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_gate_route_test")
    calls = []

    class FakeSession:
        closed = False

        def close(self):
            self.closed = True

    session = FakeSession()
    monkeypatch.setattr(rq_update, "create_ddb_session", lambda: session)
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
    assert calls == [("minute_gate", session)]
    assert session.closed is True


def test_rq_minute_quality_gate_compares_gm_right_sample(tmp_path, monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_quality_gate_test")
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
    ).cast(rq_update.RQ_MIN_SCHEMA)
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

    def fake_fetch_minute_range(session_arg, start_date, end_date, allowed_dates=None, rq_codes=None):
        calls.append((session_arg, start_date, end_date, set(allowed_dates or []), list(rq_codes or [])))
        return sample_df

    session = object()
    monkeypatch.setattr(rq_update, "DATA_ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(rq_update, "fetch_minute_range", fake_fetch_minute_range)

    assert rq_update.run_minute_quality_gate(session) is True
    assert calls == [
        (
            session,
            sample_date,
            sample_date,
            {sample_date},
            ["000001.XSHE"],
        )
    ]


def test_rq_main_min_data_type_routes_to_minute_update(monkeypatch):
    rq_update = load_script("任务/米筐数据更新.py", "rq_update_for_minute_update_route_test")
    calls = []

    class FakeSession:
        closed = False

        def close(self):
            self.closed = True

    session = FakeSession()
    monkeypatch.setattr(rq_update, "create_ddb_session", lambda: session)
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
    assert calls == [(session, dt.date(2021, 1, 4), dt.date(2021, 1, 4), "update")]
    assert session.closed is True


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
