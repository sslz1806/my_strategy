import datetime as dt
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd
import polars as pl

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MY_UTILS_DIR = os.path.join(ROOT_DIR, "my_utils")
sys.path.insert(0, ROOT_DIR)


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RqUpdateDataTests(unittest.TestCase):
    def test_convert_code_format_supports_rq_exchange_suffix(self):
        from my_utils.mapping import convert_code_format

        self.assertEqual(convert_code_format("000001.XSHE", format="gm"), "SZSE.000001")
        self.assertEqual(convert_code_format("600000.XSHG", format="gm"), "SHSE.600000")
        self.assertEqual(convert_code_format("000001.xshe", format="gm"), "SZSE.000001")
        self.assertEqual(convert_code_format(["000001.XSHE", "600000.XSHG"], format="suffix"), ["000001.SZ", "600000.SH"])

    def test_convert_code_format_supports_polars_expression(self):
        from my_utils.mapping import convert_code_format

        source = pl.DataFrame(
            {
                "raw_code": [
                    "000001.XSHE",
                    "600000.XSHG",
                    "SZSE.000001",
                    "600000.SH",
                    "bj430047",
                    "UNKNOWN",
                ]
            }
        )

        result = source.with_columns(
            convert_code_format(pl.col("raw_code"), format="gm").alias("code")
        )

        self.assertEqual(
            result["code"].to_list(),
            ["SZSE.000001", "SHSE.600000", "SZSE.000001", "SHSE.600000", None, None],
        )

        alternate_formats = source.head(4).select(
            [
                convert_code_format(pl.col("raw_code"), format="suffix").alias("suffix"),
                convert_code_format(pl.col("raw_code"), format="pure").alias("pure"),
            ]
        )
        self.assertEqual(
            alternate_formats["suffix"].to_list(),
            ["000001.SZ", "600000.SH", "000001.SZ", "600000.SH"],
        )
        self.assertEqual(
            alternate_formats["pure"].to_list(),
            ["000001", "600000", "000001", "600000"],
        )

    def test_convert_code_format_polars_integer_matches_scalar(self):
        from my_utils.mapping import convert_code_format

        source = pl.DataFrame({"raw_code": [1, 600000]})
        result = source.select(
            [
                convert_code_format(pl.col("raw_code"), format="gm").alias("gm"),
                convert_code_format(pl.col("raw_code"), format="suffix").alias("suffix"),
                convert_code_format(pl.col("raw_code"), format="pure").alias("pure"),
            ]
        )

        self.assertEqual(
            result.to_dict(as_series=False),
            {
                "gm": ["SZSE.000001", "SHSE.600000"],
                "suffix": ["000001.SZ", "600000.SH"],
                "pure": ["000001", "600000"],
            },
        )

    def test_ddb_data_reuses_injected_session_without_taking_ownership(self):
        from my_utils.rqdata import DDBData

        class FakeSession:
            def __init__(self):
                self.closed = False
                self.queries = []

            def run(self, script, **kwargs):
                self.queries.append(script)
                return pd.DataFrame({"order_book_id": ["000001.XSHE"]})

            def close(self):
                self.closed = True

        session = FakeSession()
        with DDBData(session=session) as source:
            codes = source.get_stock_universe()

        self.assertEqual(codes, ["000001.XSHE"])
        self.assertEqual(len(session.queries), 1)
        self.assertFalse(session.closed)

    def test_ddb_data_owned_session_connects_once_and_closes(self):
        from my_utils.rqdata import DDBData

        class FakeSession:
            def __init__(self):
                self.connect_calls = []
                self.closed = False

            def connect(self, host, port, user, password):
                self.connect_calls.append((host, port, user, password))

            def run(self, script, **kwargs):
                return pd.DataFrame({"trade_date": pd.to_datetime(["2021-01-04"])})

            def close(self):
                self.closed = True

        session = FakeSession()
        with patch("my_utils.rqdata.ddb.session", return_value=session):
            with DDBData() as source:
                dates = source.get_trading_days(
                    dt.date(2021, 1, 1),
                    dt.date(2021, 1, 5),
                )

        self.assertEqual(dates, [dt.date(2021, 1, 4)])
        self.assertEqual(len(session.connect_calls), 1)
        self.assertTrue(session.closed)

    def test_rq_partition_range_cleanup_and_end_bounded_insert_cursor(self):
        from my_utils import rq_fun

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(rq_fun, "DATA_ROOT_DIR", temp_dir):
                target = os.path.join(temp_dir, "rq_stock_all_data")
                for date_text in ["2021-02-10", "2021-02-11", "2021-02-12"]:
                    partition = os.path.join(target, f"trading_date={date_text}")
                    os.makedirs(partition, exist_ok=True)
                    with open(os.path.join(partition, "part.parquet"), "w", encoding="utf-8") as file:
                        file.write("marker")

                removed = rq_fun.remove_existing_partitions_in_range(
                    "rq_stock_all_data",
                    dt.date(2021, 2, 11),
                    dt.date(2021, 2, 12),
                )
                next_date = rq_fun.infer_start_date(
                    dt.date(2021, 1, 1),
                    "rq_stock_all_data",
                    "insert",
                    end_date=dt.date(2021, 2, 10),
                )

                self.assertEqual(removed, 2)
                self.assertTrue(os.path.exists(os.path.join(target, "trading_date=2021-02-10")))
                self.assertFalse(os.path.exists(os.path.join(target, "trading_date=2021-02-11")))
                self.assertFalse(os.path.exists(os.path.join(target, "trading_date=2021-02-12")))
                self.assertIsNone(next_date)

    def test_write_partitioned_keeps_old_partition_when_staging_fails(self):
        from my_utils import rq_fun

        trade_date = dt.date(2021, 1, 4)
        old_data = pl.DataFrame(
            {
                "code": ["SZSE.000001"],
                "datetime": [dt.datetime(2021, 1, 4, 9, 30)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [100.0],
                "trading_date": [trade_date],
            },
            schema=rq_fun.RQ_MIN_SCHEMA,
        )
        new_data = old_data.with_columns(pl.lit(2.0).alias("close"))

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(rq_fun, "DATA_ROOT_DIR", temp_dir):
                partition = os.path.join(
                    temp_dir,
                    "rq_15min_stock_data_dir",
                    "trading_date=2021-01-04",
                )
                os.makedirs(partition, exist_ok=True)
                old_file = os.path.join(partition, "part.parquet")
                old_data.write_parquet(old_file)

                with patch.object(
                    pl.DataFrame,
                    "write_parquet",
                    side_effect=OSError("simulated disk full"),
                ):
                    with self.assertRaisesRegex(OSError, "simulated disk full"):
                        rq_fun.write_partitioned(
                            new_data,
                            "rq_15min_stock_data_dir",
                            rq_fun.RQ_MIN_SCHEMA,
                            "update",
                        )

                self.assertTrue(os.path.exists(old_file))
                self.assertEqual(pl.read_parquet(old_file)["close"].to_list(), [1.0])

    def test_write_partitioned_does_not_use_acl_protected_tempfile_directory(self):
        from my_utils import rq_fun

        trade_date = dt.date(2026, 8, 5)
        data = pl.DataFrame(
            {
                "code": ["SZSE.000001"],
                "datetime": [dt.datetime(2026, 8, 5, 9, 30)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [100.0],
                "trading_date": [trade_date],
            },
            schema=rq_fun.RQ_MIN_SCHEMA,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(rq_fun, "DATA_ROOT_DIR", temp_dir):
                with patch.object(
                    tempfile,
                    "mkdtemp",
                    side_effect=AssertionError(
                        "tempfile.mkdtemp creates protected ACLs on Windows"
                    ),
                ):
                    written = rq_fun.write_partitioned(
                        data,
                        "rq_15min_stock_data_dir",
                        rq_fun.RQ_MIN_SCHEMA,
                        "update",
                    )

                partition = os.path.join(
                    temp_dir,
                    "rq_15min_stock_data_dir",
                    "trading_date=2026-08-05",
                    "*.parquet",
                )
                self.assertEqual(written, 1)
                self.assertEqual(pl.read_parquet(partition)["close"].to_list(), [1.0])

    def test_day_and_adj_write_rolls_back_both_partitions_on_commit_failure(self):
        from my_utils import rq_fun

        trade_date = dt.date(2021, 1, 4)
        old_day = rq_fun.align_schema(
            pl.DataFrame(
                {
                    "code": ["SZSE.000001"],
                    "trading_date": [trade_date],
                    "close": [1.0],
                    "adj_factor": [1.0],
                }
            ),
            rq_fun.RQ_DAY_SCHEMA,
        )
        new_day = old_day.with_columns(
            [pl.lit(2.0).alias("close"), pl.lit(2.0).alias("adj_factor")]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(rq_fun, "DATA_ROOT_DIR", temp_dir):
                day_partition = os.path.join(
                    temp_dir, "rq_stock_all_data", "trading_date=2021-01-04"
                )
                adj_partition = os.path.join(
                    temp_dir, "rq_adj", "trading_date=2021-01-04"
                )
                os.makedirs(day_partition, exist_ok=True)
                os.makedirs(adj_partition, exist_ok=True)
                old_day.write_parquet(os.path.join(day_partition, "part.parquet"))
                old_day.select(["code", "trading_date", "adj_factor"]).write_parquet(
                    os.path.join(adj_partition, "part.parquet")
                )

                original_replace = rq_fun.os.replace
                replace_calls = []

                def fail_on_second_new_partition(source, target):
                    replace_calls.append((source, target))
                    if len(replace_calls) == 4:
                        raise OSError("simulated second-partition commit failure")
                    return original_replace(source, target)

                with patch.object(rq_fun.os, "replace", side_effect=fail_on_second_new_partition):
                    with self.assertRaisesRegex(OSError, "second-partition commit failure"):
                        rq_fun.write_rq_day_and_adj_partitioned(
                            new_day,
                            "rq_stock_all_data",
                            "rq_adj",
                            "update",
                        )

                restored_day = pl.read_parquet(
                    os.path.join(day_partition, "part.parquet")
                )
                restored_adj = pl.read_parquet(
                    os.path.join(adj_partition, "part.parquet")
                )
                self.assertEqual(restored_day["close"].to_list(), [1.0])
                self.assertEqual(restored_adj["adj_factor"].to_list(), [1.0])

    def test_backup_cleanup_failure_does_not_turn_committed_write_into_failure(self):
        from my_utils import rq_fun

        trade_date = dt.date(2021, 1, 4)
        old_data = pl.DataFrame(
            {
                "code": ["SZSE.000001"],
                "datetime": [dt.datetime(2021, 1, 4, 9, 30)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [100.0],
                "trading_date": [trade_date],
            },
            schema=rq_fun.RQ_MIN_SCHEMA,
        )
        new_data = old_data.with_columns(pl.lit(2.0).alias("close"))

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(rq_fun, "DATA_ROOT_DIR", temp_dir):
                partition = os.path.join(
                    temp_dir,
                    "rq_15min_stock_data_dir",
                    "trading_date=2021-01-04",
                )
                os.makedirs(partition, exist_ok=True)
                old_data.write_parquet(os.path.join(partition, "part.parquet"))

                with patch.object(
                    rq_fun,
                    "_remove_path",
                    side_effect=PermissionError("simulated backup lock"),
                ):
                    written = rq_fun.write_partitioned(
                        new_data,
                        "rq_15min_stock_data_dir",
                        rq_fun.RQ_MIN_SCHEMA,
                        "update",
                    )

                current = pl.read_parquet(
                    os.path.join(partition, "*.parquet")
                )
                self.assertEqual(written, 1)
                self.assertEqual(current["close"].to_list(), [2.0])

    def test_expand_rq_multiindex_adds_code_and_date_columns(self):
        from my_utils import rq_fun as module
        raw = pd.DataFrame(
            {"close": [10.0, 20.0]},
            index=pd.MultiIndex.from_arrays(
                [
                    ["000001.XSHE", "600000.XSHG"],
                    pd.to_datetime(["2026-03-23", "2026-03-24"]),
                ],
                names=["order_book_id", "date"],
            ),
        )

        result = module.expand_rq_multiindex(raw, timestamp_col="trading_date", shift_minutes=0)

        self.assertEqual(result["code"].tolist(), ["SZSE.000001", "SHSE.600000"])
        self.assertEqual(result["trading_date"].tolist(), [dt.date(2026, 3, 23), dt.date(2026, 3, 24)])
        self.assertEqual(result["close"].tolist(), [10.0, 20.0])

    def test_normalize_day_data_keeps_rq_units_and_aligns_schema(self):
        from my_utils import rq_fun as module
        raw_price = pd.DataFrame(
            {
                "open": [10.1],
                "high": [10.8],
                "low": [9.9],
                "close": [10.5],
                "prev_close": [10.0],
                "volume": [123456.0],
                "total_turnover": [1296288.0],
                "limit_up": [11.0],
                "limit_down": [9.0],
            },
            index=pd.MultiIndex.from_arrays(
                [["000001.XSHE"], pd.to_datetime(["2026-03-23"])],
                names=["order_book_id", "date"],
            ),
        )
        instruments = pd.DataFrame(
            {
                "order_book_id": ["000001.XSHE"],
                "symbol": ["平安银行"],
                "listed_date": ["1991-04-03"],
                "de_listed_date": ["0000-00-00"],
                "status": ["Active"],
            }
        )
        is_st = pd.DataFrame({"000001.XSHE": [False]}, index=pd.to_datetime(["2026-03-23"]))
        suspended = pd.DataFrame({"000001.XSHE": [True]}, index=pd.to_datetime(["2026-03-23"]))
        turnover = pd.DataFrame(
            {"today": [0.8142]},
            index=pd.MultiIndex.from_arrays(
                [["000001.XSHE"], pd.to_datetime(["2026-03-23"])],
                names=["order_book_id", "date"],
            ),
        )
        shares = pd.DataFrame(
            {
                "code": ["SZSE.000001"],
                "trading_date": [dt.date(2026, 3, 23)],
                "total_a": [202791845169.0 / 10.5],
                "circulation_a": [202791845169.1 / 10.5],
                "free_circulation": [202788526823.85 / 10.5],
            }
        )
        adj = pl.DataFrame(
            {
                "code": ["SZSE.000001"],
                "trading_date": [dt.date(2026, 3, 23)],
                "adj_factor": [177.964],
            }
        )

        result = module.normalize_day_data(raw_price, instruments, is_st, suspended, turnover, shares, adj)

        self.assertEqual(result.schema, module.RQ_DAY_SCHEMA)
        row = result.row(0, named=True)
        self.assertEqual(row["code"], "SZSE.000001")
        self.assertEqual(row["name"], "平安银行")
        self.assertEqual(row["amount"], 1296288.0)
        self.assertEqual(row["volume"], 123456.0)
        self.assertEqual(row["total_mv"], 202791845169.0)
        self.assertEqual(row["mv_A_free_float"], 202788526823.85)
        self.assertEqual(row["adj_factor"], 177.964)
        self.assertFalse(row["is_st"])
        self.assertTrue(row["is_suspended"])

    def test_normalize_minute_data_shifts_rq_end_time_to_local_bar_start(self):
        from my_utils import rq_fun as module
        raw_minute = pd.DataFrame(
            {
                "open": [10.68, 10.54],
                "high": [10.68, 10.64],
                "low": [10.47, 10.54],
                "close": [10.55, 10.61],
                "volume": [46986480.0, 15199823.0],
            },
            index=pd.MultiIndex.from_arrays(
                [["000001.XSHE", "000001.XSHE"], pd.to_datetime(["2026-03-23 09:45:00", "2026-03-23 10:00:00"])],
                names=["order_book_id", "date"],
            ),
        )

        result = module.normalize_minute_data(raw_minute, bar_minutes=15)

        self.assertEqual(result.schema, module.RQ_MIN_SCHEMA)
        self.assertEqual(result["code"].to_list(), ["SZSE.000001", "SZSE.000001"])
        self.assertEqual(result["datetime"].dt.time().to_list(), [dt.time(9, 30), dt.time(9, 45)])
        self.assertEqual(result["trading_date"].to_list(), [dt.date(2026, 3, 23), dt.date(2026, 3, 23)])

    def test_build_daily_adj_factor_forward_fills_ex_cum_factor(self):
        from my_utils import rq_fun as module
        ex_factor = pd.DataFrame(
            {
                "order_book_id": ["000001.XSHE", "000001.XSHE"],
                "ex_cum_factor": [100.0, 120.0],
            },
            index=pd.to_datetime(["2026-01-01", "2026-03-01"]),
        )
        trading_dates = [dt.date(2025, 12, 31), dt.date(2026, 1, 2), dt.date(2026, 3, 2)]

        result = module.build_daily_adj_factor(ex_factor, ["000001.XSHE"], trading_dates)

        self.assertEqual(result.schema, module.RQ_ADJ_SCHEMA)
        self.assertEqual(result["adj_factor"].to_list(), [1.0, 100.0, 120.0])

    def test_normalize_ex_factor_parses_rq_millisecond_index(self):
        from my_utils import rq_fun as module
        ex_factor = pd.DataFrame(
            {
                "order_book_id": ["000001.XSHE"],
                "ex_cum_factor": [177.964],
            },
            index=[pd.Timestamp("2025-10-15").value // 1_000_000],
        )

        result = module.normalize_ex_factor(ex_factor)

        self.assertEqual(result["ex_date"].tolist(), [dt.date(2025, 10, 15)])

    def test_trade_accepts_min_data_file_path(self):
        my_utils_pkg = types.ModuleType("my_utils")
        my_utils_pkg.__path__ = [MY_UTILS_DIR]
        sys.modules["my_utils"] = my_utils_pkg
        my_utils_pkg.fun = _load_module("my_utils.fun", os.path.join(MY_UTILS_DIR, "fun.py"))
        trade_fun = _load_module("my_utils.trade_fun", os.path.join(MY_UTILS_DIR, "trade_fun.py"))

        day = dt.date(2026, 3, 23)
        read_min_paths = []
        day_rows = [
            {
                "code": "SZSE.000001",
                "trading_date": day,
                "pre_close": 10.0,
                "limit_up": 11.0,
                "limit_down": 9.0,
                "adj_factor": 1.0,
            }
        ]
        min_rows = [
            {
                "code": "SZSE.000001",
                "trading_date": day,
                "datetime": dt.datetime.combine(day, dt.time(9, 30)),
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 100.0,
            }
        ]

        def fake_read_min_data(*args, **kwargs):
            read_min_paths.append(kwargs.get("file_path"))
            return pl.DataFrame(min_rows)

        with patch.object(trade_fun, "read_day_data", return_value=pl.DataFrame(day_rows)), patch.object(
            trade_fun, "read_min_data", side_effect=fake_read_min_data
        ):
            trade_fun.trade(
                ["SZSE.000001"],
                day,
                min_data_file_path="rq_15min_stock_data_dir",
            )

        self.assertEqual(read_min_paths[0], "rq_15min_stock_data_dir")


if __name__ == "__main__":
    unittest.main()
