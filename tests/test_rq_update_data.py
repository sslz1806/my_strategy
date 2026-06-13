import datetime as dt
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd
import polars as pl

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MY_UTILS_DIR = os.path.join(ROOT_DIR, "my_utils")
SCRIPT_PATH = os.path.join(ROOT_DIR, "任务", "米筐数据更新.py")
sys.path.insert(0, ROOT_DIR)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("rq_update_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rq_update_data"] = module
    spec.loader.exec_module(module)
    return module


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
        self.assertEqual(convert_code_format(["000001.XSHE", "600000.XSHG"], format="suffix"), ["000001.SZ", "600000.SH"])

    def test_expand_rq_multiindex_adds_code_and_date_columns(self):
        module = _load_script_module()
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
        module = _load_script_module()
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
        market_value = pd.DataFrame(
            {
                "market_cap": [202791845169.0],
                "a_share_market_val": [202791845169.1],
                "a_share_market_val_in_circulation": [202788526823.85],
            },
            index=pd.MultiIndex.from_arrays(
                [["000001.XSHE"], pd.to_datetime(["2026-03-23"])],
                names=["order_book_id", "date"],
            ),
        )
        adj = pl.DataFrame(
            {
                "code": ["SZSE.000001"],
                "trading_date": [dt.date(2026, 3, 23)],
                "adj_factor": [177.964],
            }
        )

        result = module.normalize_day_data(raw_price, instruments, is_st, suspended, turnover, market_value, adj)

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
        module = _load_script_module()
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
        module = _load_script_module()
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
        module = _load_script_module()
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
