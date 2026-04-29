import datetime as dt
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch

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


my_utils_pkg = types.ModuleType("my_utils")
my_utils_pkg.__path__ = [MY_UTILS_DIR]
sys.modules["my_utils"] = my_utils_pkg
my_utils_pkg.fun = _load_module("my_utils.fun", os.path.join(MY_UTILS_DIR, "fun.py"))
tf = _load_module("my_utils.trade_fun", os.path.join(MY_UTILS_DIR, "trade_fun.py"))


class TradeDelayTests(unittest.TestCase):
    def test_cal_trade_info_joins_delayed_buy_result_on_signal_date(self):
        signal_date = dt.date(2024, 1, 2)
        buy_date = dt.date(2024, 1, 3)
        signal_df = pl.DataFrame(
            {
                "trading_date": [signal_date],
                "code": ["000001"],
                "signal": [1],
            }
        )

        def fake_trade(
            code_list,
            trade_date,
            buy_delay_days=0,
            extend_holding_days=0,
            day_data_file_path="gm_stock_all_data",
        ):
            return [
                {
                    "code": code_list[0],
                    "buy_time": dt.datetime.combine(buy_date, dt.time(9, 30)),
                    "buy_price": 10.0,
                    "sell_time": dt.datetime.combine(dt.date(2024, 1, 4), dt.time(11, 30)),
                    "sell_price": 10.5,
                    "profit": 5.0,
                    "holding_days": 1.5,
                    "sell_reason": "未涨停卖出",
                }
            ]

        result_df, merged_df = tf.cal_trade_info(
            signal_df,
            trade_fun=fake_trade,
            start_date="2024-01-02",
            end_date="2024-01-02",
            buy_delay_days=1,
        )

        self.assertEqual(len(result_df), 1)
        self.assertEqual(merged_df.height, 1)
        self.assertEqual(merged_df["trading_date"].to_list(), [signal_date])
        self.assertEqual(merged_df["buy_time"].dt.date().to_list(), [buy_date])
        self.assertEqual(merged_df["profit"].to_list(), [5.0])

    def test_trade_starts_sell_scan_after_delayed_buy_date(self):
        code = "000001"
        days = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
        day_rows = [
            {
                "trading_date": day,
                "code": code,
                "pre_close": 10.0,
                "limit_up": 12.0,
                "limit_down": 8.0,
            }
            for day in days
        ]
        min_rows = []
        for day in days:
            for t in [dt.time(9, 30), dt.time(11, 30), dt.time(15, 0)]:
                price = 10.0
                min_rows.append(
                    {
                        "trading_date": day,
                        "code": code,
                        "datetime": dt.datetime.combine(day, t),
                        "open": price,
                        "close": price,
                        "high": price,
                        "low": price,
                    }
                )

        with patch.object(tf, "read_day_data", return_value=pl.DataFrame(day_rows)), patch.object(
            tf, "read_min_data", return_value=pl.DataFrame(min_rows)
        ):
            result = tf.trade(
                [code],
                days[0],
                need_adj=False,
                buy_delay_days=1,
                extend_holding_days=0,
            )

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["buy_time"].date(), days[1])
        self.assertEqual(result[0]["sell_time"].date(), days[2])
        self.assertGreater(result[0]["holding_days"], 0)

    def test_trade_uses_configured_day_data_source(self):
        code = "000001"
        day = dt.date(2024, 1, 2)
        used_file_paths = []

        day_rows = [
            {
                "trading_date": day,
                "code": code,
                "pre_close": 10.0,
                "limit_up": 12.0,
                "limit_down": 8.0,
            }
        ]
        min_rows = [
            {
                "trading_date": day,
                "code": code,
                "datetime": dt.datetime.combine(day, dt.time(9, 30)),
                "open": 10.0,
                "close": 10.0,
                "high": 10.0,
                "low": 10.0,
            }
        ]

        def fake_read_day_data(*args, **kwargs):
            used_file_paths.append(kwargs.get("file_path"))
            return pl.DataFrame(day_rows)

        with patch.object(tf, "read_day_data", side_effect=fake_read_day_data), patch.object(
            tf, "read_min_data", return_value=pl.DataFrame(min_rows)
        ):
            tf.trade(
                [code],
                day,
                need_adj=False,
                day_data_file_path="ts_stock_all_data",
            )

        self.assertEqual(used_file_paths[0], "ts_stock_all_data")

    def test_consecutive_loss_weight_treats_all_null_profit_day_as_flat(self):
        data = pl.DataFrame(
            {
                "trading_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
                "code": ["000001", "000002"],
                "profit": [None, 1.0],
            }
        )

        result = tf.adjust_weight_by_consecutive_losses(data)

        self.assertEqual(result["weight_consec_loss"].to_list(), [0.4, 0.4])


if __name__ == "__main__":
    unittest.main()
