import importlib.util
import os
import sys
import unittest

import polars as pl

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPT_PATH = os.path.join(ROOT_DIR, "hot_rebound_data_source_backtest.py")
sys.path.insert(0, ROOT_DIR)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("hot_rebound_data_source_backtest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hot_rebound_data_source_backtest"] = module
    spec.loader.exec_module(module)
    return module


class HotReboundDataSourceBacktestTests(unittest.TestCase):
    def test_normalize_gm_data_matches_strategy_field_units(self):
        module = _load_script_module()
        raw = pl.DataFrame(
            {
                "code": ["SHSE.600000", "SHSE.600001"],
                "trading_date": ["2024-01-02", "2024-01-02"],
                "open": [10.0, 20.0],
                "high": [11.0, 21.0],
                "low": [9.0, 19.0],
                "close": [10.5, 20.5],
                "pre_close": [10.0, 20.0],
                "limit_up": [11.0, 0.0],
                "limit_down": [9.0, 0.0],
                "volume": [123400.0, 200000.0],
                "amount": [1295700.0, 4100000.0],
                "turnover_rate": [3.5, 4.2],
                "mv_A_free_float": [12_300_000_000.0, 45_600_000_000.0],
                "total_mv": [20_000_000_000.0, 60_000_000_000.0],
                "is_st": [False, True],
                "name": ["AAA", "BBB"],
            }
        )

        result = module.normalize_gm_stock_data(raw)

        self.assertEqual(result["volume"].to_list(), [1234.0, 2000.0])
        self.assertEqual(result["amount"].to_list(), [129.57, 410.0])
        self.assertEqual(result["turn_over"].to_list(), [3.5, 4.2])
        self.assertEqual(result["free_float_mv"].to_list(), [123.0, 456.0])
        self.assertEqual(result["type"].to_list(), [None, "ST"])
        self.assertEqual(result["limit_up"].to_list(), [11.0, None])
        self.assertEqual(result["limit_down"].to_list(), [9.0, None])


if __name__ == "__main__":
    unittest.main()
