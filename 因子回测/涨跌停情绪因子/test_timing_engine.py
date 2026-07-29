"""
回归测试：清洗后的日线收益列为 stock_daily_ret，基准构造不得再依赖原始 pct。
"""

from pathlib import Path
import importlib.util

import polars as pl


MODULE_PATH = Path(__file__).with_name("timing_engine.py")
SPEC = importlib.util.spec_from_file_location("timing_engine", MODULE_PATH)
timing_engine = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(timing_engine)


def test_value_weighted_benchmark_uses_cleaned_daily_return_column():
    """
    若实现重新读取 pct，这个只含清洗后字段的入参会立即复现 ColumnNotFoundError。
    """
    prepared = pl.DataFrame(
        {
            "trading_date": ["2024-01-02", "2024-01-03"],
            "code_prefix": ["6", "6"],
            "benchmark_weight": [100.0, 100.0],
            "stock_daily_ret": [0.10, -0.05],
        }
    ).with_columns(pl.col("trading_date").str.to_date())
    calendar = prepared.select("trading_date")

    actual = timing_engine.build_value_weighted_benchmark(prepared, calendar)

    assert actual["market_daily_ret"].tolist() == [0.10, -0.05]
