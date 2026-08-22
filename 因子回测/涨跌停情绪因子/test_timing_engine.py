"""
回归测试：清洗后的日线收益列为 stock_daily_ret，基准构造不得再依赖原始 pct。
"""

import importlib
import importlib.util
import json
from pathlib import Path

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


def test_notebook_reload_restores_new_timing_engine_exports():
    """Notebook 重跑导入单元时，应刷新内核中缓存的旧版择时模块。"""
    notebook_path = Path(__file__).with_name("sentiment_factors_5d_research.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup_cell = next(
        cell
        for cell in notebook["cells"]
        if "timing_engine import" in "".join(cell.get("source", []))
    )
    setup_source = "".join(setup_cell["source"])
    assert "run_multi_benchmark_timing" not in setup_source
    assert "plot_multi_benchmark_summary" not in setup_source

    module_name = "因子回测.涨跌停情绪因子.timing_engine"
    cached_module = importlib.import_module(module_name)
    del cached_module.analyze_ic

    try:
        namespace = {}
        exec(setup_source, namespace)
        assert namespace["analyze_ic"] is cached_module.analyze_ic
        assert namespace["run_time_backtest"] is cached_module.run_time_backtest
    finally:
        # 即使断言失败，也恢复共享解释器中的模块，避免污染后续测试。
        importlib.reload(cached_module)
