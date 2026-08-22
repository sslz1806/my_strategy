"""20 日收益率动量因子 Notebook 的核心口径验收测试。"""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import nbformat
import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "因子回测" / "20日收益率动量因子.ipynb"


def load_notebook_function(function_name: str):
    """只执行指定 Notebook 函数，避免单元测试读取本地全量行情。"""
    assert NOTEBOOK_PATH.exists(), f"Notebook 尚未创建：{NOTEBOOK_PATH}"
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                function_module = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(function_module)
                namespace = {"pl": pl}
                exec(compile(function_module, str(NOTEBOOK_PATH), "exec"), namespace)
                return namespace[function_name]

    raise AssertionError(f"Notebook 未定义 {function_name}")


def load_factor_function():
    return load_notebook_function("add_momentum_20d_factor")


def build_constant_return_panel(periods: int = 21) -> pl.DataFrame:
    """构造两只股票的确定性收益序列，预期值可直接手算。"""
    rows = []
    for code, daily_return in [("A", 0.01), ("B", -0.01)]:
        close = 10.0
        for index in range(periods):
            pre_close = close
            close = pre_close * (1 + daily_return)
            rows.append(
                {
                    "code": code,
                    "trading_date": date(2024, 1, 1) + timedelta(days=index),
                    "close": close,
                    "pre_close": pre_close,
                }
            )
    return pl.DataFrame(rows)


def test_twenty_day_factor_compounds_returns_within_each_stock():
    """捕获把不同股票串联滚动、少算一期或把百分比误当小数的错误。"""
    add_factor = load_factor_function()
    result = add_factor(build_constant_return_panel(), window=20)

    for code, expected in [("A", 1.01**20 - 1), ("B", 0.99**20 - 1)]:
        stock_result = result.filter(pl.col("code") == code).sort("trading_date")
        assert stock_result["momentum_20d"][:19].null_count() == 19
        assert np.isclose(stock_result["momentum_20d"][19], expected)


def test_future_return_change_does_not_rewrite_earlier_factor():
    """捕获滚动窗口方向写反、从未来收益构造因子的前视偏差。"""
    add_factor = load_factor_function()
    original = build_constant_return_panel(periods=21)
    changed_future = original.with_columns(
        pl.when(
            (pl.col("code") == "A")
            & (pl.col("trading_date") == date(2024, 1, 21))
        )
        .then(pl.col("pre_close") * 2)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    original_result = add_factor(original, window=20)
    changed_result = add_factor(changed_future, window=20)
    factor_date = date(2024, 1, 20)
    original_value = original_result.filter(
        (pl.col("code") == "A") & (pl.col("trading_date") == factor_date)
    )["momentum_20d"].item()
    changed_value = changed_result.filter(
        (pl.col("code") == "A") & (pl.col("trading_date") == factor_date)
    )["momentum_20d"].item()

    assert np.isclose(original_value, changed_value)


def test_factor_uses_pre_close_return_across_adjustment_gap():
    """捕获直接用原始 close 比值导致除权窗口产生虚假负动量的错误。"""
    add_factor = load_factor_function()
    rows = [
        {
            "code": "A",
            "trading_date": date(2024, 1, 1) + timedelta(days=index),
            "close": 5.0 if index == 19 else 10.0,
            "pre_close": 5.0 if index == 19 else 10.0,
        }
        for index in range(20)
    ]

    result = add_factor(pl.DataFrame(rows), window=20).sort("trading_date")

    assert np.isclose(result["momentum_20d"][-1], 0.0)


def test_forward_horizon_audit_flags_stock_rows_that_skip_market_dates():
    """捕获把个股下一条记录无条件当作下一市场交易日的 IC 对齐风险。"""
    audit_alignment = load_notebook_function("audit_forward_horizon_alignment")
    market_dates = [date(2024, 1, day) for day in (2, 3, 4)]
    panel = pl.DataFrame(
        {
            "trading_date": market_dates + [market_dates[0], market_dates[2]],
            "code": ["A", "A", "A", "B", "B"],
        }
    )

    result = audit_alignment(panel, horizons=(1,))
    row = result.row(0, named=True)

    assert row["window"] == 1
    assert row["comparable_pairs"] == 3
    assert row["misaligned_pairs"] == 1
    assert np.isclose(row["misaligned_ratio"], 1 / 3)


def test_source_profile_counts_duplicate_keys_and_core_nulls():
    """捕获股票日线主键重复或核心行情字段缺失却未被披露的问题。"""
    profile_source = load_notebook_function("profile_stock_daily_source")
    panel = pl.DataFrame(
        {
            "trading_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3)],
            "code": ["A", "A", "B"],
            "open": [10.0, 10.0, None],
            "close": [10.0, 10.0, 20.0],
            "pre_close": [9.0, 9.0, 19.0],
            "limit_up": [11.0, 11.0, 22.0],
            "limit_down": [9.0, 9.0, 18.0],
            "total_mv": [100.0, 100.0, 200.0],
        }
    )

    row = profile_source(panel).row(0, named=True)

    assert row["rows"] == 3
    assert row["duplicate_key_rows"] == 2
    assert row["open_nulls"] == 1
    assert row["close_nulls"] == 0
    assert row["date_min"] == date(2024, 1, 2)
    assert row["date_max"] == date(2024, 1, 3)


def test_all_code_cells_compile():
    """Notebook 的所有代码单元格必须能独立通过 Python 语法编译。"""
    assert NOTEBOOK_PATH.exists(), f"Notebook 尚未创建：{NOTEBOOK_PATH}"
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK_PATH}:cell_{cell_index}", "exec")
