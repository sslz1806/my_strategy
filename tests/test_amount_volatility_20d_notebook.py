"""成交额 20 日波动率因子 Notebook 的核心口径回归测试。"""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import nbformat
import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "因子回测" / "成交额20日波动率因子.ipynb"


def load_notebook_function(function_name: str):
    """只执行指定函数定义，避免单元测试读取本地全量行情。"""
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


def build_amount_panel(periods: int = 21) -> pl.DataFrame:
    """构造递增成交额与恒定成交额两只股票的确定性面板。"""
    rows = []
    for code in ("A", "B"):
        for index in range(periods):
            rows.append(
                {
                    "code": code,
                    "trading_date": date(2024, 1, 1) + timedelta(days=index),
                    "amount": float(index + 1) if code == "A" else 100.0,
                }
            )
    return pl.DataFrame(rows)


def test_first_factor_value_requires_twenty_complete_observations():
    """防止滚动窗口误用不足 20 期的部分样本。"""
    add_factor = load_notebook_function("add_amount_volatility_20d_factor")
    result = add_factor(build_amount_panel(), window=20)
    stock_a = result.filter(pl.col("code") == "A").sort("trading_date")

    assert stock_a["amount_volatility_20d"][:19].null_count() == 19
    # 1..20 的样本方差为 35，因此样本标准差为 sqrt(35)。
    assert np.isclose(stock_a["amount_volatility_20d"][19], np.sqrt(35.0))


def test_rolling_window_is_isolated_by_stock():
    """防止不同股票的成交额被串入同一个滚动窗口。"""
    add_factor = load_notebook_function("add_amount_volatility_20d_factor")
    result = add_factor(build_amount_panel(), window=20)
    stock_b = result.filter(pl.col("code") == "B").sort("trading_date")

    assert np.isclose(stock_b["amount_volatility_20d"][19], 0.0)


def test_future_amount_change_does_not_rewrite_earlier_factor():
    """防止滚动方向写反或因子构造引入未来成交额。"""
    add_factor = load_notebook_function("add_amount_volatility_20d_factor")
    original = build_amount_panel(periods=21)
    changed_future = original.with_columns(
        pl.when(
            (pl.col("code") == "A")
            & (pl.col("trading_date") == date(2024, 1, 21))
        )
        .then(1_000_000.0)
        .otherwise(pl.col("amount"))
        .alias("amount")
    )

    factor_date = date(2024, 1, 20)
    original_value = add_factor(original, window=20).filter(
        (pl.col("code") == "A") & (pl.col("trading_date") == factor_date)
    )["amount_volatility_20d"].item()
    changed_value = add_factor(changed_future, window=20).filter(
        (pl.col("code") == "A") & (pl.col("trading_date") == factor_date)
    )["amount_volatility_20d"].item()

    assert np.isclose(original_value, changed_value)


def test_all_code_cells_compile():
    """Notebook 的所有代码单元格必须能独立通过 Python 语法编译。"""
    assert NOTEBOOK_PATH.exists(), f"Notebook 尚未创建：{NOTEBOOK_PATH}"
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK_PATH}:cell_{cell_index}", "exec")
