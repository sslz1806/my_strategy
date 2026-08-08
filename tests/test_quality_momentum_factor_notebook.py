"""质量动量因子 Notebook 的行为验收测试。"""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import nbformat
import numpy as np
import polars as pl
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "因子回测" / "质量动量因子.ipynb"


def load_notebook_functions() -> dict:
    """仅执行 Notebook 中的函数定义，避免测试读取真实行情。"""
    notebook = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    function_nodes = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        source = "\n".join(
            line
            for line in cell.source.splitlines()
            if not line.lstrip().startswith("%")
        )
        tree = ast.parse(source)
        function_nodes.extend(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )

    module = ast.Module(body=function_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "pl": pl}
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def test_example_score_matches_literal_reference_values():
    """示例版必须保留线性权重与原始 R² 口径。"""
    calc_score = load_notebook_functions()["calc_momentum_score"]
    x = np.arange(26, dtype=float)
    prices = 100.0 * np.exp(0.002 * x + 0.00008 * x**2)

    actual = calc_score(prices, mode="example")
    expected = (1.8578009654578476, 0.9831522318751459, 1.8265011655696837)

    assert np.allclose(actual, expected)


def test_exponential_trend_has_known_return_and_perfect_fit():
    """完全指数趋势应在两个权重版本中得到已知年化收益和 R²=1。"""
    calc_score = load_notebook_functions()["calc_momentum_score"]
    prices = 100.0 * np.exp(0.001 * np.arange(26, dtype=float))
    expected_annual_ret = 0.2840254166877415

    for mode in ("example", "halflife"):
        annual_ret, r2, score = calc_score(prices, mode=mode)
        assert np.isclose(annual_ret, expected_annual_ret)
        assert np.isclose(r2, 1.0)
        assert np.isclose(score, expected_annual_ret)


def test_constant_prices_return_zero_metrics():
    """常数价格不应因 R² 分母为零产生 NaN 或无意义得分。"""
    calc_score = load_notebook_functions()["calc_momentum_score"]

    for mode in ("example", "halflife"):
        assert np.allclose(
            calc_score(np.full(26, 10.0), mode=mode),
            (0.0, 0.0, 0.0),
        )


def test_halflife_weights_and_invalid_parameters():
    """半衰期版的权重方向和非法参数必须明确。"""
    weights = load_notebook_functions()["quality_momentum_weights"]
    halflife = weights(26, mode="halflife", half_life=63.0)

    assert np.isclose(halflife[0], 0.5 ** (25 / 63))
    assert np.isclose(halflife[-1], 1.0)
    assert np.all(np.diff(halflife) > 0)

    with pytest.raises(ValueError, match="window"):
        weights(1, mode="example")
    with pytest.raises(ValueError, match="half_life"):
        weights(26, mode="halflife", half_life=0)
    with pytest.raises(ValueError, match="mode"):
        weights(26, mode="unknown")


def test_rolling_factor_waits_for_26_prices_and_stays_within_code():
    """滚动窗口不能跨股票，且无效价格会使涉及它的窗口保持空值。"""
    add_factors = load_notebook_functions()["add_quality_momentum_factors"]
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(30)]
    rows = []
    for code in ("B", "A"):
        for index, trading_date in enumerate(dates):
            close = 100.0 * np.exp(0.001 * index)
            if code == "B" and index == 4:
                close = 0.0
            rows.append({"code": code, "trading_date": trading_date, "close": close})

    source = pl.DataFrame(rows).reverse()
    result = add_factors(source, window=26)
    a_rows = result.filter(pl.col("code") == "A").sort("trading_date")
    b_rows = result.filter(pl.col("code") == "B").sort("trading_date")

    assert a_rows["momentum_score"].head(25).null_count() == 25
    assert a_rows["momentum_score"].tail(5).null_count() == 0
    assert np.allclose(a_rows["momentum_r2"].tail(5), np.ones(5))
    assert b_rows["momentum_score"].null_count() == 30
    assert result.select("code", "trading_date").rows() == (
        source.sort("code", "trading_date").select("code", "trading_date").rows()
    )


def test_rolling_factor_rejects_missing_input_columns():
    """公共输入列缺失时应立即报出具体列名。"""
    add_factors = load_notebook_functions()["add_quality_momentum_factors"]
    frame = pl.DataFrame({"code": ["A"], "trading_date": [date(2024, 1, 1)]})

    with pytest.raises(ValueError, match="close"):
        add_factors(frame)


def test_all_python_cells_compile_after_removing_ipython_magics():
    """Notebook 代码单元去除 IPython 魔法后必须保持普通 Python 语法有效。"""
    notebook = nbformat.read(str(NOTEBOOK_PATH), as_version=4)

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = "\n".join(
            line
            for line in cell.source.splitlines()
            if not line.lstrip().startswith("%")
        )
        compile(source, f"{NOTEBOOK_PATH}:cell_{cell_index}", "exec")


def test_backtest_cell_analyzes_both_score_columns_with_same_parameters():
    """两个得分列必须在同一回测参数下各调用一次 analyze_factor。"""
    notebook = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    backtest_cell = [cell for cell in notebook.cells if cell.cell_type == "code"][-1]
    calls = []

    def fake_analyze_factor(**kwargs):
        calls.append(kwargs)
        return {
            "ic_stats": pl.DataFrame({"window": [1], "ic_mean": [0.1]}),
            "group_stats": pl.DataFrame({"window": [1], "group": ["G1"]}),
        }

    namespace = {
        "analysis_data": pl.DataFrame(
            {
                "trading_date": [date(2024, 1, 1)],
                "code": ["A"],
                "momentum_score": [1.0],
                "momentum_score_halflife": [1.0],
                "daily_ret": [0.01],
                "benchmark_ret": [0.01],
            }
        ),
        "analyze_factor": fake_analyze_factor,
    }
    exec(backtest_cell.source, namespace)

    assert [call["factor_col"] for call in calls] == [
        "momentum_score",
        "momentum_score_halflife",
    ]
    assert all(call["ret_windows"] == [1, 3, 5] for call in calls)
    assert all(call["ic_windows"] == [1, 3, 5] for call in calls)
    assert all(call["group_num"] == 5 for call in calls)
