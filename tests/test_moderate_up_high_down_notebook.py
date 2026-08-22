"""“适度涨停 + 大量跌停”复合因子 Notebook 的行为验收测试。"""

import ast
from bisect import bisect_left, bisect_right, insort
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "因子回测"
    / "涨跌停情绪因子"
    / "适度涨停_大量跌停_分组回测.ipynb"
)


def load_notebook_functions() -> dict:
    """只执行 Cell 2 中的函数定义，避免测试访问真实行情和外部数据源。"""
    assert NOTEBOOK_PATH.exists(), f"Notebook 尚未创建：{NOTEBOOK_PATH}"
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    tree = ast.parse(notebook.cells[2].source)
    function_nodes = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    function_module = ast.Module(body=function_nodes, type_ignores=[])
    ast.fix_missing_locations(function_module)

    namespace = {
        "np": np,
        "pd": pd,
        "bisect_left": bisect_left,
        "bisect_right": bisect_right,
        "insort": insort,
    }
    exec(compile(function_module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def test_all_code_cells_compile():
    """生成 Notebook 时不得把转义换行写成破坏字符串语法的真实换行。"""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK_PATH}:cell_{cell_index}", "exec")


def test_causal_percentile_uses_only_prior_history_and_midrank_for_ties():
    """修改未来观测不得改变此前百分位；相同历史值按中间秩处理。"""
    namespace = load_notebook_functions()
    causal_percentile = namespace["causal_expanding_percentile"]

    original = pd.Series([1.0, 2.0, 2.0, 2.0, 100.0])
    changed_future = original.copy()
    changed_future.iloc[-1] = -100.0

    original_rank = causal_percentile(original, min_history=3)
    changed_rank = causal_percentile(changed_future, min_history=3)

    assert original_rank.iloc[:3].isna().all()
    assert np.isclose(original_rank.iloc[3], 2.0 / 3.0)
    pd.testing.assert_series_equal(original_rank.iloc[:4], changed_rank.iloc[:4])


def test_composite_score_peaks_at_moderate_up_and_rewards_high_down():
    """涨停百分位居中且跌停百分位更高时，复合因子得分应更高。"""
    namespace = load_notebook_functions()
    build_factor = namespace["build_moderate_up_high_down_factor"]

    up_percentile = pd.Series([0.60, 0.40, 0.60, 0.90])
    down_percentile = pd.Series([0.80, 0.80, 0.40, 0.80])
    score = build_factor(
        up_percentile,
        down_percentile,
        up_center=0.60,
        up_width=0.20,
        down_power=2,
    )

    assert np.isclose(score.iloc[0], 0.64)
    assert score.iloc[0] > score.iloc[1]
    assert score.iloc[0] > score.iloc[2]
    assert score.iloc[0] > score.iloc[3]
    assert score.between(0.0, 1.0).all()
