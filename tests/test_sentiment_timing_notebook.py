"""涨跌停情绪择时 Notebook 的结构回归测试。

该测试不读取本地行情数据，只锁定研究 Notebook 的两个可维护性要求：
一是因子有效性检验必须调用已测试的统计函数；二是策略函数应集中定义，避免
执行顺序依赖和无意义的多 Cell 拆分。
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import polars as pl


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "因子回测"
    / "涨跌停情绪因子"
    / "reproduce_sentiment_timing.ipynb"
)


def test_notebook_has_complete_effectiveness_section_and_combined_strategy_helpers() -> None:
    """Notebook 必须展示完整检验，并把相关策略函数放在同一个代码单元。"""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]

    assert "## 5. 因子有效性检验" in markdown
    assert any("analyze_factor_effectiveness(" in source for source in code_cells)
    assert any("analyze_threshold_effectiveness(" in source for source in code_cells)
    assert "触发收益 - 未触发收益" in "\n".join(code_cells)

    strategy_cells = [
        source
        for source in code_cells
        if "def generate_signals(" in source
        or "def run_backtest(" in source
        or "def compute_perf(" in source
    ]
    assert len(strategy_cells) == 1
    assert all(
        function in strategy_cells[0]
        for function in (
            "def generate_signals(",
            "def run_backtest(",
            "def compute_perf(",
        )
    )


def test_strategy_performance_table_does_not_require_ipython_display(capsys) -> None:
    """回测展示函数也应能被脚本或 nbconvert 调用，而非依赖交互式隐式名称。"""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    strategy_source = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "def print_perf_table(" in cell.source
    )
    namespace = {
        "np": np,
        "pd": pd,
        "pl": pl,
        "FACTOR_WEIGHTS": {
            "net_limit_ratio": 0.15,
            "limit_down_next_ret": 0.10,
            "limit_up_ratio": 0.25,
            "limit_down_ratio": 0.20,
            "chase_ret": 0.30,
        },
    }
    exec(strategy_source, namespace)

    namespace["print_perf_table"](
        {
            "测试策略": {
                "年化收益率": 0.10,
                "年化波动率": 0.20,
                "最大回撤": -0.08,
                "夏普比率": 0.50,
                "胜率": 0.55,
                "盈亏比": 1.20,
            }
        },
        "回测展示",
    )

    assert "测试策略" in capsys.readouterr().out
