"""双信号周频择时回测的关键口径测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT_ROOT / "因子回测" / "涨跌停情绪因子"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from double_signal_timing_backtest import (  # noqa: E402
    calculate_performance,
    calculate_strategy_returns,
    combine_pair_position,
    load_weekly_backtest_data,
)


def test_pair_rules_map_two_binary_signals_to_expected_positions() -> None:
    """AND、SCORE 和 OR 必须分别对应确认、分档和放宽三种仓位语义。"""
    first = pd.Series([0, 0, 1, 1])
    second = pd.Series([0, 1, 0, 1])

    assert combine_pair_position(first, second, "AND").tolist() == [0.0, 0.0, 0.0, 1.0]
    assert combine_pair_position(first, second, "SCORE").tolist() == [0.0, 0.5, 0.5, 1.0]
    assert combine_pair_position(first, second, "OR").tolist() == [0.0, 1.0, 1.0, 1.0]


def test_cost_is_charged_on_each_position_change_including_initial_entry() -> None:
    """成本按单边仓位变化计提，不能只在买入或只在卖出时扣除。"""
    next_week_returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    position = pd.Series([1.0, 1.0, 0.0, 0.5])

    returns, turnover = calculate_strategy_returns(
        next_week_returns, position, cost_bps=10
    )

    assert turnover.tolist() == [1.0, 0.0, 1.0, 0.5]
    assert returns.tolist() == pytest.approx([0.009, 0.02, -0.001, 0.0145])


def test_performance_drawdown_includes_initial_nav_of_one() -> None:
    """首周亏损也必须计入最大回撤，不能从首周收盘后的净值才开始找高点。"""
    returns = pd.Series([-0.10, 0.05, 0.02, 0.01])
    position = pd.Series([1.0, 1.0, 1.0, 1.0])
    turnover = pd.Series([1.0, 0.0, 0.0, 0.0])

    metrics = calculate_performance(returns, position, turnover)

    assert metrics["最大回撤"] == pytest.approx(-0.10)
    assert metrics["平均仓位"] == pytest.approx(1.0)
    assert metrics["持仓周数"] == 4


def test_loader_rejects_same_week_returns_and_treats_factor_null_as_no_signal(
    tmp_path: Path,
) -> None:
    """收益必须来自下一周；合法事件因子空值沿用主研究规则记为不触发。"""
    rows = pd.DataFrame(
        {
            "week_id": ["2024-01", "2024-02"],
            "week_end_date": ["2024-01-05", "2024-01-12"],
            "return_week_end_date": ["2024-01-12", "2024-01-19"],
            "next_week_ret": [0.01, -0.02],
            "limit_up_ratio": [0.09, 0.01],
            "limit_down_ratio": [0.005, 0.02],
            "net_limit_ratio": [0.05, -0.01],
            "limit_up_next_ret": [np.nan, 0.03],
            "limit_down_next_ret": [-0.02, 0.01],
            "chase_ret": [0.00, -0.02],
        }
    )
    input_path = tmp_path / "weekly.csv"
    rows.to_csv(input_path, index=False)

    loaded = load_weekly_backtest_data(input_path)
    assert loaded["signal_limit_up_next_ret"].tolist() == [0, 1]
    assert loaded["signal_limit_up_ratio"].tolist() == [1, 0]

    rows.loc[0, "return_week_end_date"] = rows.loc[0, "week_end_date"]
    rows.to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="持有收益结束日"):
        load_weekly_backtest_data(input_path)
