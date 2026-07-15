"""涨跌停情绪择时核心计算的回归测试。

这些测试只使用极小的合成样本，重点锁定研究中最容易引入未来函数或
数值错误的边界：周末事件、无符号计数相减和交易日收益对齐。
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "因子回测"
    / "涨跌停情绪因子"
)
sys.path.insert(0, str(MODULE_DIR))

from sentiment_timing_analysis import (  # noqa: E402
    analyze_threshold_effectiveness,
    analyze_factor_effectiveness,
    build_market_forward_returns,
    build_weekly_basic_factors,
)


def _row(
    code: str,
    trading_date: date,
    *,
    close: float = 10.0,
    pre_close: float = 10.0,
    pct: float = 0.0,
    limit_up: float = 11.0,
    limit_down: float = 9.0,
    total_mv: float = 100.0,
) -> dict:
    """构造一个可交易 A 股日线样本。"""
    return {
        "code": code,
        "trading_date": trading_date,
        "close": close,
        "pre_close": pre_close,
        "pct": pct,
        "limit_up": limit_up,
        "limit_down": limit_down,
        "is_st": False,
        "is_suspended": False,
        "total_mv": total_mv,
    }


def test_net_limit_ratio_preserves_negative_value_when_limit_downs_dominate() -> None:
    """跌停数多于涨停数时，净涨停占比必须为有限负值而非无符号下溢。"""
    monday = date(2025, 1, 6)
    tuesday = date(2025, 1, 7)
    raw = pl.DataFrame(
        [
            _row("SHSE.600001", monday, close=9.0, pct=-10.0),
            _row("SHSE.600002", monday, close=9.0, pct=-10.0),
            _row("SHSE.600001", tuesday, close=9.1, pre_close=9.0, pct=1.11),
            _row("SHSE.600002", tuesday, close=9.1, pre_close=9.0, pct=1.11),
        ]
    )

    result = build_weekly_basic_factors(raw)

    assert result.height == 1
    assert result["net_limit_ratio"][0] == -1.0
    assert np.isfinite(result["net_limit_ratio"][0])


def test_last_trading_day_event_is_excluded_but_earlier_event_is_kept() -> None:
    """短周最后交易日的次日收益属于下周，不能提前进入本周因子。"""
    monday = date(2025, 1, 6)
    tuesday = date(2025, 1, 7)
    wednesday = date(2025, 1, 8)
    next_monday = date(2025, 1, 13)
    raw = pl.DataFrame(
        [
            _row("SHSE.600001", monday, close=11.0, pre_close=10.0, pct=10.0),
            _row("SHSE.600001", tuesday, close=10.1, pre_close=10.0, pct=1.0),
            _row("SHSE.600001", wednesday),
            _row("SHSE.600002", monday),
            _row("SHSE.600002", tuesday),
            _row("SHSE.600002", wednesday, close=11.0, pre_close=10.0, pct=10.0),
            _row("SHSE.600002", next_monday, close=10.2, pre_close=10.0, pct=2.0),
        ]
    )

    result = build_weekly_basic_factors(raw).filter(pl.col("week_end_date") == wednesday)

    assert result["limit_up_event_count"][0] == 1
    assert result["limit_up_next_ret"][0] == 0.01


def test_missing_next_market_day_does_not_turn_into_multiday_stock_return() -> None:
    """停牌缺行时，不能把两日后的股票记录误当作“次日收益”。"""
    monday = date(2025, 1, 6)
    tuesday = date(2025, 1, 7)
    wednesday = date(2025, 1, 8)
    raw = pl.DataFrame(
        [
            _row("SHSE.600001", monday, close=11.0, pre_close=10.0, pct=10.0),
            # 该股周二无记录；周三重新出现不能作为周一的次日收益。
            _row("SHSE.600001", wednesday, close=10.2, pre_close=10.0, pct=2.0),
            _row("SHSE.600002", monday),
            _row("SHSE.600002", tuesday),
            _row("SHSE.600002", wednesday),
        ]
    )

    result = build_weekly_basic_factors(raw)

    assert result["limit_up_event_count"][0] == 0
    assert result["limit_up_next_ret"][0] is None


def test_market_forward_returns_use_trading_steps_not_calendar_days() -> None:
    """未来收益 h 日应定位到第 h 个后续交易日，即使中间有周末。"""
    trading_days = [
        date(2025, 1, 6),
        date(2025, 1, 7),
        date(2025, 1, 9),
        date(2025, 1, 10),
        date(2025, 1, 13),
        date(2025, 1, 14),
        date(2025, 1, 15),
        date(2025, 1, 16),
        date(2025, 1, 17),
        date(2025, 1, 20),
        date(2025, 1, 21),
    ]
    raw = pl.DataFrame(
        [_row("SHSE.600001", day, pct=1.0) for day in trading_days]
    )

    result = build_market_forward_returns(raw)
    first = result.row(0, named=True)

    for horizon in (1, 3, 5, 10):
        assert first[f"future_return_{horizon}d"] == pytest.approx(1.01**horizon - 1)


def test_bearish_factor_direction_makes_g5_minus_g1_positive() -> None:
    """跌停占比这类反向因子，方向调整后 G5 必须代表更强的看多信号。"""
    weekly = pl.DataFrame(
        {
            "fear": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "future_return_1d": [-0.10, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01],
        }
    )

    summary, grouped = analyze_factor_effectiveness(
        weekly,
        factor_columns=["fear"],
        horizons=(1,),
        factor_directions={"fear": -1},
        n_groups=5,
    )

    assert summary.loc[0, "q5_minus_q1"] > 0
    assert grouped.query("group == 'G5'")["mean_return"].iloc[0] > grouped.query(
        "group == 'G1'"
    )["mean_return"].iloc[0]


def test_threshold_effectiveness_compares_triggered_and_untriggered_samples() -> None:
    """研报阈值信号应直接给出触发/未触发的未来收益差。"""
    weekly = pl.DataFrame(
        {
            "optimism": [0.0, 0.0, 1.0, 1.0],
            "future_return_1d": [-0.02, -0.01, 0.02, 0.03],
        }
    )

    result = analyze_threshold_effectiveness(
        weekly,
        thresholds={"optimism": ("gt", 0.5)},
        horizons=(1,),
    )

    assert result.loc[0, "trigger_count"] == 2
    assert result.loc[0, "trigger_mean_return"] == pytest.approx(0.025)
    assert result.loc[0, "not_trigger_mean_return"] == pytest.approx(-0.015)
    assert result.loc[0, "mean_diff"] == pytest.approx(0.04)


def test_zero_market_cap_rows_do_not_pollute_market_return() -> None:
    """零市值的混入记录必须被市场收益聚合排除。"""
    monday = date(2025, 1, 6)
    tuesday = date(2025, 1, 7)
    raw = pl.DataFrame(
        [
            _row("SHSE.600001", monday, pct=1.0, total_mv=100.0),
            _row("SHSE.000001", monday, pct=100.0, total_mv=0.0),
            _row("SHSE.600001", tuesday, pct=1.0, total_mv=100.0),
            _row("SHSE.000001", tuesday, pct=-100.0, total_mv=0.0),
        ]
    )

    result = build_market_forward_returns(raw, horizons=(1,))

    assert result["market_daily_ret"].to_list() == [0.01, 0.01]
