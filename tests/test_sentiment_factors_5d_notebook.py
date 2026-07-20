from datetime import date
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "因子回测"
    / "涨跌停情绪因子"
    / "sentiment_factors_5d_research.ipynb"
)


def load_notebook_definitions():
    """只执行带 definitions 标签的单元，避免测试时读取真实行情。"""
    assert NOTEBOOK_PATH.exists(), f"Notebook 尚未创建：{NOTEBOOK_PATH}"
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and "definitions" in cell.metadata.get("tags", []):
            exec(compile(cell.source, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def build_synthetic_daily_data() -> pl.DataFrame:
    """构造六个市场交易日、三只股票，其中一只股票存在跨日缺行。"""
    trading_dates = [date(2024, 1, day) for day in (2, 3, 4, 5, 8, 9)]
    rows = []

    for index, trading_date in enumerate(trading_dates):
        close_a = 10.0 + index
        rows.append(
            {
                "code": "XSHG.600001",
                "trading_date": trading_date,
                "close": close_a,
                "pre_close": close_a - 1.0 if index else 9.0,
                "limit_up": close_a if index < 5 else close_a + 1.0,
                "limit_down": close_a - 2.0,
                "is_st": False,
                "is_suspended": False,
                "total_mv": 100.0 + index,
            }
        )

        close_b = 20.0
        rows.append(
            {
                "code": "XSHE.000002",
                "trading_date": trading_date,
                "close": close_b,
                "pre_close": close_b,
                "limit_up": 22.0,
                "limit_down": 18.0,
                "is_st": False,
                "is_suspended": False,
                "total_mv": 200.0,
            }
        )

    # 该股票在 1 月 3 日封涨停，但下一条记录跨过 1 月 4 日，不能把跨日收益当成“次日收益”。
    for trading_date, close, limit_up in [
        (trading_dates[0], 30.0, 33.0),
        (trading_dates[1], 33.0, 33.0),
        (trading_dates[3], 35.0, 38.5),
    ]:
        rows.append(
            {
                "code": "XSHG.600003",
                "trading_date": trading_date,
                "close": close,
                "pre_close": 30.0,
                "limit_up": limit_up,
                "limit_down": 27.0,
                "is_st": False,
                "is_suspended": False,
                "total_mv": 50.0,
            }
        )

    return pl.DataFrame(rows).sort(["code", "trading_date"])


def test_five_day_factors_count_repeated_events_and_avoid_division_by_zero():
    namespace = load_notebook_definitions()
    prepared, calendar = namespace["prepare_stock_daily"](build_synthetic_daily_data())
    factors = namespace["build_daily_sentiment_factors"](prepared, calendar, window=5)

    fifth_day = factors.row(4, named=True)
    assert fifth_day["limit_up_count_5d"] == 6
    assert fifth_day["limit_down_count_5d"] == 0
    assert fifth_day["max_eligible_stock_count_5d"] == 3
    assert fifth_day["limit_up_ratio"] == 2.0
    assert fifth_day["limit_up_down_ratio"] == 6.0


def test_next_day_event_return_rejects_stock_record_that_skips_market_day():
    namespace = load_notebook_definitions()
    prepared, _ = namespace["prepare_stock_daily"](build_synthetic_daily_data())
    skipped_event = prepared.filter(
        (pl.col("code") == "XSHG.600003")
        & (pl.col("trading_date") == date(2024, 1, 3))
    ).row(0, named=True)

    assert skipped_event["is_limit_up"]
    assert skipped_event["event_next_ret"] is None


def test_forward_returns_start_after_factor_date():
    namespace = load_notebook_definitions()
    market = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "market_daily_ret": [0.50, 0.10, -0.20, 0.25],
        }
    )
    result = namespace["add_forward_returns"](market, horizons=(1, 3))

    assert np.isclose(result.loc[0, "future_return_1d"], 0.10)
    assert np.isclose(result.loc[0, "future_return_3d"], 1.10 * 0.80 * 1.25 - 1)


def test_ic_reports_raw_and_direction_adjusted_correlation():
    namespace = load_notebook_definitions()
    factor = np.linspace(-1.0, 1.0, 80)
    research = pd.DataFrame(
        {
            "test_factor": factor,
            "future_return_1d": factor * 0.02,
        }
    )
    summary = namespace["analyze_ic"](
        research,
        factor_columns=["test_factor"],
        horizons=(1,),
        factor_directions={"test_factor": -1},
    )

    assert len(summary) == 1
    assert np.isclose(summary.loc[0, "pearson_ic"], 1.0)
    assert np.isclose(summary.loc[0, "spearman_ic"], 1.0)
    assert np.isclose(summary.loc[0, "directional_pearson_ic"], -1.0)
    assert summary.loc[0, "n_obs"] == 80


def test_expanding_threshold_uses_only_prior_observations():
    namespace = load_notebook_definitions()
    research = pd.DataFrame(
        {
            "trading_date": pd.date_range("2024-01-01", periods=6, freq="B"),
            "test_factor": [1.0, 2.0, 3.0, 100.0, 5.0, 6.0],
        }
    )
    result = namespace["build_expanding_thresholds"](
        research,
        factor_columns=["test_factor"],
        quantile=0.8,
        min_history=3,
    )

    expected = pd.Series([1.0, 2.0, 3.0]).quantile(0.8)
    assert result["threshold_test_factor"].first_valid_index() == 3
    assert np.isclose(result.loc[3, "threshold_test_factor"], expected)


def test_non_overlapping_timing_applies_signal_only_to_following_block():
    namespace = load_notebook_definitions()
    research = pd.DataFrame(
        {
            "trading_date": pd.date_range("2024-01-01", periods=7, freq="B"),
            "market_daily_ret": [0.90, 0.10, 0.10, -0.10, -0.10, 0.05, 0.05],
            "test_factor": [10.0, 8.0, 1.0, 2.0, 10.0, 9.0, 8.0],
            "threshold_test_factor": [5.0] * 7,
        }
    )
    daily, blocks = namespace["run_non_overlapping_timing"](
        research,
        factor="test_factor",
        horizon=2,
        anchor_date=research.loc[0, "trading_date"],
        direction=1,
    )
    summary = namespace["summarize_timing"](
        daily,
        blocks,
        factor="test_factor",
        horizon=2,
    )

    assert len(blocks) == 3
    assert blocks["position"].tolist() == [1.0, 0.0, 1.0]
    assert np.isclose(daily.iloc[0]["market_daily_ret"], 0.10)
    assert np.isclose(daily.iloc[0]["strategy_daily_ret"], 0.10)
    assert np.isclose(summary["holding_win_rate"], 1.0)
    assert np.isclose(summary["timing_hit_rate"], 1.0)
