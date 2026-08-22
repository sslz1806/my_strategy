"""区间阈值研究在“薄 Notebook + 通用回测器”架构下的回归测试。"""

import numpy as np
import pandas as pd
import pytest

from 因子回测.涨跌停情绪因子 import timing_engine


@pytest.fixture
def simple_timing_data() -> pd.DataFrame:
    """构造可独立验证阈值、信号与次日收益对齐的日频表。"""
    return pd.DataFrame(
        {
            "trading_date": pd.date_range("2024-01-02", periods=10, freq="B"),
            "factor_a": [0.1, 0.2, 0.35, 0.4, 0.5, 0.6, 0.7, 0.75, 0.9, 1.0],
            "factor_b": [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "factor_c": [np.nan] * 10,
            "market_daily_ret": [
                0.02, -0.01, 0.02, -0.01, 0.02,
                -0.01, 0.02, -0.01, 0.02, -0.01,
            ],
        }
    )


def make_band_signal(
    data: pd.DataFrame,
    factor: str,
    lower_column: str,
    upper_column: str,
) -> pd.Series:
    """信号在回测器外生成；闭区间内为 1，缺失值或区间外为 0。"""
    valid = (
        data[factor].notna()
        & data[lower_column].notna()
        & data[upper_column].notna()
    )
    return (
        valid
        & data[factor].between(data[lower_column], data[upper_column], inclusive="both")
    ).astype(float)


class TestComputeThresholdBand:
    """双阈值仍由 compute_threshold 的原接口生成。"""

    def test_output_columns(self, simple_timing_data):
        result = timing_engine.compute_threshold(
            simple_timing_data,
            factor_columns=["factor_a", "factor_b"],
            lower_quantile=0.30,
            upper_quantile=0.70,
            min_history=2,
        )

        assert {
            "lower_factor_a", "upper_factor_a",
            "lower_factor_b", "upper_factor_b",
        }.issubset(result.columns)

    def test_lower_less_than_upper(self, simple_timing_data):
        result = timing_engine.compute_threshold(
            simple_timing_data,
            factor_columns=["factor_a"],
            lower_quantile=0.30,
            upper_quantile=0.70,
            min_history=2,
        )
        valid = result.dropna(subset=["lower_factor_a", "upper_factor_a"])

        assert (valid["lower_factor_a"] < valid["upper_factor_a"]).all()

    def test_shift_one_prevents_current_row_leakage(self, simple_timing_data):
        result = timing_engine.compute_threshold(
            simple_timing_data,
            factor_columns=["factor_a"],
            lower_quantile=0.30,
            upper_quantile=0.70,
            min_history=3,
        )
        history = simple_timing_data.loc[:2, "factor_a"]

        assert result.loc[3, "lower_factor_a"] == history.quantile(0.30)
        assert result.loc[3, "upper_factor_a"] == history.quantile(0.70)


class TestExternalBandSignal:
    """Notebook 外层负责信号，run_time_backtest 只负责回测。"""

    @staticmethod
    def add_fixed_band(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["lower_factor_a"] = 0.35
        result["upper_factor_a"] = 0.75
        result["lower_factor_c"] = 0.30
        result["upper_factor_c"] = 0.70
        return result

    def test_closed_band_signal_is_applied_on_next_day(self, simple_timing_data):
        data = self.add_fixed_band(simple_timing_data)
        data["signal"] = make_band_signal(
            data, "factor_a", "lower_factor_a", "upper_factor_a"
        )

        daily, _ = timing_engine.run_time_backtest(
            data,
            signal_column="signal",
            ret_column="market_daily_ret",
            horizon=1,
            anchor_date=data.loc[0, "trading_date"],
        )

        assert data.loc[2, "signal"] == 1.0
        assert data.loc[7, "signal"] == 1.0
        assert daily["position"].tolist() == data["signal"].iloc[:-1].tolist()
        assert daily["trading_date"].tolist() == data["trading_date"].iloc[1:].tolist()

    def test_nan_factor_produces_zero_signal(self, simple_timing_data):
        data = self.add_fixed_band(simple_timing_data)
        data["signal"] = make_band_signal(
            data, "factor_c", "lower_factor_c", "upper_factor_c"
        )

        daily, blocks = timing_engine.run_time_backtest(
            data,
            signal_column="signal",
            ret_column="market_daily_ret",
            horizon=1,
            anchor_date=data.loc[0, "trading_date"],
        )

        assert not data["signal"].any()
        assert not daily["position"].any()
        assert blocks["position"].tolist() == [0.0]

    def test_output_columns_follow_generic_contract(self, simple_timing_data):
        data = self.add_fixed_band(simple_timing_data)
        data["signal"] = make_band_signal(
            data, "factor_a", "lower_factor_a", "upper_factor_a"
        )

        daily, blocks = timing_engine.run_time_backtest(
            data,
            signal_column="signal",
            ret_column="market_daily_ret",
            horizon=1,
            anchor_date=data.loc[0, "trading_date"],
            require_complete_exit=False,
        )

        assert daily.columns.tolist() == [
            "trading_date", "position", "market_daily_ret",
            "strategy_daily_ret", "benchmark_nav", "strategy_nav",
        ]
        assert blocks.columns.tolist() == [
            "block_id", "position", "decision_date", "block_start_date",
            "block_end_date", "block_duration", "benchmark_block_return",
            "strategy_block_return",
        ]


def test_band_sensitivity_is_an_external_parameter_loop(simple_timing_data):
    """敏感性研究不进入核心：外层枚举阈值后复用同一个回测入口。"""
    rows = []
    for lower_quantile in (0.20, 0.30):
        for upper_quantile in (0.70, 0.80):
            data = timing_engine.compute_threshold(
                simple_timing_data,
                factor_columns=["factor_a"],
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                min_history=2,
            )
            data["signal"] = make_band_signal(
                data, "factor_a", "lower_factor_a", "upper_factor_a"
            )
            anchor = data.loc[data["lower_factor_a"].notna(), "trading_date"].iloc[0]
            daily, blocks = timing_engine.run_time_backtest(
                data, "signal", "market_daily_ret", horizon=1, anchor_date=anchor
            )
            summary = timing_engine.summarize_timing(
                daily, blocks, factor="factor_a", horizon=1
            )
            rows.append(
                {
                    "lower_quantile": lower_quantile,
                    "upper_quantile": upper_quantile,
                    **summary,
                }
            )

    result = pd.DataFrame(rows)
    assert len(result) == 4
    assert {
        "factor", "horizon", "lower_quantile", "upper_quantile",
        "holding_ratio", "annual_return", "sharpe",
    }.issubset(result.columns)
