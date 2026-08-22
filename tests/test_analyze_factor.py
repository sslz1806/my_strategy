from __future__ import annotations

import inspect
from datetime import date, timedelta
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import pytest

matplotlib.use("Agg")

from 因子回测.alpha import analyze_factor
from 因子回测.alpha_191.calculator import Alpha191Calculator


def make_panel(include_benchmark: bool = True) -> pl.DataFrame:
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(7)]
    rows = []
    for date_index, trading_date in enumerate(dates):
        ascending = date_index % 2 == 0
        for code_index, code in enumerate(["A", "B", "C", "D"]):
            row = {
                "trading_date": trading_date,
                "code": code,
                "factor": float(code_index if ascending else 3 - code_index),
                "daily_ret": [0.01, 0.03, 0.10, 0.20][code_index],
            }
            if include_benchmark:
                row["benchmark_ret"] = 0.005
            rows.append(row)
    return pl.DataFrame(rows)


def test_ret_window_holds_rebalance_group_and_uses_each_daily_return_once():
    result = analyze_factor(
        make_panel(),
        factor_col="factor",
        ret_windows=(1, 3),
        ic_windows=(1,),
        group_num=2,
        plot=False,
    )

    window_3_g1 = (
        result["group_returns"]
        .filter((pl.col("window") == 3) & (pl.col("group") == "G1"))
        .sort("trading_date")
    )
    assert np.allclose(
        window_3_g1["return"].to_numpy(),
        [0.02, 0.02, 0.02, 0.15, 0.15, 0.15],
    )
    assert window_3_g1["trading_date"].to_list() == [
        date(2024, 1, 3) + timedelta(days=i) for i in range(6)
    ]

    window_1_g1 = (
        result["group_returns"]
        .filter((pl.col("window") == 1) & (pl.col("group") == "G1"))
        .sort("trading_date")
    )
    assert np.allclose(
        window_1_g1["return"].to_numpy(),
        [0.02, 0.15, 0.02, 0.15, 0.02, 0.15],
    )
    assert np.allclose(
        window_3_g1["nav"].to_numpy(),
        np.cumprod([1.02, 1.02, 1.02, 1.15, 1.15, 1.15]),
    )


def test_ic_windows_use_future_returns_and_build_cumulative_ic():
    data = make_panel(include_benchmark=False).with_columns(
        pl.col("code").replace_strict(
            {"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04},
            return_dtype=pl.Float64,
        ).alias("daily_ret"),
        pl.col("code").replace_strict(
            {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
            return_dtype=pl.Float64,
        ).alias("factor"),
    )

    result = analyze_factor(
        data,
        factor_col="factor",
        ret_windows=(3,),
        ic_windows=(1, 2),
        group_num=2,
        plot=False,
    )

    ic_1 = result["ic"].filter(pl.col("window") == 1).sort("trading_date")
    assert np.allclose(ic_1["ic"].to_numpy(), np.ones(6))
    assert np.allclose(ic_1["rank_ic"].to_numpy(), np.ones(6))
    assert np.allclose(ic_1["cum_ic"].to_numpy(), np.arange(1.0, 7.0))
    assert np.allclose(ic_1["cum_rank_ic"].to_numpy(), np.arange(1.0, 7.0))

    ic_2 = result["ic"].filter(pl.col("window") == 2).sort("trading_date")
    assert ic_2.height == 5
    assert np.allclose(ic_2["rank_ic"].to_numpy(), np.ones(5))
    assert np.allclose(ic_2["cum_rank_ic"].to_numpy(), np.arange(1.0, 6.0))

    stats_1 = result["ic_stats"].filter(pl.col("window") == 1).row(0, named=True)
    assert stats_1["ic_ir"] is None or np.isnan(stats_1["ic_ir"])
    assert stats_1["rank_ic_ir"] is None or np.isnan(stats_1["rank_ic_ir"])


def test_factor_autocorr_uses_previous_day_rank_correlation():
    """因子排序每日完全反转时，平均一阶自相关应为 -1。"""
    result = analyze_factor(
        make_panel(include_benchmark=False),
        factor_col="factor",
        ret_windows=(1,),
        ic_windows=(1,),
        group_num=2,
        plot=False,
    )

    assert np.isclose(result["factor_autocorr"], -1.0)


def test_ic_figure_labels_mean_ic_and_rank_ic():
    data = make_panel(include_benchmark=False).with_columns(
        pl.col("code").replace_strict(
            {"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04},
            return_dtype=pl.Float64,
        ).alias("daily_ret"),
        pl.col("code").replace_strict(
            {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
            return_dtype=pl.Float64,
        ).alias("factor"),
    )

    result = analyze_factor(
        data,
        factor_col="factor",
        ret_windows=(1,),
        ic_windows=(1,),
        ic_rolling_window=2,
        group_num=2,
        plot=True,
    )

    lines = {
        line.get_label(): line
        for line in result["figures"]["ic_series"].axes[0].lines
    }
    labels = set(lines)
    assert "IC均值: 1.0000" in labels
    assert "RankIC均值: 1.0000" in labels
    assert "IC 2日滚动" in labels
    assert "RankIC 2日滚动" in labels
    assert lines["IC"].get_alpha() == 0.3
    assert lines["RankIC"].get_alpha() == 0.3
    assert lines["IC 2日滚动"].get_linewidth() == 2
    assert lines["RankIC 2日滚动"].get_linewidth() == 2

    import matplotlib.pyplot as plt

    plt.close("all")


def test_plot_returns_framework_figures_when_no_valid_ic_exists():
    """常量因子没有截面相关性时，绘图不应因 None 格式化失败。"""
    data = make_panel(include_benchmark=False).with_columns(pl.lit(1.0).alias("factor"))

    result = analyze_factor(
        data,
        factor_col="factor",
        ret_windows=(1,),
        ic_windows=(1,),
        group_num=2,
        plot=True,
    )

    assert result["ic"].is_empty()
    assert set(result["figures"]) == {"nav", "ic_series", "cumulative_ic"}

    import matplotlib.pyplot as plt

    plt.close("all")


def test_ic_rolling_window_defaults_to_30():
    assert inspect.signature(analyze_factor).parameters["ic_rolling_window"].default == 30


def test_ic_uses_next_period_return_instead_of_same_day_return():
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    rows = []
    for date_index, trading_date in enumerate(dates):
        returns = (
            [0.04, 0.03, 0.02, 0.01]
            if date_index == 0
            else [0.01, 0.02, 0.03, 0.04]
        )
        for code_index, code in enumerate(["A", "B", "C", "D"]):
            rows.append(
                {
                    "trading_date": trading_date,
                    "code": code,
                    "factor": float(code_index + 1),
                    "daily_ret": returns[code_index],
                }
            )

    result = analyze_factor(
        pl.DataFrame(rows),
        factor_col="factor",
        ret_windows=(1,),
        ic_windows=(1,),
        group_num=2,
        plot=False,
    )

    first_ic = result["ic"].sort("trading_date").row(0, named=True)
    assert first_ic["trading_date"] == date(2024, 1, 2)
    assert np.isclose(first_ic["ic"], 1.0)
    assert np.isclose(first_ic["rank_ic"], 1.0)


@pytest.mark.parametrize("missing_return", [None, float("nan")])
def test_rank_ic_ranks_only_pairwise_valid_stocks(missing_return):
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    rows = []
    for date_index, trading_date in enumerate(dates):
        for code_index, code in enumerate(["A", "B", "C", "D"]):
            daily_return = 0.01 * (code_index + 1)
            if date_index == 1 and code == "B":
                daily_return = missing_return
            rows.append(
                {
                    "trading_date": trading_date,
                    "code": code,
                    "factor": float(code_index + 1),
                    "daily_ret": daily_return,
                }
            )

    result = analyze_factor(
        pl.DataFrame(rows),
        factor_col="factor",
        ret_windows=(1,),
        ic_windows=(1,),
        group_num=2,
        plot=False,
    )

    first_ic = result["ic"].sort("trading_date").row(0, named=True)
    assert first_ic["trading_date"] == date(2024, 1, 2)
    assert np.isclose(first_ic["rank_ic"], 1.0)


def test_benchmark_is_optional_and_multi_window_figures_match_windows():
    with_benchmark = analyze_factor(
        make_panel(),
        factor_col="factor",
        ret_windows=(1, 3),
        ic_windows=(1, 2),
        group_num=2,
        plot=True,
    )

    benchmark = with_benchmark["benchmark"].sort("trading_date")
    assert benchmark["trading_date"][0] == date(2024, 1, 3)
    assert np.isclose(benchmark["return"][0], 0.005)
    assert np.isclose(benchmark["nav"][0], 1.005)
    assert set(with_benchmark["figures"]) == {
        "nav",
        "ic_series",
        "cumulative_ic",
    }
    assert all(len(figure.axes) == 2 for figure in with_benchmark["figures"].values())
    assert "Benchmark" in {
        line.get_label()
        for axis in with_benchmark["figures"]["nav"].axes
        for line in axis.lines
    }

    without_benchmark = analyze_factor(
        make_panel(include_benchmark=False),
        factor_col="factor",
        ret_windows=(1,),
        ic_windows=(1,),
        group_num=2,
        plot=True,
    )
    assert without_benchmark["benchmark"] is None
    assert "Benchmark" not in {
        line.get_label()
        for line in without_benchmark["figures"]["nav"].axes[0].lines
    }

    import matplotlib.pyplot as plt

    plt.close("all")


def test_alpha191_legacy_wrapper_explicitly_uses_backup_api():
    dates = pd.date_range("2024-01-02", periods=3)
    factor = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)
    calculator = Alpha191Calculator.__new__(Alpha191Calculator)
    calculator._is_loaded = True
    calculator.data = {"close": factor + 10}
    calculator.compute_df = lambda alpha_num: factor

    with patch("因子回测.alpha.analyze_factor_bak", return_value={"source": "bak"}):
        result = calculator.analyze_factor(5, return_period=1, group_num=1)

    assert result == {"source": "bak", "alpha_num": 5}


def test_group_labels_keep_numeric_order_above_nine_groups():
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    data = pl.DataFrame(
        [
            {
                "trading_date": trading_date,
                "code": f"S{code_index:02d}",
                "factor": float(code_index),
                "daily_ret": code_index / 1000,
            }
            for trading_date in dates
            for code_index in range(20)
        ]
    )

    result = analyze_factor(
        data,
        factor_col="factor",
        ret_windows=(1,),
        ic_windows=(1,),
        group_num=10,
        plot=False,
    )

    assert result["group_stats"]["group"].to_list() == [
        f"G{group}" for group in range(1, 11)
    ]


def test_all_empty_benchmark_column_is_skipped():
    data = make_panel().with_columns(
        pl.lit(float("nan")).alias("benchmark_ret")
    )

    result = analyze_factor(
        data,
        factor_col="factor",
        ret_windows=(1,),
        ic_windows=(1,),
        group_num=2,
        plot=False,
    )

    assert result["benchmark"] is None


def test_max_drawdown_includes_initial_nav_of_one():
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    data = pl.DataFrame(
        [
            {
                "trading_date": trading_date,
                "code": code,
                "factor": float(code_index),
                "daily_ret": -0.10,
            }
            for trading_date in dates
            for code_index, code in enumerate(["A", "B", "C", "D"])
        ]
    )

    result = analyze_factor(
        data,
        factor_col="factor",
        ret_windows=(1,),
        ic_windows=(1,),
        group_num=2,
        plot=False,
    )

    g1 = result["group_stats"].filter(pl.col("group") == "G1").row(0, named=True)
    assert np.isclose(g1["max_drawdown"], -0.19)
    assert g1["sharpe"] is None or np.isnan(g1["sharpe"])
