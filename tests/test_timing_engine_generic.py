"""通用单指数择时回测接口的回归测试。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from 因子回测.涨跌停情绪因子 import timing_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "因子回测"
    / "涨跌停情绪因子"
    / "sentiment_factors_5d_research.ipynb"
)


def make_backtest_data() -> pd.DataFrame:
    """构造能独立检查次日对齐与连续信号续期的日频数据。"""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=6, freq="B"),
            "entry": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "hs300_ret": [0.99, 0.10, -0.20, 0.30, -0.10, 0.05],
        }
    )


def test_run_time_backtest_uses_named_columns_and_next_day_returns():
    data = make_backtest_data()

    daily, _ = timing_engine.run_time_backtest(
        data,
        signal_column="entry",
        ret_column="hs300_ret",
        horizon=2,
        date_column="date",
        anchor_date=data.loc[0, "date"],
    )

    assert daily["date"].tolist() == data["date"].iloc[1:].tolist()
    assert daily["hs300_ret"].tolist() == [0.10, -0.20, 0.30, -0.10, 0.05]
    assert daily["position"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert daily["strategy_daily_ret"].tolist() == [0.10, -0.20, 0.30, -0.10, 0.0]
    assert "market_daily_ret" not in daily.columns


def test_run_timing_alias_accepts_the_generic_return_column():
    data = make_backtest_data()

    daily, _ = timing_engine.run_timing(
        data,
        signal_column="entry",
        ret_column="hs300_ret",
        horizon=1,
        date_column="date",
        anchor_date=data.loc[0, "date"],
    )

    assert daily["hs300_ret"].tolist() == [0.10, -0.20, 0.30, -0.10, 0.05]


def test_compute_threshold_keeps_lagged_single_and_band_behaviour():
    data = pd.DataFrame(
        {
            "trading_date": pd.date_range("2024-01-02", periods=5, freq="B"),
            "factor": [1.0, 2.0, 3.0, 100.0, 5.0],
        }
    )

    single = timing_engine.compute_threshold(
        data, factor_columns=["factor"], quantile=0.5, min_history=3,
    )
    band = timing_engine.compute_threshold(
        data, factor_columns=["factor"], lower_quantile=0.25,
        upper_quantile=0.75, min_history=3,
    )

    assert single["threshold_factor"].iloc[:3].isna().all()
    assert single.loc[3, "threshold_factor"] == 2.0
    assert band.loc[3, "lower_factor"] == 1.5
    assert band.loc[3, "upper_factor"] == 2.5


def test_run_time_backtest_block_includes_its_last_return_day():
    data = pd.DataFrame(
        {
            "trading_date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "entry": [1.0, 0.0, 0.0],
            "market_daily_ret": [0.99, 0.10, -0.20],
        }
    )

    _, blocks = timing_engine.run_timing(
        data,
        "entry",
        horizon=1,
        anchor_date=data.loc[0, "trading_date"],
    )

    holding = blocks.loc[blocks["position"] == 1.0].iloc[0]
    assert holding["block_duration"] == 1
    assert holding["block_start_date"] == data.loc[1, "trading_date"]
    assert holding["block_end_date"] == data.loc[1, "trading_date"]
    assert np.isclose(holding["benchmark_block_return"], 0.10)


def test_zero_and_nan_do_not_interrupt_holding_and_incomplete_tail_is_optional():
    data = pd.DataFrame(
        {
            "trading_date": pd.date_range("2024-01-02", periods=5, freq="B"),
            "entry": [1.0, np.nan, 0.0, 1.0, 0.0],
            "benchmark_ret": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )

    daily, complete_blocks = timing_engine.run_time_backtest(
        data, "entry", "benchmark_ret", horizon=2,
        anchor_date=data.loc[0, "trading_date"],
    )
    _, all_blocks = timing_engine.run_time_backtest(
        data, "entry", "benchmark_ret", horizon=2,
        anchor_date=data.loc[0, "trading_date"], require_complete_exit=False,
    )

    assert daily["position"].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert complete_blocks["position"].tolist() == [1.0, 0.0]
    assert all_blocks["position"].tolist() == [1.0, 0.0, 1.0]
    assert all_blocks.iloc[-1]["block_end_date"] == data.iloc[-1]["trading_date"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.drop(columns="hs300_ret"), "hs300_ret"),
        (
            lambda frame: frame.assign(date=[frame.loc[0, "date"]] * len(frame)),
            "重复",
        ),
        (lambda frame: frame.assign(hs300_ret=[0.1, np.nan, 0.2, 0.3, 0.4, 0.5]), "有限"),
        (lambda frame: frame.assign(entry=[1.0, 0.5, 0.0, 0.0, 0.0, 0.0]), "0、1 或 NaN"),
    ],
)
def test_run_time_backtest_rejects_invalid_input(change, message):
    with pytest.raises(ValueError, match=message):
        timing_engine.run_time_backtest(
            change(make_backtest_data()),
            signal_column="entry",
            ret_column="hs300_ret",
            horizon=2,
            date_column="date",
        )


def test_run_time_backtest_rejects_invalid_horizon_and_anchor():
    data = make_backtest_data()

    with pytest.raises(TypeError, match="pandas.DataFrame"):
        timing_engine.run_time_backtest(
            data.to_dict("list"), "entry", "hs300_ret", horizon=1,
            date_column="date",
        )
    with pytest.raises(ValueError, match="horizon"):
        timing_engine.run_time_backtest(
            data, "entry", "hs300_ret", horizon=0, date_column="date"
        )
    with pytest.raises(ValueError, match="anchor_date"):
        timing_engine.run_time_backtest(
            data,
            "entry",
            "hs300_ret",
            horizon=1,
            date_column="date",
            anchor_date="1999-01-01",
        )


def test_run_time_backtest_empty_result_has_stable_columns():
    data = make_backtest_data().assign(entry=np.nan)

    daily, blocks = timing_engine.run_time_backtest(
        data, "entry", "hs300_ret", horizon=2, date_column="date"
    )

    assert daily.empty
    assert daily.columns.tolist() == [
        "date",
        "position",
        "hs300_ret",
        "strategy_daily_ret",
        "benchmark_nav",
        "strategy_nav",
    ]
    assert blocks.empty
    assert blocks.columns.tolist() == [
        "block_id",
        "position",
        "decision_date",
        "block_start_date",
        "block_end_date",
        "block_duration",
        "benchmark_block_return",
        "strategy_block_return",
    ]

    summary = timing_engine.summarize_timing(
        daily,
        blocks,
        factor="test_factor",
        horizon=2,
        ret_column="hs300_ret",
    )
    assert np.isnan(summary["annual_return"])
    assert np.isnan(summary["benchmark_annual_return"])


def test_summarize_timing_uses_the_selected_return_column():
    data = make_backtest_data()
    daily, blocks = timing_engine.run_time_backtest(
        data,
        "entry",
        "hs300_ret",
        horizon=2,
        date_column="date",
        anchor_date=data.loc[0, "date"],
    )

    summary = timing_engine.summarize_timing(
        daily,
        blocks,
        factor="test_factor",
        horizon=2,
        ret_column="hs300_ret",
    )

    assert np.isclose(summary["benchmark_final_nav"], 1.10 * 0.80 * 1.30 * 0.90 * 1.05)
    assert np.isclose(summary["final_nav"], 1.10 * 0.80 * 1.30 * 0.90)


def test_analyze_ic_derives_targets_from_the_selected_return_column():
    data = pd.DataFrame(
        {
            "factor": [1.0, 2.0, 3.0, 4.0],
            "future_hs300_ret_1d": [4.0, 3.0, 2.0, 1.0],
        }
    )

    result = timing_engine.analyze_ic(
        data,
        factor_columns=["factor"],
        horizons=(1,),
        factor_directions={"factor": -1},
        ret_column="hs300_ret",
    )

    assert result.loc[0, "n_obs"] == 4
    assert np.isclose(result.loc[0, "pearson_ic"], -1.0)
    assert np.isclose(result.loc[0, "directional_pearson_ic"], 1.0)


def test_notebook_plot_timing_nav_comparison_uses_custom_labels(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {}
    exec("".join(notebook["cells"][1]["source"]), namespace)
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    daily_results = {
        ("factor", 1): pd.DataFrame(
            {
                "trading_date": dates,
                "benchmark_nav": [1.0, 1.1],
                "strategy_nav": [1.0, 1.05],
                "position": [1.0, 1.0],
            }
        )
    }

    namespace["plot_timing_nav_comparison"](
        daily_results,
        factor_columns=["factor"],
        horizons=(1,),
        factor_labels={"factor": "测试因子"},
        benchmark_label="沪深300",
    )

    figure = plt.gcf()
    assert figure._suptitle.get_text() == "测试因子：单因子择时净值与基准"
    assert figure.axes[0].get_legend_handles_labels()[1] == ["沪深300", "单因子择时"]
    plt.close(figure)


def test_notebook_plot_rolling_ic_history_uses_precomputed_results(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {}
    exec("".join(notebook["cells"][1]["source"]), namespace)
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    rolling_detail = pd.DataFrame(
        {
            "factor": ["factor"] * 3,
            "horizon": [1] * 3,
            "window": [2] * 3,
            "factor_window_end_date": dates,
            "available_date": dates,
            "rolling_ic": [0.1, 0.2, 0.3],
        }
    )
    full_ic_summary = pd.DataFrame(
        {"factor": ["factor"], "horizon": [1], "pearson_ic": [0.2]}
    )

    namespace["plot_rolling_ic_history"](
        rolling_detail,
        full_ic_summary,
        factor_columns=["factor"],
        horizons=(1,),
        windows=(2,),
    )

    assert plt.gcf()._suptitle.get_text() == "factor：滚动时序 IC"
    plt.close(plt.gcf())


def test_notebook_ic_cell_computes_and_stores_rolling_results(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {}
    exec("".join(notebook["cells"][1]["source"]), namespace)
    namespace.update(
        HORIZONS=(1,),
        FACTOR_COLUMNS=["factor"],
        FACTOR_DIRECTIONS={"factor": 1},
        ROLLING_IC_WINDOWS=(3,),
        ROLLING_IC_MIN_VALID_RATIO=0.5,
        BENCHMARK_LABELS={"test": "测试指数"},
        benchmark_data={
            "test": pd.DataFrame(
                {
                    "trading_date": pd.date_range("2024-01-02", periods=8, freq="B"),
                    "benchmark_ret": [0.01, -0.01, 0.02, 0.01, -0.02, 0.03, 0.01, 0.02],
                    "factor": np.arange(8, dtype=float),
                }
            )
        },
    )

    exec("".join(notebook["cells"][7]["source"]), namespace)

    assert set(namespace["rolling_ic"]) == {"test"}
    assert not namespace["rolling_ic"]["test"].empty


def test_notebook_prepares_rq_multiindex_returns_without_network():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    setup_source = "".join(notebook["cells"][1]["source"])
    namespace = {}
    exec(setup_source, namespace)

    assert "prepare_rq_benchmark_returns" in namespace
    rq_index = pd.MultiIndex.from_tuples(
        [
            ("000300.XSHG", pd.Timestamp("2024-01-02")),
            ("000300.XSHG", pd.Timestamp("2024-01-03")),
            ("399006.XSHE", pd.Timestamp("2024-01-02")),
            ("399006.XSHE", pd.Timestamp("2024-01-03")),
        ],
        names=["order_book_id", "date"],
    )
    rq_returns = pd.DataFrame({"return": [0.01, 0.02, -0.01, 0.03]}, index=rq_index)

    actual = namespace["prepare_rq_benchmark_returns"](
        rq_returns,
        {"hs300": "000300.XSHG", "cyb": "399006.XSHE"},
    )

    assert list(actual) == ["hs300", "cyb"]
    assert actual["hs300"].columns.tolist() == ["trading_date", "benchmark_ret"]
    assert actual["hs300"]["benchmark_ret"].tolist() == [0.01, 0.02]
    assert actual["cyb"]["benchmark_ret"].tolist() == [-0.01, 0.03]


def test_notebook_common_range_and_five_benchmark_loop_without_network():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {}
    exec("".join(notebook["cells"][1]["source"]), namespace)

    codes = namespace["BENCHMARK_CODES"]
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    rows = []
    for offset, code in enumerate(codes.values()):
        for trading_date in dates[offset % 2 : 5 + offset % 2]:
            rows.append((code, trading_date, 0.001 * (offset + 1)))
    rq_returns = pd.DataFrame(
        {"return": [row[2] for row in rows]},
        index=pd.MultiIndex.from_tuples(
            [(row[0], row[1]) for row in rows],
            names=["order_book_id", "date"],
        ),
    )

    prepared = namespace["prepare_rq_benchmark_returns"](rq_returns, codes)
    trimmed, common_start, common_end = namespace[
        "trim_benchmark_returns_to_common_range"
    ](prepared)

    assert common_start == dates[1]
    assert common_end == dates[4]
    assert set(trimmed) == set(codes)
    assert all(
        frame["trading_date"].between(common_start, common_end).all()
        for frame in trimmed.values()
    )

    signals = pd.DataFrame(
        {
            "trading_date": pd.date_range(common_start, common_end, freq="B"),
            "signal_factor": [1.0, 0.0, 0.0, 0.0],
        }
    )
    details = {}
    for benchmark, returns in trimmed.items():
        data = signals.merge(returns, on="trading_date", validate="one_to_one")
        details[(benchmark, "factor", 1)] = timing_engine.run_time_backtest(
            data, "signal_factor", "benchmark_ret", horizon=1,
            anchor_date=common_start,
        )[0]

    assert len(details) == 5
    assert all(not detail.empty for detail in details.values())
