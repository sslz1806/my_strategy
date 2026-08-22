from __future__ import annotations

import sys
import types
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import pytest

from 因子回测.alpha import analyze_factor
from 因子回测.alpha_191.alpha_formulas import Alpha191Formulas
from 因子回测.alpha_191 import batch_backtest


matplotlib.use("Agg")


def _make_import_safe_batch_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """以隔离依赖执行当前脚本，避免红灯阶段触发真实数据读取或邮件发送。"""
    dates = pd.date_range("2024-01-02", periods=12, freq="B")
    codes = [f"S{index:03d}" for index in range(30)]
    close = pd.DataFrame(
        {
            code: 10 + np.arange(len(dates)) * (1 + code_index / 1000)
            for code_index, code in enumerate(codes)
        },
        index=dates,
    )
    factor = pd.DataFrame(
        {
            code: np.arange(len(dates), dtype=float) + code_index
            for code_index, code in enumerate(codes)
        },
        index=dates,
    )

    adapter_module = types.ModuleType("因子回测.alpha_191.adapter")
    adapter_module.load_factor_data = lambda *args, **kwargs: {"close": close.copy()}
    adapter_module.load_factor_data_with_industry = lambda *args, **kwargs: {"close": close.copy()}

    class FakeFormulas:
        def __init__(self, data, industry_map=None):
            self.data = data

        def __getattr__(self, name):
            if name.startswith("alpha_") and name.endswith("_df"):
                return lambda: factor.copy()
            raise AttributeError(name)

    formulas_module = types.ModuleType("因子回测.alpha_191.alpha_formulas")
    formulas_module.Alpha191Formulas = FakeFormulas

    email_module = types.ModuleType("my_utils.email_fun")
    email_module.send_email = lambda **kwargs: None

    monkeypatch.setitem(sys.modules, "因子回测.alpha_191.adapter", adapter_module)
    monkeypatch.setitem(sys.modules, "因子回测.alpha_191.alpha_formulas", formulas_module)
    monkeypatch.setitem(sys.modules, "my_utils.email_fun", email_module)

    source_path = Path("因子回测/alpha_191/batch_backtest.py")
    module = types.ModuleType("alpha191_batch_backtest_under_test")
    module.__file__ = str(tmp_path / "batch_backtest.py")
    exec(compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec"), module.__dict__)
    return module


@pytest.fixture
def batch_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    return _make_import_safe_batch_module(monkeypatch, tmp_path)


def make_panel() -> pl.DataFrame:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(8)]
    return pl.DataFrame(
        [
            {
                "trading_date": trading_date,
                "code": code,
                "factor": float(code_index + 1),
                "daily_ret": 0.005 * (code_index + 1),
            }
            for trading_date in dates
            for code_index, code in enumerate(["A", "B", "C", "D"])
        ]
    )


def test_factor_panel_keeps_factor_and_daily_return_aligned_by_date_and_code(batch_module):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    factor = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, index=dates)
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]}, index=dates)

    panel = batch_module.build_factor_panel(factor, returns)

    assert panel.columns == ["trading_date", "code", "factor", "daily_ret"]
    assert panel.sort("trading_date", "code").to_dicts() == [
        {"trading_date": date(2024, 1, 2), "code": "A", "factor": 1.0, "daily_ret": 0.01},
        {"trading_date": date(2024, 1, 2), "code": "B", "factor": 3.0, "daily_ret": 0.03},
        {"trading_date": date(2024, 1, 3), "code": "A", "factor": 2.0, "daily_ret": 0.02},
        {"trading_date": date(2024, 1, 3), "code": "B", "factor": 4.0, "daily_ret": 0.04},
    ]


def test_single_alpha_record_uses_rank_ic_returned_by_analyze_factor(batch_module):
    panel = make_panel()
    factor = (
        panel.pivot(on="code", index="trading_date", values="factor")
        .to_pandas()
        .set_index("trading_date")
    )
    returns = (
        panel.pivot(on="code", index="trading_date", values="daily_ret")
        .to_pandas()
        .set_index("trading_date")
    )

    record = batch_module.run_single_alpha(
        alpha_num=1,
        factor_wide=factor,
        daily_returns=returns,
        analyze=analyze_factor,
        return_period=1,
        group_num=2,
    )

    expected = analyze_factor(
        panel,
        factor_col="factor",
        ret_col="daily_ret",
        ret_windows=(1,),
        ic_windows=(1,),
        group_num=2,
        plot=False,
    )["ic_stats"].row(0, named=True)["rank_ic_mean"]
    assert record["status"] == "ok"
    assert record["rank_ic_mean"] == pytest.approx(expected)
    assert set(record["analysis"]["figures"]) == {"nav", "ic_series", "cumulative_ic"}


def test_html_report_orders_successes_by_rank_ic_and_marks_unavailable_factor(batch_module):
    analysis = analyze_factor(
        make_panel(),
        factor_col="factor",
        ret_col="daily_ret",
        ret_windows=(1,),
        ic_windows=(1,),
        group_num=2,
        plot=True,
    )
    records = [
        {
            "alpha": 1,
            "status": "unavailable",
            "reason": "本地与米筐均未取得所需字段",
            "rank_ic_mean": None,
            "analysis": None,
        },
        {
            "alpha": 2,
            "status": "ok",
            "rank_ic_mean": 0.20,
            "analysis": analysis,
            "elapsed_seconds": 1.2,
        },
        {
            "alpha": 3,
            "status": "ok",
            "rank_ic_mean": 0.10,
            "analysis": analysis,
            "elapsed_seconds": 1.3,
        },
    ]

    report = batch_module.render_html_report(
        records=records,
        metadata={"start_date": "2024-01-02", "end_date": "2024-01-09", "stock_count": 4},
    )

    assert report.index("id=\"alpha-002\"") < report.index("id=\"alpha-003\"")
    assert report.index("id=\"alpha-003\"") < report.index("id=\"alpha-001\"")
    assert "data:image/png;base64," in report
    assert "IC / RankIC 统计" in report
    assert "分组回测统计" in report
    assert "本地与米筐均未取得所需字段" in report


def test_empty_industry_map_is_not_treated_as_successful_industry_neutralization():
    dates = pd.date_range("2024-01-02", periods=2)
    values = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, index=dates)
    formulas = Alpha191Formulas(
        {
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": values,
            "vwap": values,
            "returns": values,
        },
        industry_map={},
    )

    with pytest.raises(ValueError, match="缺少行业分类数据"):
        formulas.ind_neutralize(values)


def test_batch_marks_missing_industry_input_as_unavailable(monkeypatch: pytest.MonkeyPatch):
    class MissingIndustryFormula:
        def __init__(self, data, industry_map=None):
            pass

        def alpha_001_df(self):
            raise ValueError("缺少行业分类数据（industry_map），无法进行行业中性化")

    monkeypatch.setattr(batch_backtest, "Alpha191Formulas", MissingIndustryFormula)
    dates = pd.date_range("2024-01-02", periods=3)
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)

    record = batch_backtest.run_batch_backtest(
        {"returns": returns, "industry": {}}, alpha_numbers=[1]
    )[0]

    assert record["status"] == "unavailable"
    assert "缺少行业分类数据" in record["reason"]
