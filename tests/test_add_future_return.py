from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from 因子回测.alpha import add_future_return


def make_pd_series():
    return pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "pct": [0.10, 0.20, -0.10, 0.25],
        }
    )


def make_pl_series():
    return pl.DataFrame(
        {
            "trading_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
            "pct": [0.10, 0.20, -0.10, 0.25],
        }
    )


def make_pd_panel():
    return pd.DataFrame(
        {
            "code": ["A", "A", "A", "B", "B", "B"],
            "trading_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "pct": [0.10, 0.20, -0.10, 0.05, 0.15, -0.05],
        }
    )


def make_pl_panel():
    return pl.DataFrame(
        {
            "code": ["A", "A", "A", "B", "B", "B"],
            "trading_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)] * 2,
            "pct": [0.10, 0.20, -0.10, 0.05, 0.15, -0.05],
        }
    )


def test_pandas_time_series():
    df = make_pd_series()
    result = add_future_return(df, horizons=(1, 3))

    assert np.isclose(result.loc[0, "future_pct_1d"], 0.20)
    assert np.isclose(
        result.loc[0, "future_pct_3d"],
        (1.20 * 0.90 * 1.25) - 1.0,
    )
    assert pd.isna(result.loc[3, "future_pct_1d"])
    assert pd.isna(result.loc[2, "future_pct_3d"])


def test_polars_time_series():
    df = make_pl_series()
    result = add_future_return(df, horizons=(1, 3))

    assert np.isclose(result["future_pct_1d"][0], 0.20)
    assert np.isclose(
        result["future_pct_3d"][0],
        (1.20 * 0.90 * 1.25) - 1.0,
    )
    assert result["future_pct_1d"][3] is None
    assert result["future_pct_3d"][2] is None


def test_pandas_panel_groups_by_code():
    df = make_pd_panel()
    result = add_future_return(df, horizons=(1, 2))

    row_a0 = result[(result["code"] == "A") & (result["trading_date"] == "2024-01-02")].iloc[0]
    row_b0 = result[(result["code"] == "B") & (result["trading_date"] == "2024-01-02")].iloc[0]

    assert np.isclose(row_a0["future_pct_1d"], 0.20)
    assert np.isclose(row_b0["future_pct_1d"], 0.15)
    assert np.isclose(row_a0["future_pct_2d"], (1.20 * 0.90) - 1.0)
    assert np.isclose(row_b0["future_pct_2d"], (1.15 * 0.95) - 1.0)


def test_polars_panel_groups_by_code():
    df = make_pl_panel()
    result = add_future_return(df, horizons=(1, 2))

    a0 = result.filter((pl.col("code") == "A") & (pl.col("trading_date") == date(2024, 1, 2))).row(0, named=True)
    b0 = result.filter((pl.col("code") == "B") & (pl.col("trading_date") == date(2024, 1, 2))).row(0, named=True)

    assert np.isclose(a0["future_pct_1d"], 0.20)
    assert np.isclose(b0["future_pct_1d"], 0.15)
    assert np.isclose(a0["future_pct_2d"], (1.20 * 0.90) - 1.0)
    assert np.isclose(b0["future_pct_2d"], (1.15 * 0.95) - 1.0)


def test_ret_col_fallback_to_price():
    df = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "close": [11.0, 12.0, 10.8],
            "pre_close": [10.0, 11.0, 12.0],
        }
    )
    result = add_future_return(df, ret_col="pct", horizons=(1, 2))

    # pct 列不存在，应回退到 close / pre_close - 1
    assert np.isclose(result.loc[0, "future_pct_1d"], 12.0 / 11.0 - 1.0)
    assert np.isclose(result.loc[0, "future_pct_2d"], (12.0 / 11.0) * (10.8 / 12.0) - 1.0)


def test_output_column_names_use_ret_col():
    df = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "market_daily_ret": [0.01, 0.02],
        }
    )
    result = add_future_return(df, ret_col="market_daily_ret", horizons=(1,))

    assert "future_market_daily_ret_1d" in result.columns
