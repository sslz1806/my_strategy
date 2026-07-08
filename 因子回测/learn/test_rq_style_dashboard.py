import pandas as pd
import matplotlib
import json
from pathlib import Path

matplotlib.use("Agg")

from rq_style_dashboard import (
    FACTOR_INFO,
    build_factor_direction_table,
    build_style_performance_table,
    calc_cumulative_return,
    generate_market_style_commentary,
    plot_key_style_groups,
    plot_market_temperature,
    plot_style_heatmap,
    plot_style_rank_bar,
    select_explicit_style_factors,
)


def test_select_explicit_style_factors_excludes_comovement_and_chinese_industries():
    columns = [
        "comovement",
        "momentum",
        "beta",
        "residual_volatility",
        "银行",
        "食品饮料",
    ]

    result = select_explicit_style_factors(columns)

    assert result == ["beta", "momentum", "residual_volatility"]


def test_factor_direction_table_contains_investor_interpretation():
    latest_returns = pd.Series(
        {"momentum": 0.03, "residual_volatility": -0.02},
        name="latest",
    )

    table = build_factor_direction_table(
        ["momentum", "residual_volatility"],
        latest_returns=latest_returns,
    )

    assert list(table["因子"]) == ["momentum", "residual_volatility"]
    assert table.loc[0, "正收益代表"] == FACTOR_INFO["momentum"]["positive"]
    assert table.loc[1, "当前方向"] == FACTOR_INFO["residual_volatility"]["negative"]
    assert "低波动" in table.loc[1, "当前方向"]


def test_calc_cumulative_return_uses_tail_window():
    series = pd.Series([0.10, -0.05, 0.02])

    result = calc_cumulative_return(series, window=2)

    assert round(result, 6) == round((1 - 0.05) * (1 + 0.02) - 1, 6)


def test_build_style_performance_table_sorts_by_rank_window_and_explains_direction():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    fr = pd.DataFrame(
        {
            "momentum": [0.01, 0.01, 0.01, 0.01, 0.01],
            "residual_volatility": [-0.01, -0.01, -0.01, -0.01, -0.01],
            "beta": [0.00, 0.00, 0.02, 0.02, 0.02],
        },
        index=dates,
    )

    table = build_style_performance_table(
        fr,
        ["momentum", "residual_volatility", "beta"],
        windows=(2, 3),
        rank_window=3,
    )

    assert list(table.columns) == [
        "中文名",
        "风格组",
        "2日收益%",
        "3日收益%",
        "年初至今收益%",
        "最新5日收益%",
        "当前方向",
        "投资者解读",
        "信号强度",
    ]
    assert table.index[0] == "beta"
    assert "高 beta" in table.loc["beta", "当前方向"]
    assert "低波动" in table.loc["residual_volatility", "当前方向"]


def test_generate_market_style_commentary_mentions_strong_and_weak_factors():
    dates = pd.date_range("2026-01-01", periods=65, freq="D")
    fr = pd.DataFrame(
        {
            "momentum": [0.002] * 65,
            "beta": [0.001] * 65,
            "residual_volatility": [-0.0015] * 65,
            "liquidity": [-0.001] * 65,
        },
        index=dates,
    )
    table = build_style_performance_table(
        fr,
        ["momentum", "beta", "residual_volatility", "liquidity"],
        windows=(20, 60),
        rank_window=60,
    )

    text = generate_market_style_commentary(table)

    assert "当前市场风格" in text
    assert "动量" in text
    assert "低波动" in text
    assert "高换手" in text or "低换手" in text


def test_plotting_functions_return_axes_objects():
    dates = pd.date_range("2026-01-01", periods=130, freq="D")
    fr = pd.DataFrame(
        {
            "momentum": [0.001] * 130,
            "beta": [0.0005] * 130,
            "residual_volatility": [-0.0008] * 130,
            "size": [0.0002] * 130,
            "comovement": [0.0015] * 130,
        },
        index=dates,
    )
    table = build_style_performance_table(
        fr,
        ["momentum", "beta", "residual_volatility", "size"],
        windows=(20, 60, 120),
        rank_window=60,
    )

    assert plot_style_heatmap(table).get_title()
    assert plot_style_rank_bar(table).get_title()
    assert plot_key_style_groups(fr).get_title()
    assert plot_market_temperature(fr).get_title()


def test_notebook_contains_investor_dashboard_section():
    nb_path = Path(__file__).with_name("米筐官方因子收益率_风格趋势.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "投资者风格雷达" in all_source
    assert "build_style_performance_table" in all_source
    assert "generate_market_style_commentary" in all_source
    assert "研究附录" in all_source
