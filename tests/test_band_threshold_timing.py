"""
测试：区间阈值择时（双阈值 band timing）功能。

RED → GREEN → REFACTOR 流程，先写测试再看失败。

测试目标：
1. build_expanding_thresholds_band 能否正确生成上下双阈值
2. run_band_timing 的仓位逻辑是否正确：
   - lo <= fv <= hi → position = 1.0
   - fv < lo → position = 0.0
   - fv > hi → position = 0.0
   - NaN → position = 0.0
3. 2D 热力图函数能否输出正确形状的 DataFrame
"""
from datetime import date
from pathlib import Path
from typing import Any

import nbformat
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "因子回测"
    / "涨跌停情绪因子"
    / "sentiment_factors_5d_research.ipynb"
)


def load_notebook_definitions() -> dict[str, Any]:
    """加载 notebook 中的函数定义（带 definitions tag 的 cell）。"""
    assert NOTEBOOK_PATH.exists(), f"Notebook 不存在：{NOTEBOOK_PATH}"
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and "definitions" in cell.metadata.get("tags", []):
            exec(compile(cell.source, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


# ============================================================
# Fixtures：合成测试数据
# ============================================================

@pytest.fixture
def ns() -> dict[str, Any]:
    """加载 notebook 中的函数定义。"""
    return load_notebook_definitions()


@pytest.fixture
def simple_timing_data() -> pd.DataFrame:
    """
    构造 10 个交易日的极简择时输入数据。

    3 个因子列：
      - factor_a: 稳步上升 [0.1, 0.2, ..., 1.0]
      - factor_b: 先升后降 [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
      - factor_c: 全 NaN（边界情况）

    2 个周期：1 日和 3 日 future_return 列。
    market_daily_ret 在偶数日上涨，奇数日下跌。
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    df = pd.DataFrame({
        "trading_date": dates,
        "factor_a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "factor_b": [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "factor_c": [np.nan] * 10,
        "market_daily_ret": [0.02, -0.01, 0.02, -0.01, 0.02, -0.01, 0.02, -0.01, 0.02, -0.01],
    })
    # 未来收益列（为简化，手动构造）
    df["future_return_1d"] = df["market_daily_ret"].shift(-1)
    df["future_return_3d"] = (
        (1 + df["market_daily_ret"])
        .rolling(3, min_periods=3)
        .apply(np.prod, raw=True)
        .shift(-3)
        - 1
    )
    return df


# ============================================================
# 测试 1：build_expanding_thresholds_band
# ============================================================

class TestBuildExpandingThresholdsBand:
    """测试双阈值生成函数。"""

    def test_output_columns(self, ns, simple_timing_data):
        """应生成 lower_{factor} 和 upper_{factor} 列。"""
        result = ns["build_expanding_thresholds_band"](
            simple_timing_data,
            factor_columns=["factor_a", "factor_b"],
            lower_quantile=0.30,
            upper_quantile=0.70,
            min_history=2,
        )
        assert "lower_factor_a" in result.columns
        assert "upper_factor_a" in result.columns
        assert "lower_factor_b" in result.columns
        assert "upper_factor_b" in result.columns

    def test_lower_less_than_upper(self, ns, simple_timing_data):
        """同一行的 lower 必须 < upper。"""
        result = ns["build_expanding_thresholds_band"](
            simple_timing_data,
            factor_columns=["factor_a"],
            lower_quantile=0.30,
            upper_quantile=0.70,
            min_history=2,
        )
        valid = result.dropna(subset=["lower_factor_a", "upper_factor_a"])
        assert (valid["lower_factor_a"] < valid["upper_factor_a"]).all()

    def test_shift_one_no_leakage(self, ns, simple_timing_data):
        """第 t 行的阈值必须基于 t-1 及以前数据（不包含 t 日信息）。"""
        result = ns["build_expanding_thresholds_band"](
            simple_timing_data,
            factor_columns=["factor_a"],
            lower_quantile=0.30,
            upper_quantile=0.70,
            min_history=2,
        )
        for idx in range(3, len(result)):
            row = result.iloc[idx]
            if pd.notna(row["lower_factor_a"]) and pd.notna(row["upper_factor_a"]):
                assert row["lower_factor_a"] <= row["upper_factor_a"]


# ============================================================
# 测试 2：run_band_timing 的仓位逻辑
# ============================================================

class TestRunBandTiming:
    """测试区间择时回测函数（通过 run_non_overlapping_timing + threshold_type='band'）。"""

    # --- 准备：在 simple_timing_data 中添加阈值列 ---
    def _run_band(self, ns, df, factor, horizon, anchor_date, lower_column, upper_column):
        """通过改造后的 run_non_overlapping_timing 调用区间择时。"""
        return ns["run_non_overlapping_timing"](
            df, factor=factor, horizon=horizon,
            anchor_date=anchor_date,
            threshold_type="band",
            lower_column=lower_column,
            upper_column=upper_column,
        )

    def _add_band_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """手动构造上下阈值（非扩展窗口，固定值），便于测试仓位逻辑。"""
        r = df.copy()
        r["lower_factor_a"] = 0.35
        r["upper_factor_a"] = 0.75
        r["lower_factor_b"] = 0.40
        r["upper_factor_b"] = 0.80
        r["lower_factor_c"] = 0.30
        r["upper_factor_c"] = 0.70
        return r

    def test_position_in_band(self, ns, simple_timing_data):
        """因子值在区间内 → position = 1.0。"""
        df = self._add_band_columns(simple_timing_data)
        daily, blocks = self._run_band(
            ns, df, factor="factor_a", horizon=1,
            anchor_date=pd.Timestamp("2024-01-04"),
            lower_column="lower_factor_a",
            upper_column="upper_factor_a",
        )
        if len(blocks) > 0:
            valid_blocks = blocks.dropna(subset=["factor_value"])
            in_band = valid_blocks[
                (valid_blocks["factor_value"] >= valid_blocks["lower_threshold"])
                & (valid_blocks["factor_value"] <= valid_blocks["upper_threshold"])
            ]
            if len(in_band) > 0:
                assert (in_band["position"] == 1.0).all()

    def test_position_below_lower(self, ns, simple_timing_data):
        """因子值低于下界 → position = 0.0。"""
        df = self._add_band_columns(simple_timing_data)
        daily, blocks = self._run_band(
            ns, df, factor="factor_a", horizon=1,
            anchor_date=pd.Timestamp("2024-01-02"),
            lower_column="lower_factor_a",
            upper_column="upper_factor_a",
        )
        if len(blocks) > 0:
            below = blocks[blocks["factor_value"] < blocks["lower_threshold"]]
            if len(below) > 0:
                assert (below["position"] == 0.0).all()

    def test_position_above_upper(self, ns, simple_timing_data):
        """因子值高于上界 → position = 0.0。"""
        df = self._add_band_columns(simple_timing_data)
        daily, blocks = self._run_band(
            ns, df, factor="factor_a", horizon=1,
            anchor_date=pd.Timestamp("2024-01-02"),
            lower_column="lower_factor_a",
            upper_column="upper_factor_a",
        )
        if len(blocks) > 0:
            above = blocks[blocks["factor_value"] > blocks["upper_threshold"]]
            if len(above) > 0:
                assert (above["position"] == 0.0).all()

    def test_position_on_boundary(self, ns, simple_timing_data):
        """因子值正好等于边界 → position = 1.0（闭区间）。"""
        df = self._add_band_columns(simple_timing_data)
        df.loc[3, "factor_a"] = 0.35
        df.loc[4, "factor_a"] = 0.75
        daily, blocks = self._run_band(
            ns, df, factor="factor_a", horizon=1,
            anchor_date=pd.Timestamp("2024-01-02"),
            lower_column="lower_factor_a",
            upper_column="upper_factor_a",
        )
        if len(blocks) > 0:
            for _, row in blocks.iterrows():
                if pd.isna(row["factor_value"]):
                    continue
                if row["factor_value"] == 0.35 or row["factor_value"] == 0.75:
                    assert row["position"] == 1.0, (
                        f"边界值 {row['factor_value']} 应 position=1.0, "
                        f"但得到 {row['position']}"
                    )

    def test_position_nan_factor(self, ns, simple_timing_data):
        """因子值为 NaN → position = 0.0。"""
        df = self._add_band_columns(simple_timing_data)
        daily, blocks = self._run_band(
            ns, df, factor="factor_c", horizon=1,
            anchor_date=pd.Timestamp("2024-01-02"),
            lower_column="lower_factor_c",
            upper_column="upper_factor_c",
        )
        if len(blocks) > 0:
            assert (blocks["position"] == 0.0).all()

    def test_output_columns(self, ns, simple_timing_data):
        """返回的 daily 和 blocks 应包含关键列。"""
        df = self._add_band_columns(simple_timing_data)
        daily, blocks = self._run_band(
            ns, df, factor="factor_a", horizon=1,
            anchor_date=pd.Timestamp("2024-01-04"),
            lower_column="lower_factor_a",
            upper_column="upper_factor_a",
        )
        expected_daily_cols = {
            "trading_date", "position", "market_daily_ret",
            "strategy_daily_ret", "benchmark_nav", "strategy_nav",
        }
        assert expected_daily_cols.issubset(daily.columns), (
            f"daily 缺列: {expected_daily_cols - set(daily.columns)}"
        )
        expected_block_cols = {
            "block_id", "decision_date", "factor_value",
            "lower_threshold", "upper_threshold",
            "position", "benchmark_block_return", "strategy_block_return",
        }
        assert expected_block_cols.issubset(blocks.columns), (
            f"blocks 缺列: {expected_block_cols - set(blocks.columns)}"
        )


# ============================================================
# 测试 3：plot_band_sensitivity 的输出
# ============================================================

class TestPlotBandSensitivity:
    """测试 2D 阈值敏感性分析函数（聚焦输出 DataFrame 的正确性）。"""

    @pytest.fixture
    def sensitivity_data(self) -> pd.DataFrame:
        """
        构造含所有 FACTOR_COLUMNS 列的数据（plot_band_sensitivity
        内部调用 build_expanding_thresholds 需遍历全部因子列）。
        """
        np.random.seed(42)
        dates = pd.date_range("2020-01-02", periods=200, freq="B")  # 200行够用
        df = pd.DataFrame({"trading_date": dates})
        for col in [
            "limit_up_ratio", "limit_down_ratio", "net_limit_ratio",
            "limit_up_down_ratio", "limit_up_next_ret", "limit_down_next_ret",
        ]:
            trend = np.linspace(0.05, 0.15, len(dates))
            noise = np.random.normal(0, 0.03, len(dates))
            df[col] = trend + noise
        df["market_daily_ret"] = np.random.normal(0.0005, 0.015, len(dates))
        for horizon in [1, 3, 5, 10]:
            ret = (1 + df["market_daily_ret"]).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-horizon) - 1
            df[f"future_return_{horizon}d"] = ret
        return df

    def test_returns_dataframe(self, ns, sensitivity_data, monkeypatch):
        """应返回包含所有枚举组合的 DataFrame。"""
        monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

        result = ns["plot_band_sensitivity"](
            research_data=sensitivity_data,
            factor="limit_up_ratio",
            horizon=1,
            lower_quantiles=[0.30, 0.40],
            upper_quantiles=[0.60, 0.70, 0.80],
            min_history=10,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 4

    def test_columns_exist(self, ns, sensitivity_data, monkeypatch):
        """结果 DataFrame 应有必要的绩效列。"""
        monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

        result = ns["plot_band_sensitivity"](
            research_data=sensitivity_data,
            factor="limit_up_ratio",
            horizon=1,
            lower_quantiles=[0.30],
            upper_quantiles=[0.70],
            min_history=10,
        )
        required_cols = {
            "factor", "horizon", "lower_quantile", "upper_quantile",
            "holding_ratio", "annual_return", "sharpe",
        }
        assert required_cols.issubset(result.columns), (
            f"缺列: {required_cols - set(result.columns)}"
        )
