from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "因子回测"
    / "涨跌停情绪因子"
    / "择时增强_分组回测.ipynb"
)


def test_benchmark_codes_become_chinese_names_throughout_notebook(monkeypatch):
    """指数代码进入 RQData 后，宽表、汇总与缓存应直接使用中文名称。"""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    namespace = {}
    exec(notebook.cells[1].source, namespace)

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    requested_codes = []

    class FakeRqData:
        def get_return(self, codes, start_date, end_date):
            requested_codes.extend(codes)
            rows = [
                (code, trading_date, (code_index + 1) * 0.001)
                for code_index, code in enumerate(codes)
                for trading_date in dates
            ]
            return pd.DataFrame(
                {"return": [row[2] for row in rows]},
                index=pd.MultiIndex.from_tuples(
                    [(row[0], row[1]) for row in rows],
                    names=["order_book_id", "date"],
                ),
            )

    backtest_kwargs = []

    def fake_backtest(data, **kwargs):
        backtest_kwargs.append(kwargs)
        performance = pd.DataFrame(
            {
                "累计收益": [1.0, 2.0, 1.5],
                "夏普比率": [0.1, 0.2, 0.15],
            },
            index=["G1", "G5", "买入持有基准"],
        )
        group_nav = pd.DataFrame(
            {
                "G1": [1.0, 1.01, 1.02],
                "G5": [1.0, 1.02, 1.04],
                "买入持有基准": [1.0, 1.005, 1.01],
            },
            index=data.index,
        )
        return {"group_performance": performance, "group_nav": group_nav}

    namespace.update(
        RqData=FakeRqData,
        factor_data=pd.DataFrame({"factor": [1.0, 2.0, 3.0]}, index=dates),
        FACTOR_COLUMNS=["factor"],
        FACTOR_LABELS={"factor": "测试因子"},
        HORIZONS=(1,),
        GROUP_WINDOW=1,
        backtest_timeseries_factor=fake_backtest,
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    # 第 2 个代码单元前半段依赖本地行情文件；这里只执行指数收益合并部分，
    # 保留 RQData → 中文列名 → 批量回测的真实数据流。
    benchmark_section = notebook.cells[2].source.split(
        "# ===== 2. 获取多指数收益（宽表），与因子数据合并 =====", maxsplit=1,
    )[1]
    exec(benchmark_section, namespace)
    exec(notebook.cells[3].source, namespace)
    plt.close("all")

    expected_codes = [
        "000300.XSHG",
        "000905.XSHG",
        "000852.XSHG",
        "399303.XSHE",
    ]
    expected_names = ["沪深300", "中证500", "中证1000", "国证2000"]
    assert requested_codes == expected_codes
    assert namespace["analysis_data"].columns.tolist() == ["factor", *expected_names]
    assert namespace["summary"]["benchmark"].tolist() == expected_names
    assert [key[1] for key in namespace["nav_cache"]] == expected_names
    assert all(kwargs["window"] == 1 for kwargs in backtest_kwargs)
    assert all(
        isinstance(nav, pd.DataFrame)
        and "买入持有基准" in nav.columns
        for nav in namespace["nav_cache"].values()
    )
