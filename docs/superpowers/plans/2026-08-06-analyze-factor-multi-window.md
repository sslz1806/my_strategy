# Analyze Factor Multi-Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 将 analyze_factor 重构为基于单张 Polars 长表的多窗口因子分析接口，正确计算无未来函数的 IC、非重叠分组收益、可选基准和多窗口图表。

**Architecture:** 计算层先对股票日收益所需窗口取并集，通过现有 add_future_return 一次生成未来收益，再分别派生 IC 长表和非重叠分组收益长表。展示层只消费计算结果，统一生成四类多窗口 Figure 并按需保存；Alpha191 适配器只负责把已有 Pandas 宽表向量化转换成新入口要求的 Polars 长表。

**Tech Stack:** Python 3.9+、Polars 1.33.1、Pandas、NumPy、Matplotlib、pytest。

## Global Constraints

- 所有 Python 与 pytest 命令使用 E:\working\anaconda3\envs\quant\python.exe。
- 不增加第三方依赖；复用 因子回测/alpha.py::add_future_return。
- 输入固定使用 trading_date、code 和可选 benchmark_ret；股票收益与基准收益均为小数口径。
- factor[t] 的未来 w 期收益只包含 t+1 至 t+w；ret[t] 不得进入。
- ret_windows=w 的调仓日期按全市场交易日序号每 w 期取一次，收益区间不得重叠。
- 分组净值按 period_end=t+w 入账和绘图，不得按 signal_date=t 提前显示。
- IC 每日计算；ic_window 大于 1 时允许相邻 IC 样本的未来收益窗口重叠。
- 不保留旧 analyze_factor 参数兼容分支，也不保留 analyze_factor_bak。
- 输入正确性由调用方保证，只保留 benchmark_ret 缺失或全空这一项业务分支。
- 新增或修改的关键业务注释使用中文，解释收益口径、入账日期与非重叠规则。
- 当前工作区已有未提交改动。每次暂存前检查 git diff；对因子回测/alpha.py 和 因子回测/因子回测框架.ipynb 使用 git add -p 只选择本计划修改的函数或单元 hunk，再用 git diff --cached 逐项确认，绝不暂存用户原有的 backtest_timeseries_factor 或其他 Notebook 改动。若目标 hunk 无法与用户改动安全分离，则不提交该任务并在交付中说明。

## File Structure

- Create: tests/test_analyze_factor.py
  - 对新计算核心、公开接口、绘图、保存和可选基准进行手算级验证。
- Create: tests/test_alpha191_analyze_factor.py
  - 验证 Alpha191Calculator 将宽表转换为单张 Polars 长表，并调用新接口。
- Create: tests/test_analyze_factor_notebook.py
  - 静态验证主因子回测 Notebook 不再构造未来收益宽表或调用旧参数。
- Modify: 因子回测/alpha.py:406-641
  - 替换旧 analyze_factor，增加精简的计算、绩效和绘图私有辅助函数。
- Modify: 因子回测/alpha_191/calculator.py:22-28, 218-252
  - 将 wrapper 改为 ret_windows/ic_windows，并生成 Polars 长表。
- Modify: 因子回测/alpha_191/README.md:29-36, 165-169
  - 更新调用示例与数据格式说明。
- Modify: 因子回测/alpha_191/__init__.py:8-17
  - 更新包级示例和“宽表分析”描述。
- Modify: 因子回测/因子回测框架.ipynb: code cells 1-2
  - 删除未来收益 shift，改为日收益列和新 analyze_factor 入口。

---

### Task 1: 实现向量化计算核心

**Files:**
- Create: tests/test_analyze_factor.py
- Modify: 因子回测/alpha.py:406-641

**Interfaces:**
- Consumes: add_future_return(df, ret_col, horizons, date_col, code_col)。
- Produces: _compute_factor_analysis(data, factor_col, ret_col, ret_windows, ic_windows, group_num) -> dict。
- Produces: _build_nav_and_stats(period_returns) -> tuple[pl.DataFrame, pl.DataFrame]。
- 计算结果键为 ic、ic_stats、group_returns、group_stats、nav、benchmark_stats。

- [ ] **Step 1: 写计算核心的失败测试**

在 tests/test_analyze_factor.py 中加入以下初始内容：

    from __future__ import annotations

    from datetime import date, timedelta
    from unittest.mock import patch

    import matplotlib
    import numpy as np
    import polars as pl

    matplotlib.use("Agg")

    from 因子回测.alpha import (
        _compute_factor_analysis,
        add_future_return,
    )


    def make_panel(include_benchmark: bool = True) -> pl.DataFrame:
        dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(10)]
        codes = ["A", "B", "C", "D"]
        rows = []
        for date_index, trading_date in enumerate(dates):
            for code_index, code in enumerate(codes):
                daily_ret = 0.20 - code_index * 0.10 if date_index == 0 else 0.01 * (code_index + 1)
                row = {
                    "trading_date": trading_date,
                    "code": code,
                    "factor": float(code_index + 1),
                    "daily_ret": daily_ret,
                }
                if include_benchmark:
                    row["benchmark_ret"] = 0.005
                rows.append(row)
        return pl.DataFrame(rows)


    def test_core_uses_t_plus_one_return_and_realization_date():
        result = _compute_factor_analysis(
            make_panel(include_benchmark=False),
            factor_col="factor",
            ret_col="daily_ret",
            ret_windows=(1,),
            ic_windows=(1,),
            group_num=2,
        )

        first_g1 = (
            result["group_returns"]
            .filter(
                (pl.col("window") == 1)
                & (pl.col("group") == "G1")
            )
            .sort("signal_date")
            .row(0, named=True)
        )
        assert first_g1["signal_date"] == date(2024, 1, 2)
        assert first_g1["period_end"] == date(2024, 1, 3)
        assert np.isclose(first_g1["return"], (0.01 + 0.02) / 2)


    def test_core_calculates_union_of_future_windows_once():
        panel = make_panel(include_benchmark=False)
        with patch(
            "因子回测.alpha.add_future_return",
            wraps=add_future_return,
        ) as future_return_spy:
            _compute_factor_analysis(
                panel,
                factor_col="factor",
                ret_col="daily_ret",
                ret_windows=(1, 3),
                ic_windows=(1, 3),
                group_num=2,
            )

        assert future_return_spy.call_count == 1
        assert future_return_spy.call_args.kwargs["horizons"] == (1, 3)


    def test_core_has_multi_window_ic_and_cumulative_ic():
        result = _compute_factor_analysis(
            make_panel(include_benchmark=False),
            factor_col="factor",
            ret_col="daily_ret",
            ret_windows=(1, 3),
            ic_windows=(1, 3),
            group_num=2,
        )

        assert result["ic"]["window"].unique().sort().to_list() == [1, 3]
        for window in (1, 3):
            window_ic = result["ic"].filter(pl.col("window") == window).sort("trading_date")
            assert np.allclose(
                window_ic["cum_ic"].to_numpy(),
                window_ic["ic"].cum_sum().to_numpy(),
            )
            assert np.allclose(
                window_ic["cum_rank_ic"].to_numpy(),
                window_ic["rank_ic"].cum_sum().to_numpy(),
            )


    def test_three_period_group_returns_do_not_overlap():
        result = _compute_factor_analysis(
            make_panel(include_benchmark=False),
            factor_col="factor",
            ret_col="daily_ret",
            ret_windows=(3,),
            ic_windows=(1,),
            group_num=2,
        )

        g1 = (
            result["group_returns"]
            .filter(pl.col("group") == "G1")
            .sort("signal_date")
        )
        assert g1["signal_date"].to_list() == [
            date(2024, 1, 2),
            date(2024, 1, 5),
            date(2024, 1, 8),
        ]
        assert g1["period_end"].to_list() == [
            date(2024, 1, 5),
            date(2024, 1, 8),
            date(2024, 1, 11),
        ]
        expected = ((1.01 ** 3 - 1) + (1.02 ** 3 - 1)) / 2
        assert np.allclose(g1["return"].to_numpy(), expected)


    def test_core_adds_benchmark_on_matching_non_overlapping_dates():
        result = _compute_factor_analysis(
            make_panel(include_benchmark=True),
            factor_col="factor",
            ret_col="daily_ret",
            ret_windows=(3,),
            ic_windows=(1,),
            group_num=2,
        )

        benchmark_nav = (
            result["nav"]
            .filter(
                (pl.col("window") == 3)
                & (pl.col("group") == "benchmark")
            )
            .sort("period_end")
        )
        assert benchmark_nav["period_end"][0] == date(2024, 1, 5)
        assert np.isclose(benchmark_nav["nav"][0], 1.005 ** 3)
        assert result["benchmark_stats"] is not None

- [ ] **Step 2: 运行测试并确认因新计算入口不存在而失败**

Run:

    E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_analyze_factor.py -q

Expected: collection 阶段因无法导入 _compute_factor_analysis 而失败。

- [ ] **Step 3: 添加净值和绩效计算辅助函数**

在旧 analyze_factor 之前添加：

    def _build_nav_and_stats(period_returns: pl.DataFrame):
        """从非重叠区间收益生成净值和同口径绩效。"""
        keys = ["window", "group"]
        nav = (
            period_returns
            .sort(keys + ["period_end"])
            .with_columns(
                (1 + pl.col("return"))
                .cum_prod()
                .over(keys)
                .alias("nav")
            )
            .with_columns(
                pl.max_horizontal(
                    pl.lit(1.0),
                    pl.col("nav").cum_max().over(keys),
                ).alias("_peak")
            )
            .with_columns(
                (pl.col("nav") / pl.col("_peak") - 1).alias("_drawdown")
            )
        )

        stats = (
            nav.group_by(keys)
            .agg(
                pl.len().alias("periods"),
                pl.col("return").mean().alias("mean_return"),
                pl.col("return").std().alias("_return_std"),
                (1 + pl.col("return")).product().alias("_gross"),
                pl.col("_drawdown").min().alias("max_drawdown"),
                (pl.col("return") > 0).mean().alias("positive_ratio"),
            )
            .with_columns(
                (
                    pl.col("_gross").pow(
                        252.0
                        / (
                            pl.col("window").cast(pl.Float64)
                            * pl.col("periods")
                        )
                    )
                    - 1
                ).alias("annual_return"),
                pl.when(pl.col("_return_std") != 0)
                .then(
                    pl.col("mean_return")
                    / pl.col("_return_std")
                    * (252.0 / pl.col("window")).sqrt()
                )
                .otherwise(None)
                .alias("sharpe"),
            )
            .select(
                "window",
                "group",
                "mean_return",
                "annual_return",
                "sharpe",
                "max_drawdown",
                "positive_ratio",
                "periods",
            )
            .sort(keys)
        )
        return (
            nav.select("period_end", "window", "group", "nav"),
            stats,
        )

- [ ] **Step 4: 添加多窗口计算核心**

在 _build_nav_and_stats 后添加：

    def _compute_factor_analysis(
        data: pl.DataFrame,
        factor_col: str,
        ret_col: str,
        ret_windows: Sequence[int],
        ic_windows: Sequence[int],
        group_num: int,
    ) -> dict:
        """计算多窗口 IC、非重叠分组收益及可选基准。"""
        ret_windows = tuple(sorted(set(ret_windows)))
        ic_windows = tuple(sorted(set(ic_windows)))
        all_windows = tuple(sorted(set(ret_windows) | set(ic_windows)))
        future_columns = {
            window: f"future_{ret_col}_{window}d"
            for window in all_windows
        }

        prepared = add_future_return(
            data,
            ret_col=ret_col,
            horizons=all_windows,
            date_col="trading_date",
            code_col="code",
        )
        date_map = (
            prepared.select("trading_date")
            .unique()
            .sort("trading_date")
            .with_row_index("_date_index")
            .with_columns(
                [
                    pl.col("trading_date")
                    .shift(-window)
                    .alias(f"_period_end_{window}")
                    for window in ret_windows
                ]
            )
        )
        prepared = (
            prepared.join(date_map, on="trading_date", how="left")
            .sort(
                ["trading_date", factor_col, "code"],
                nulls_last=True,
            )
            .with_columns(
                pl.col(factor_col)
                .rank("ordinal")
                .over("trading_date")
                .alias("_factor_rank"),
                pl.col(factor_col)
                .count()
                .over("trading_date")
                .alias("_factor_count"),
            )
            .with_columns(
                (
                    (
                        (pl.col("_factor_rank") - 1)
                        * group_num
                        / pl.col("_factor_count")
                    )
                    .floor()
                    .cast(pl.Int64)
                    + 1
                ).alias("_group_no")
            )
            .with_columns(
                pl.format("G{}", pl.col("_group_no")).alias("group")
            )
        )

        ic_frames = []
        for window in ic_windows:
            ic_frames.append(
                prepared.group_by("trading_date")
                .agg(
                    pl.corr(
                        factor_col,
                        future_columns[window],
                        method="pearson",
                    ).alias("ic"),
                    pl.corr(
                        factor_col,
                        future_columns[window],
                        method="spearman",
                    ).alias("rank_ic"),
                )
                .drop_nulls(["ic", "rank_ic"])
                .with_columns(pl.lit(window).alias("window"))
                .select("trading_date", "window", "ic", "rank_ic")
            )

        ic = (
            pl.concat(ic_frames)
            .sort(["window", "trading_date"])
            .with_columns(
                pl.col("ic").cum_sum().over("window").alias("cum_ic"),
                pl.col("rank_ic")
                .cum_sum()
                .over("window")
                .alias("cum_rank_ic"),
            )
        )
        ic_stats = (
            ic.group_by("window")
            .agg(
                pl.len().alias("observations"),
                pl.col("ic").mean().alias("ic_mean"),
                pl.col("ic").std().alias("ic_std"),
                (pl.col("ic").mean() / pl.col("ic").std()).alias("ic_ir"),
                (pl.col("ic") > 0).mean().alias("ic_positive_ratio"),
                pl.col("rank_ic").mean().alias("rank_ic_mean"),
                pl.col("rank_ic").std().alias("rank_ic_std"),
                (
                    pl.col("rank_ic").mean()
                    / pl.col("rank_ic").std()
                ).alias("rank_ic_ir"),
                (pl.col("rank_ic") > 0)
                .mean()
                .alias("rank_ic_positive_ratio"),
            )
            .sort("window")
        )

        group_frames = []
        for window in ret_windows:
            group_frames.append(
                prepared.filter(
                    pl.col("_date_index") % window == 0
                )
                .select(
                    pl.col("trading_date").alias("signal_date"),
                    pl.col(f"_period_end_{window}").alias("period_end"),
                    "group",
                    pl.col(future_columns[window]).alias("_future_return"),
                )
                .drop_nulls(["period_end", "group", "_future_return"])
                .group_by("signal_date", "period_end", "group")
                .agg(
                    pl.col("_future_return").mean().alias("return")
                )
                .with_columns(pl.lit(window).alias("window"))
                .select(
                    "signal_date",
                    "period_end",
                    "window",
                    "group",
                    "return",
                )
            )
        group_returns = pl.concat(group_frames).sort(
            ["window", "signal_date", "group"]
        )

        has_benchmark = (
            "benchmark_ret" in data.columns
            and data["benchmark_ret"].null_count() < data.height
        )
        benchmark_returns = None
        if has_benchmark:
            benchmark_daily = (
                data.select("trading_date", "benchmark_ret")
                .drop_nulls("benchmark_ret")
                .unique(subset=["trading_date"], keep="first")
                .sort("trading_date")
            )
            benchmark_future = add_future_return(
                benchmark_daily,
                ret_col="benchmark_ret",
                horizons=ret_windows,
                date_col="trading_date",
                code_col="code",
            ).join(date_map, on="trading_date", how="left")
            benchmark_frames = []
            for window in ret_windows:
                benchmark_frames.append(
                    benchmark_future.filter(
                        pl.col("_date_index") % window == 0
                    )
                    .select(
                        pl.col("trading_date").alias("signal_date"),
                        pl.col(f"_period_end_{window}").alias("period_end"),
                        pl.lit(window).alias("window"),
                        pl.lit("benchmark").alias("group"),
                        pl.col(
                            f"future_benchmark_ret_{window}d"
                        ).alias("return"),
                    )
                    .drop_nulls(["period_end", "return"])
                )
            benchmark_returns = pl.concat(benchmark_frames)

        all_period_returns = (
            group_returns
            if benchmark_returns is None
            else pl.concat([group_returns, benchmark_returns])
        )
        nav, all_stats = _build_nav_and_stats(all_period_returns)
        group_stats = all_stats.filter(
            pl.col("group") != "benchmark"
        )
        benchmark_stats = (
            None
            if benchmark_returns is None
            else all_stats.filter(pl.col("group") == "benchmark")
        )

        return {
            "ic": ic,
            "ic_stats": ic_stats,
            "group_returns": group_returns,
            "group_stats": group_stats,
            "nav": nav,
            "benchmark_stats": benchmark_stats,
        }

- [ ] **Step 5: 运行计算测试并修正 Polars 表达式兼容性**

Run:

    E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_analyze_factor.py -q

Expected: 5 passed。

- [ ] **Step 6: 运行现有未来收益回归测试**

Run:

    E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_add_future_return.py tests/test_add_future_return_polars_panel.py -q

Expected: 全部通过。

- [ ] **Step 7: 审查本任务差异并安全记录**

Run:

    git diff --check -- tests/test_analyze_factor.py 因子回测/alpha.py
    git diff -- tests/test_analyze_factor.py 因子回测/alpha.py

因子回测/alpha.py 已含用户原有修改，交互暂存时只选择 _build_nav_and_stats、_compute_factor_analysis 和被替换的旧 analyze_factor 所在 hunk，不选择 backtest_timeseries_factor hunk：

    git add -- tests/test_analyze_factor.py
    git add -p -- 因子回测/alpha.py
    git diff --cached -- tests/test_analyze_factor.py 因子回测/alpha.py
    git diff --cached --check
    git commit -m "refactor: add factor analysis computation core"

Expected: 提交包含可共同运行的核心测试和计算实现，不含 backtest_timeseries_factor 的原有工作区修改。

---

### Task 2: 替换公开接口并生成多窗口图表

**Files:**
- Modify: tests/test_analyze_factor.py
- Modify: 因子回测/alpha.py:406-641

**Interfaces:**
- Consumes: _compute_factor_analysis 的六个结果表。
- Produces: analyze_factor(data, factor_col, ret_col="daily_ret", ret_windows=(1, 3, 5), ic_windows=(1, 3, 5), group_num=5, save_result=False) -> dict。
- Produces: figures 字典，固定包含 group_returns、ic_series、cumulative_ic、nav 四个 Figure。

- [ ] **Step 1: 添加公开接口、基准为空和绘图结构测试**

向 tests/test_analyze_factor.py 追加：

    import matplotlib.pyplot as plt
    import pytest

    from 因子回测.alpha import analyze_factor


    @pytest.mark.parametrize("benchmark_mode", ["missing", "all_null"])
    def test_public_api_omits_empty_benchmark(benchmark_mode):
        panel = make_panel(include_benchmark=False)
        if benchmark_mode == "all_null":
            panel = panel.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("benchmark_ret")
            )

        result = analyze_factor(
            panel,
            factor_col="factor",
            ret_windows=(1, 3),
            ic_windows=(1, 3),
            group_num=2,
        )

        assert result["benchmark_stats"] is None
        assert "benchmark" not in result["nav"]["group"].unique().to_list()
        assert all(
            line.get_label() != "benchmark"
            for axis in result["figures"]["nav"].axes
            for line in axis.lines
        )
        plt.close("all")


    def test_public_api_returns_one_subplot_per_window():
        result = analyze_factor(
            make_panel(include_benchmark=True),
            factor_col="factor",
            ret_windows=(1, 3),
            ic_windows=(1, 3),
            group_num=2,
        )

        assert len(result["figures"]["group_returns"].axes) == 2
        assert len(result["figures"]["nav"].axes) == 2
        assert len(result["figures"]["ic_series"].axes) == 2
        assert len(result["figures"]["cumulative_ic"].axes) == 2
        assert any(
            line.get_label() == "benchmark"
            for axis in result["figures"]["nav"].axes
            for line in axis.lines
        )
        plt.close("all")


    def test_save_result_writes_tables_and_figures(tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = analyze_factor(
            make_panel(include_benchmark=True),
            factor_col="factor",
            ret_windows=(1,),
            ic_windows=(1,),
            group_num=2,
            save_result=True,
        )

        output_dir = tmp_path / "因子分析结果"
        expected_names = {
            "ic.csv",
            "ic_stats.csv",
            "group_returns.csv",
            "group_stats.csv",
            "nav.csv",
            "benchmark_stats.csv",
            "group_returns.png",
            "ic_series.png",
            "cumulative_ic.png",
            "nav.png",
        }
        assert expected_names == {
            path.name for path in output_dir.iterdir()
        }
        assert set(result["figures"]) == {
            "group_returns",
            "ic_series",
            "cumulative_ic",
            "nav",
        }
        plt.close("all")

- [ ] **Step 2: 运行新增测试并确认旧 analyze_factor 接口失败**

Run:

    E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_analyze_factor.py -q

Expected: 新的三个公开接口测试因旧签名不接受 data/factor_col 而失败。

- [ ] **Step 3: 添加多窗口绘图辅助函数**

在 _compute_factor_analysis 后添加 _plot_factor_analysis。实现采用每窗口一个纵向子图，所有图都从返回长表读取；净值横轴必须使用 period_end：

    def _plot_factor_analysis(results, ret_windows, ic_windows):
        """为每类结果创建一个多窗口 Figure。"""
        ret_windows = tuple(sorted(set(ret_windows)))
        ic_windows = tuple(sorted(set(ic_windows)))

        def make_axes(windows, title):
            figure, axes = plt.subplots(
                len(windows),
                1,
                figsize=(12, 4 * len(windows)),
                squeeze=False,
            )
            figure.suptitle(title)
            return figure, axes[:, 0]

        group_figure, group_axes = make_axes(
            ret_windows,
            "分组平均收益",
        )
        for axis, window in zip(group_axes, ret_windows):
            frame = results["group_stats"].filter(
                pl.col("window") == window
            )
            rows = sorted(
                frame.iter_rows(named=True),
                key=lambda row: int(row["group"][1:]),
            )
            axis.bar(
                [row["group"] for row in rows],
                [row["mean_return"] * 100 for row in rows],
            )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{window}期非重叠收益")
            axis.set_ylabel("平均收益(%)")
            axis.grid(alpha=0.3, axis="y")

        ic_figure, ic_axes = make_axes(ic_windows, "IC / RankIC")
        cumulative_figure, cumulative_axes = make_axes(
            ic_windows,
            "累计 IC / RankIC",
        )
        for ic_axis, cumulative_axis, window in zip(
            ic_axes,
            cumulative_axes,
            ic_windows,
        ):
            frame = (
                results["ic"]
                .filter(pl.col("window") == window)
                .sort("trading_date")
            )
            dates = frame["trading_date"].to_list()
            ic_axis.plot(dates, frame["ic"], label="IC")
            ic_axis.plot(dates, frame["rank_ic"], label="RankIC")
            ic_axis.axhline(0, color="black", linewidth=0.8)
            ic_axis.set_title(f"{window}期未来收益")
            ic_axis.grid(alpha=0.3)
            ic_axis.legend()

            cumulative_axis.plot(
                dates,
                frame["cum_ic"],
                label="累计IC",
            )
            cumulative_axis.plot(
                dates,
                frame["cum_rank_ic"],
                label="累计RankIC",
            )
            cumulative_axis.axhline(0, color="black", linewidth=0.8)
            cumulative_axis.set_title(f"{window}期未来收益")
            cumulative_axis.grid(alpha=0.3)
            cumulative_axis.legend()

        nav_figure, nav_axes = make_axes(ret_windows, "分组净值")
        for axis, window in zip(nav_axes, ret_windows):
            frame = results["nav"].filter(
                pl.col("window") == window
            )
            groups = sorted(
                [
                    group
                    for group in frame["group"].unique().to_list()
                    if group != "benchmark"
                ],
                key=lambda group: int(group[1:]),
            )
            for group in groups:
                group_nav = (
                    frame.filter(pl.col("group") == group)
                    .sort("period_end")
                )
                axis.plot(
                    group_nav["period_end"].to_list(),
                    group_nav["nav"].to_list(),
                    label=group,
                )
            benchmark = (
                frame.filter(pl.col("group") == "benchmark")
                .sort("period_end")
            )
            if not benchmark.is_empty():
                axis.plot(
                    benchmark["period_end"].to_list(),
                    benchmark["nav"].to_list(),
                    linestyle="--",
                    color="black",
                    label="benchmark",
                )
            axis.set_title(f"{window}期非重叠净值")
            axis.grid(alpha=0.3)
            axis.legend()

        figures = {
            "group_returns": group_figure,
            "ic_series": ic_figure,
            "cumulative_ic": cumulative_figure,
            "nav": nav_figure,
        }
        for figure in figures.values():
            figure.tight_layout(rect=(0, 0, 1, 0.97))
        return figures

- [ ] **Step 4: 用新签名替换旧 analyze_factor**

删除旧 Pandas 宽表实现及全局 warnings.filterwarnings("ignore")，保留已有 backtest_timeseries_factor。增加 pathlib.Path 导入，并写入：

    def analyze_factor(
        data: pl.DataFrame,
        factor_col: str,
        ret_col: str = "daily_ret",
        ret_windows: Sequence[int] = (1, 3, 5),
        ic_windows: Sequence[int] = (1, 3, 5),
        group_num: int = 5,
        save_result: bool = False,
    ) -> dict:
        """
        分析横截面因子的多窗口 IC 和非重叠分组收益。

        daily_ret[t] 表示 t-1 至 t 的收益；factor[t] 只对应
        t+1 至 t+w 的未来收益。benchmark_ret 为可选固定列。
        """
        results = _compute_factor_analysis(
            data=data,
            factor_col=factor_col,
            ret_col=ret_col,
            ret_windows=ret_windows,
            ic_windows=ic_windows,
            group_num=group_num,
        )
        figures = _plot_factor_analysis(
            results,
            ret_windows=ret_windows,
            ic_windows=ic_windows,
        )

        if save_result:
            output_dir = Path("因子分析结果")
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "ic",
                "ic_stats",
                "group_returns",
                "group_stats",
                "nav",
            ):
                results[name].write_csv(output_dir / f"{name}.csv")
            if results["benchmark_stats"] is not None:
                results["benchmark_stats"].write_csv(
                    output_dir / "benchmark_stats.csv"
                )
            for name, figure in figures.items():
                figure.savefig(
                    output_dir / f"{name}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

        plt.show()
        return {**results, "figures": figures}

- [ ] **Step 5: 运行公开接口测试**

Run:

    E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_analyze_factor.py -q

Expected: 8 passed，且无 Matplotlib 交互窗口阻塞。

- [ ] **Step 6: 验证语法与差异**

Run:

    E:\working\anaconda3\envs\quant\python.exe -m py_compile 因子回测/alpha.py
    git diff --check -- 因子回测/alpha.py tests/test_analyze_factor.py

Expected: 两条命令退出码均为 0。不要整文件暂存 alpha.py。

- [ ] **Step 7: 提交公开接口和绘图**

对 alpha.py 使用交互暂存，只选择 _plot_factor_analysis 和新 analyze_factor 的 hunk：

    git add -- tests/test_analyze_factor.py
    git add -p -- 因子回测/alpha.py
    git diff --cached -- tests/test_analyze_factor.py 因子回测/alpha.py
    git diff --cached --check
    git commit -m "feat: add multi-window factor analysis plots"

Expected: 暂存差异不包含 backtest_timeseries_factor。

---

### Task 3: 迁移 Alpha191 适配入口

**Files:**
- Create: tests/test_alpha191_analyze_factor.py
- Modify: 因子回测/alpha_191/calculator.py:22-28, 218-252
- Modify: 因子回测/alpha_191/README.md:29-36, 165-169
- Modify: 因子回测/alpha_191/__init__.py:8-17

**Interfaces:**
- Consumes: Alpha191Calculator.compute_df(alpha_num) 与 self.data["returns"] 两张 Pandas 宽表。
- Produces: Alpha191Calculator.analyze_factor(alpha_num, ret_windows=(1, 3, 5), ic_windows=(1, 3, 5), group_num=5)。
- 传给底层 analyze_factor 的 data 固定含 trading_date、code、factor、daily_ret。

- [ ] **Step 1: 写 Alpha191 转换测试**

创建 tests/test_alpha191_analyze_factor.py：

    from datetime import datetime

    import pandas as pd
    import polars as pl

    import 因子回测.alpha as alpha_module
    from 因子回测.alpha_191.calculator import Alpha191Calculator


    def test_alpha191_analyze_factor_builds_one_long_polars_table(
        monkeypatch,
    ):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        factor_wide = pd.DataFrame(
            {"A": [1.0, 2.0], "B": [3.0, 4.0]},
            index=dates,
        )
        return_wide = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.03, 0.04]},
            index=dates,
        )

        calculator = object.__new__(Alpha191Calculator)
        calculator._is_loaded = True
        calculator.data = {"returns": return_wide}
        monkeypatch.setattr(
            calculator,
            "compute_df",
            lambda alpha_num: factor_wide,
        )

        captured = {}

        def fake_analyze_factor(**kwargs):
            captured.update(kwargs)
            return {"sentinel": True}

        monkeypatch.setattr(
            alpha_module,
            "analyze_factor",
            fake_analyze_factor,
        )

        result = calculator.analyze_factor(
            5,
            ret_windows=(1, 3),
            ic_windows=(1, 5),
            group_num=2,
        )

        assert result == {"sentinel": True, "alpha_num": 5}
        assert isinstance(captured["data"], pl.DataFrame)
        assert captured["data"].columns == [
            "trading_date",
            "code",
            "factor",
            "daily_ret",
        ]
        assert captured["data"].height == 4
        assert captured["ret_windows"] == (1, 3)
        assert captured["ic_windows"] == (1, 5)
        assert captured["group_num"] == 2
        row = (
            captured["data"]
            .filter(
                (pl.col("trading_date") == datetime(2024, 1, 3))
                & (pl.col("code") == "B")
            )
            .row(0, named=True)
        )
        assert row["factor"] == 4.0
        assert row["daily_ret"] == 0.04

- [ ] **Step 2: 运行测试并确认旧 wrapper 签名失败**

Run:

    E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_alpha191_analyze_factor.py -q

Expected: FAIL，错误指出 return_period/adjust_freq 旧接口或新窗口参数不匹配。

- [ ] **Step 3: 向量化转换宽表并调用新接口**

在 calculator.py 中导入 polars as pl，并把 typing 导入扩展为 Sequence。将方法替换为：

    def analyze_factor(
        self,
        alpha_num: int,
        ret_windows: Sequence[int] = (1, 3, 5),
        ic_windows: Sequence[int] = (1, 3, 5),
        group_num: int = 5,
    ) -> dict:
        """使用单张 Polars 长表执行多窗口因子分析。"""
        self._check_loaded()
        from 因子回测.alpha import analyze_factor

        factor_long = (
            pl.from_pandas(
                self.compute_df(alpha_num)
                .rename_axis("trading_date")
                .reset_index()
            )
            .unpivot(
                index="trading_date",
                variable_name="code",
                value_name="factor",
            )
        )
        return_long = (
            pl.from_pandas(
                self.data["returns"]
                .rename_axis("trading_date")
                .reset_index()
            )
            .unpivot(
                index="trading_date",
                variable_name="code",
                value_name="daily_ret",
            )
        )
        analysis_data = factor_long.join(
            return_long,
            on=["trading_date", "code"],
            how="inner",
        )

        result = analyze_factor(
            data=analysis_data,
            factor_col="factor",
            ret_col="daily_ret",
            ret_windows=ret_windows,
            ic_windows=ic_windows,
            group_num=group_num,
        )
        result["alpha_num"] = alpha_num
        return result

- [ ] **Step 4: 更新包级示例和 README**

将 __init__.py 示例改为：

    result = calc.analyze_factor(
        5,
        ret_windows=(1, 3, 5),
        ic_windows=(1, 3, 5),
    )

将 README 快速示例改为：

    result = calc.analyze_factor(
        5,
        ret_windows=(1, 3, 5),
        ic_windows=(1, 3, 5),
        group_num=5,
    )

把“极简版宽表分析”改为“Polars 长表多窗口因子分析”，并说明 wrapper 使用 data["returns"] 的当期日收益，不再预计算未来收益宽表。

- [ ] **Step 5: 运行适配器和核心测试**

Run:

    E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_alpha191_analyze_factor.py tests/test_analyze_factor.py -q

Expected: 全部通过。

- [ ] **Step 6: 提交可安全隔离的 Alpha191 迁移**

Run:

    git diff --check -- 因子回测/alpha_191/calculator.py 因子回测/alpha_191/README.md 因子回测/alpha_191/__init__.py tests/test_alpha191_analyze_factor.py
    git add -- 因子回测/alpha_191/calculator.py 因子回测/alpha_191/README.md 因子回测/alpha_191/__init__.py tests/test_alpha191_analyze_factor.py
    git diff --cached --check
    git commit -m "refactor: migrate alpha191 factor analysis"

Expected: 提交只包含上述四个文件。

---

### Task 4: 迁移主 Notebook 并完成回归验证

**Files:**
- Create: tests/test_analyze_factor_notebook.py
- Modify: 因子回测/因子回测框架.ipynb: code cells 1-2
- Verify: 因子回测/alpha.py
- Verify: 因子回测/alpha_191/calculator.py

**Interfaces:**
- Notebook 直接传入包含 volume_factor_neutralized 和 daily_ret 的 stock_data。
- analyze_factor 调用只使用 data、factor_col、ret_col、ret_windows、ic_windows、group_num、save_result。

- [ ] **Step 1: 写 Notebook 静态失败测试**

创建 tests/test_analyze_factor_notebook.py：

    import json
    from pathlib import Path


    NOTEBOOK = (
        Path(__file__).resolve().parents[1]
        / "因子回测"
        / "因子回测框架.ipynb"
    )


    def test_factor_notebook_uses_daily_return_long_table_api():
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        assert "analyze_factor(" in source
        assert "data=stock_data" in source
        assert 'factor_col="volume_factor_neutralized"' in source
        assert 'ret_col="daily_ret"' in source
        assert "ret_windows=(1, 3, 5)" in source
        assert "ic_windows=(1, 3, 5)" in source
        assert "factor_data=" not in source
        assert "ret_data=" not in source
        assert "adjust_freq=" not in source
        assert "return_period=" not in source
        assert "shift(-5)" not in source

- [ ] **Step 2: 运行测试并确认 Notebook 仍使用旧接口**

Run:

    E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_analyze_factor_notebook.py -q

Expected: FAIL，至少命中 data=stock_data 缺失和旧参数仍存在。

- [ ] **Step 3: 修改 Notebook 的因子构造单元**

保留原均线、因子和中性化逻辑，但从第一个 with_columns 中删除 future_return_5d，增加无未来信息的当期日收益：

    stock_data = stock_data.sort(["code", "trading_date"])
    stock_data = stock_data.with_columns([
        pl.col("volume").rolling_mean(window_size=10).over("code").alias("volume_ma10"),
        pl.col("volume").rolling_mean(window_size=60).over("code").alias("volume_ma60"),
        (pl.col("close") / pl.col("pre_close") - 1).alias("daily_ret"),
    ])
    stock_data = stock_data.with_columns(
        (pl.col("volume_ma10") / pl.col("volume_ma60")).alias("volume_factor")
    )
    stock_data = stock_data.drop_nulls(subset=["volume_factor"])
    stock_data = stock_data.group_by("trading_date").map_groups(
        lambda df: ols_neutralize(
            df,
            y_column="volume_factor",
            x_columns=["total_mv"],
        )
    )

使用 apply_patch 修改 Notebook JSON 中对应 source 数组；保留其他单元和用户已有输出，不运行自动格式化整本 Notebook。

- [ ] **Step 4: 修改 Notebook 的分析调用单元**

将旧宽表构造和调用替换为：

    a = analyze_factor(
        data=stock_data,
        factor_col="volume_factor_neutralized",
        ret_col="daily_ret",
        ret_windows=(1, 3, 5),
        ic_windows=(1, 3, 5),
        group_num=5,
        save_result=False,
    )

- [ ] **Step 5: 运行完整针对性验证**

Run:

    E:\working\anaconda3\envs\quant\python.exe -m pytest tests/test_analyze_factor.py tests/test_alpha191_analyze_factor.py tests/test_analyze_factor_notebook.py tests/test_add_future_return.py tests/test_add_future_return_polars_panel.py -q
    E:\working\anaconda3\envs\quant\python.exe -m py_compile 因子回测/alpha.py 因子回测/alpha_191/calculator.py
    git diff --check

Expected: 所有测试通过、两个 Python 文件编译成功、git diff --check 无空白错误。

- [ ] **Step 6: 审计旧调用位置**

Run:

    git grep -n -I -E "factor_data=|ret_data=|adjust_freq=|return_period=" -- 因子回测

逐项确认 calculator.py、README、__init__.py 和主因子回测 Notebook 已无旧调用。其余研究 Notebook 只记录在交付说明，不做批量改写；明确列出文件名和调用形式，避免把历史 Notebook 的未迁移状态误报为新接口已兼容。

- [ ] **Step 7: 提交可安全隔离的 Notebook 迁移和测试**

Notebook 已含用户原有修改，使用交互暂存只选择 code cells 1-2 的本次 source 变化，并逐行审查暂存差异：

    git add -- tests/test_analyze_factor_notebook.py
    git add -p -- 因子回测/因子回测框架.ipynb
    git diff --cached -- tests/test_analyze_factor_notebook.py 因子回测/因子回测框架.ipynb
    git diff --cached --check
    git commit -m "docs: migrate factor analysis notebook"

Expected: 提交只包含 Notebook 两个目标代码单元和对应静态测试，不含其他输出或用户原有单元变化。

- [ ] **Step 8: 完成需求逐项核验**

核对以下证据后才允许宣告完成：

- analyze_factor 签名只接收单张 pl.DataFrame 和新窗口参数。
- future return spy 证明股票未来收益对窗口并集只调用一次。
- 手算测试证明 factor[t] 使用 t+1 至 t+w。
- signal_date/period_end 测试证明净值未提前入账。
- ret_windows=(1, 3, 5) 的调仓日期和收益区间不重叠。
- ic、IC 统计、IC 时序图和累计 IC 图覆盖所有 ic_windows。
- 四类 Figure 的子图数与各自窗口数一致。
- benchmark_ret 有值时产生基准统计和虚线；缺失或全空时完全跳过。
- Alpha191 wrapper 和主 Notebook 使用新入口。
- 目标测试与 py_compile 全部通过。
