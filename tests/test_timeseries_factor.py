"""测试时序因子分组回测函数 backtest_timeseries_factor"""
import sys
import pandas as pd
import numpy as np

sys.path.append(r'C:\Users\20561\Desktop\策略\因子回测')
import alpha
from alpha import backtest_timeseries_factor


def test_backtest_timeseries_factor_basic():
    """基础测试：验证返回结构和样本不足时的异常"""
    # 构造 200 期时序数据
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    np.random.seed(42)

    analysis_data = pd.DataFrame({
        'factor': np.random.randn(200),
        'ret': np.random.randn(200) * 0.5,  # %单位
    }, index=dates)

    # 正常调用
    result = backtest_timeseries_factor(
        analysis_data, factor_col='factor', index_ret_col='ret',
        q=5, hold_period=5, window=20, plot=False,
    )

    # 验证返回键
    assert 'group_stats' in result
    assert 'group_performance' in result
    assert 'group_nav' in result
    assert 'future_return_col' in result
    assert 'fig_bar' in result
    assert 'fig_nav' in result

    # 验证 group_stats 结构
    assert result['group_stats'].shape[0] <= 5  # duplicates='drop'
    assert '平均收益(%)' in result['group_stats'].columns
    assert '样本数' in result['group_stats'].columns

    # 验证 group_performance 结构（G1~Gq + 基准）
    assert '买入持有基准' in result['group_performance'].index
    assert '累计收益' in result['group_performance'].columns
    assert '夏普比率' in result['group_performance'].columns

    # 验证 group_nav 结构
    assert len(result['group_nav'].columns) <= 6  # 最多 q 组 + 1 列基准
    assert result['future_return_col'] == 'future_return_5d'

    # 验证 Figure 对象（plot=False 时不创建）
    assert result['fig_bar'] is None
    assert result['fig_nav'] is None


def test_backtest_timeseries_factor_sample_too_small():
    """验证样本不足时抛 ValueError"""
    dates = pd.date_range('2020-01-01', periods=3, freq='D')
    analysis_data = pd.DataFrame({
        'factor': [1, 2, 3],
        'ret': [0.1, -0.2, 0.3],
    }, index=dates)

    try:
        backtest_timeseries_factor(
            analysis_data, factor_col='factor', index_ret_col='ret',
            q=5, hold_period=2, window=1, plot=False,
        )
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_backtest_timeseries_factor_future_return():
    """验证未来收益合成逻辑：常数收益下 future_return 应正确"""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    # 每日 1% 收益，5 日复利 = (1.01^5 - 1) * 100 ≈ 5.101%
    analysis_data = pd.DataFrame({
        'factor': [1] * 5 + [2] * 5,
        'ret': [1.0] * 10,  # 每天 1%
    }, index=dates)

    result = backtest_timeseries_factor(
        analysis_data, factor_col='factor', index_ret_col='ret',
        q=2, hold_period=5, window=1, plot=False,
    )

    fr_col = result['future_return_col']
    # 前 5 行的 future_return_5d ≈ (1.01^5 - 1) * 100 ≈ 5.101
    expected = (1.01 ** 5 - 1) * 100
    # qcut 将 [1]*5+[2]*5 分成 2 组，每组 5 行
    # 每组内第 1-5 行 future_return 应该 ≈ expected
    assert abs(
        result['group_stats'].loc['G1', '平均收益(%)'] - expected
    ) < 0.01, (
        f"G1 平均收益 {result['group_stats'].loc['G1', '平均收益(%)']} != {expected}"
    )


def test_backtest_timeseries_factor_nan_ret():
    """验证 index_ret_col 含 NaN 时 fillna(0) 后仍正常工作"""
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    np.random.seed(42)

    # 构造数据，中间插入 NaN
    factor_vals = np.random.randn(50)
    ret_vals = np.random.randn(50) * 0.3
    ret_vals[10:15] = np.nan  # 中间 5 个 NaN

    analysis_data = pd.DataFrame({
        'factor': factor_vals,
        'ret': ret_vals,
    }, index=dates)

    # 不应报错
    result = backtest_timeseries_factor(
        analysis_data, factor_col='factor', index_ret_col='ret',
        q=5, hold_period=3, window=5, plot=False,
    )

    assert result['group_stats'] is not None
    assert result['fig_bar'] is None
    assert result['fig_nav'] is None


def test_backtest_timeseries_factor_default_q():
    """验证默认参数 q=5, hold_period=5 可正常执行"""
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    np.random.seed(1)
    analysis_data = pd.DataFrame({
        'factor': np.random.randn(300),
        'ret': np.random.randn(300) * 0.4,
    }, index=dates)

    # 使用全部默认参数
    result = backtest_timeseries_factor(
        analysis_data, factor_col='factor', index_ret_col='ret',
        plot=False,
    )

    assert len(result['group_nav'].columns) == 6  # q 组 + 基准
    assert result['future_return_col'] == 'future_return_5d'


def test_backtest_timeseries_factor_plot_true():
    """验证 plot=True 时正确创建并返回 Figure 对象"""
    dates = pd.date_range('2020-01-01', periods=80, freq='D')
    np.random.seed(7)
    analysis_data = pd.DataFrame({
        'factor': np.random.randn(80),
        'ret': np.random.randn(80) * 0.5,
    }, index=dates)

    result = backtest_timeseries_factor(
        analysis_data, factor_col='factor', index_ret_col='ret',
        q=5, hold_period=5, window=10, plot=True,
    )

    assert result['fig_bar'] is not None
    assert result['fig_nav'] is not None
    # Figure 对象应有 axes（即已绘制内容）
    assert len(result['fig_bar'].axes) > 0
    assert len(result['fig_nav'].axes) > 0


def test_close_signal_only_earns_returns_after_the_signal_day():
    """t 日收盘信号买入后，不能再计入已走完的 t 日 close-to-close 收益。"""
    dates = pd.date_range('2024-01-02', periods=7, freq='B')
    analysis_data = pd.DataFrame(
        {
            'factor': [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0],
            # 第 3 行的 99% 为信号当日已实现收益；下一日收益只有 1%。
            'ret': [0.0, 0.0, 99.0, 1.0, 2.0, 3.0, 4.0],
        },
        index=dates,
    )

    result = backtest_timeseries_factor(
        analysis_data,
        factor_col='factor',
        index_ret_col='ret',
        q=2,
        hold_period=1,
        window=3,
        plot=False,
        verbose=False,
    )

    # 2024-01-04 的 G1 信号按收盘成交，首个净值点应是下一日的 1% 收益。
    assert result['group_nav'].index[0] == dates[3]
    assert np.isclose(result['group_nav'].loc[dates[3], 'G1'], 1.01)


def test_factor_groups_use_recent_window_including_current_day():
    """分组必须使用截至 t 日的最近 window 期，而不是截至 t 日的全部历史。"""
    dates = pd.date_range('2024-01-02', periods=7, freq='B')
    analysis_data = pd.DataFrame(
        {
            # 2024-01-09 的最近 3 期为 [100, 90, 80]，中位数为 90，
            # 因此当日 80 应属于 G1；若错误使用 expanding，中位数为 40，会落入 G2。
            'factor': [0.0, 0.0, 0.0, 100.0, 90.0, 80.0, 0.0],
            # 只让 2024-01-09 的信号在下一日产生收益，便于从净值反推分组。
            'ret': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
        },
        index=dates,
    )

    result = backtest_timeseries_factor(
        analysis_data,
        factor_col='factor',
        index_ret_col='ret',
        q=2,
        hold_period=1,
        window=3,
        plot=False,
        verbose=False,
    )

    assert np.isclose(result['group_nav']['G1'].iloc[-1], 1.10)
    assert np.isclose(result['group_nav']['G2'].iloc[-1], 1.00)


def test_missing_factor_after_warmup_is_not_a_signal_or_a_missing_return_day():
    """因子缺失日不能误入 G1，也不能从策略时间轴中删除并压缩持仓期。"""
    dates = pd.date_range('2024-01-02', periods=7, freq='B')
    analysis_data = pd.DataFrame(
        {
            'factor': [0.0, 2.0, 0.0, np.nan, 3.0, 4.0, 5.0],
            # NaN 因子日的下一日有 10% 收益；若 NaN 被误标 G1，G1 会错误获得该收益。
            'ret': [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        },
        index=dates,
    )

    result = backtest_timeseries_factor(
        analysis_data,
        factor_col='factor',
        index_ret_col='ret',
        q=2,
        hold_period=1,
        window=3,
        plot=False,
        verbose=False,
    )

    expected_nav_index = dates[3:]
    assert result['group_nav'].index.equals(expected_nav_index)
    assert np.isclose(result['group_nav']['G1'].iloc[-1], 1.00)
