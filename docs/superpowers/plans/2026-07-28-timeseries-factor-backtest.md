# 时序因子分组回测函数 `backtest_timeseries_factor` 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `alpha.py` 中添加 `backtest_timeseries_factor()` 函数，将 `情绪因子 v2.ipynb` cell-6 中的分组测试与分组策略回测代码包装为可复用接口。

**Architecture:** 单函数注入 `alpha.py`，保持 notebook 原版 `backtest_group_strategy` 和 `calculate_group_performance_metrics` 作为内部函数，仅将 `index_ret_col` 复利合成未来收益替换原 `nav.shift(-h)/nav-1` 逻辑。

**Tech Stack:** Python 3.9+, pandas, numpy, matplotlib, `alpha.py`

## Global Constraints

- 函数定义在 `因子回测/alpha.py`，位于 `analyze_factor()` 之后
- 使用 pandas DataFrame，不引入新依赖
- `index_ret_col` 单位为百分比（5 表示 5%）
- `analysis_data.index` 为时间轴（函数不假定频率粒度），入参后先 `sort_index()`
- `index_ret_col` 的 NaN 填充为 0 后参与复利计算
- 所有图表通过 `plt.subplots()` 创建 Figure，`plot=True` 时调用 `fig.show()`，始终通过返回值返回 Figure
- 与原 notebook 保持一致的量化精度

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `因子回测/alpha.py:617+` | 追加 | `backtest_timeseries_factor()` 函数定义 |
| `tests/test_timeseries_factor.py` | 新建 | 函数正确性测试 |

---

### Task 1: 实现 `backtest_timeseries_factor` 函数

**Files:**
- Modify: `因子回测/alpha.py:617+`（`analyze_factor()` 函数之后追加）
- Test: `tests/test_timeseries_factor.py`（新建）

**Interfaces:**
- Consumes: 设计文档 `docs/superpowers/specs/2026-07-28-timeseries-factor-backtest-design.md`
- Produces:
  ```python
  def backtest_timeseries_factor(
      analysis_data: pd.DataFrame,
      factor_col: str,
      index_ret_col: str,
      q: int = 5,
      hold_period: int = 5,
      plot: bool = True,
  ) -> dict:
      """时序因子分组回测：分组统计 + 每组独立策略回测 + 净值对比

      参数
      ----
      analysis_data : pd.DataFrame
          时序因子数据。index 为时间轴（交易日/15分钟/任意有序时间戳均可，
          函数不假定其频率粒度）。必须含 factor_col 和 index_ret_col 两列。
      factor_col : str
          因子值列名。
      index_ret_col : str
          单期收益率列名（%单位，如 0.5 表示 0.5%）。NaN 在内部填充为 0。
      q : int
          等分位组数，默认 5。
      hold_period : int
          持仓期数（与 index 粒度一致），默认 5。
      plot : bool
          是否自动显示图表。Figure 始终通过返回值返回。

      返回
      ----
      dict : {
          'group_stats': pd.DataFrame,      # 分组未来收益统计（均值/标准差/样本数）
          'group_performance': pd.DataFrame, # 分组策略绩效（累计/年化/夏普/回撤/胜率/持仓占比）
          'group_nav': pd.DataFrame,         # 各分组净值宽表
          'future_return_col': str,          # 内部未来收益列名
          'fig_bar': plt.Figure | None,      # 分组收益柱状图
          'fig_nav': plt.Figure | None,      # 净值对比图
      }

      注意
      ----
      - 内部保留 backtest_group_strategy 和 calculate_group_performance_metrics
        作为嵌套函数，与 notebook 原版实现一致。
      - 未来收益从 index_ret_col 复利合成（替代原 notebook 的 nav.shift(-h)/nav-1）。
      - 有效样本数小于 q 时抛 ValueError。
      """
  ```

- [ ] **Step 1: 读取 notebook 源模板**

  从 `情绪因子 v2.ipynb` cell-6 提取两个内部函数的完整代码：
  - `backtest_group_strategy(data, group_col, target_group, hold_period)`
  - `calculate_group_performance_metrics(data, group_name)`

  以及分组测试主流程、图表代码。

- [ ] **Step 2: 确认 `alpha.py` 顶部已有必要的 import**

  `alpha.py` 当前 import 了 pandas, numpy, matplotlib。确认已有：
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  ```
  无需新增 import。

- [ ] **Step 3: 在 `alpha.py` 末尾 `analyze_factor()` 之后追加函数**

  完整函数代码如下。注意要点：

  1. **未来收益合成**（替换原 notebook 的 `nav.shift(-h)` 逻辑）：
     ```python
     future_return_col = f'future_return_{hold_period}d'
     ret_series = analysis_data[index_ret_col].fillna(0)
     gross = 1 + ret_series / 100
     future_gross = gross.rolling(hold_period).apply(np.prod, raw=True).shift(-hold_period)
     analysis_data[future_return_col] = (future_gross - 1) * 100
     ```
     尾部 `shift(-hold_period)` 产生的 NaN 随后的 `dropna` 自动丢弃。

  2. **两个内部函数**直接移植 notebook 原代码，仅将硬编码的 `daily_return` 替换为参数 `index_ret_col`。

  3. **`backtest_group_strategy` 内部**：
     ```python
     def backtest_group_strategy(data, group_col, target_group, hold_period):
         """单个分组策略回测"""
         data = data.copy().dropna(subset=[group_col])
         data['signal'] = (data[group_col] == target_group).astype(int)

         signal_arr = data['signal'].to_numpy()
         position = np.zeros(len(data), dtype=np.int8)
         remaining_days = 0

         for i, s in enumerate(signal_arr):
             if s == 1:
                 remaining_days = hold_period
             if remaining_days > 0:
                 position[i] = 1
                 remaining_days -= 1

         data['position'] = position
         data['strategy_return'] = data['position'] * data[index_ret_col] / 100
         data['strategy_nav'] = (1 + data['strategy_return']).cumprod()
         return data
     ```

  4. **`calculate_group_performance_metrics` 内部**：
     ```python
     def calculate_group_performance_metrics(data, group_name):
         """计算分组策略表现指标"""
         returns = data['strategy_return'].dropna()
         nav = data['strategy_nav'].dropna()

         if len(returns) == 0:
             return None

         total_return = nav.iloc[-1] - 1
         years = len(returns) / 252
         annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

         daily_rf = 0.03 / 252
         excess_return = returns - daily_rf
         sharpe = np.sqrt(252) * excess_return.mean() / returns.std() if returns.std() > 0 else 0

         peak = nav.expanding().max()
         drawdown = (nav - peak) / peak
         max_drawdown = drawdown.min()

         win_rate = (returns > 0).mean()
         position_days = (data['position'] == 1).sum()
         position_ratio = position_days / len(data)

         return {
             '分组': group_name,
             '累计收益': total_return * 100,
             '年化收益': annual_return * 100,
             '夏普比率': sharpe,
             '最大回撤': max_drawdown * 100,
             '胜率': win_rate * 100,
             '持仓占比': position_ratio * 100,
         }
     ```

  5. **主流程**：

     ```python
     analysis_data = analysis_data.copy()
     analysis_data = analysis_data.sort_index()

     # 1. 未来收益合成
     future_return_col = f'future_return_{hold_period}d'
     ret_series = analysis_data[index_ret_col].fillna(0)
     gross = 1 + ret_series / 100
     future_gross = gross.rolling(hold_period).apply(np.prod, raw=True).shift(-hold_period)
     analysis_data[future_return_col] = (future_gross - 1) * 100

     # 2. 分组
     analysis_data_clean = analysis_data.dropna(subset=[factor_col]).copy()
     if len(analysis_data_clean) < q:
         raise ValueError(f"有效样本数 ({len(analysis_data_clean)}) 小于分组数 ({q})，无法分组")

     analysis_data_clean['factor_group'] = pd.qcut(
         analysis_data_clean[factor_col],
         q=q,
         labels=[f'G{i+1}' for i in range(q)],
         duplicates='drop'
     )
     actual_groups = analysis_data_clean['factor_group'].nunique()
     if actual_groups < q:
         print(f"⚠ 因子重复值过多，实际分组数 {actual_groups} < {q}")

     # 3. 分组未来收益统计
     group_returns = analysis_data_clean.groupby('factor_group')[future_return_col].agg(['mean', 'std', 'count'])
     group_returns.columns = ['平均收益(%)', '收益标准差', '样本数']
     print(f"\n{factor_col}分组未来{hold_period}期收益统计:")
     print(group_returns.round(4))

     # 4. 柱状图
     fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
     colors = ['#e74c3c', '#e67e22', '#f1c40f', '#27ae60', '#2980b9', '#3498db',
               '#9b59b6', '#1abc9c', '#e84393', '#00b894'][:q]
     bars = ax_bar.bar(group_returns.index, group_returns['平均收益(%)'], color=colors, alpha=0.7)
     ax_bar.axhline(y=0, color='black', linestyle='-', alpha=0.5)

     for bar in bars:
         height = bar.get_height()
         ax_bar.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height>0 else -0.05),
                     f'{height:.4f}%', ha='center', va='bottom' if height>0 else 'top')

     ax_bar.set_title(f'{factor_col} 分组未来{hold_period}期平均收益', fontsize=14)
     ax_bar.set_xlabel('因子分组（G1最低，Gq最高）', fontsize=12)
     ax_bar.set_ylabel('平均收益(%)', fontsize=12)
     ax_bar.grid(alpha=0.3, axis='y')
     fig_bar.tight_layout()

     # 5. 分组策略回测
     groups = [f'G{i+1}' for i in range(actual_groups)]
     group_results = {}
     for group in groups:
         group_strategy = backtest_group_strategy(
             analysis_data_clean, 'factor_group', target_group=group, hold_period=hold_period
         )
         group_results[group] = group_strategy

     # 基准：买入持有
     benchmark_data = analysis_data_clean.copy()
     benchmark_data['position'] = 1
     benchmark_data['strategy_return'] = benchmark_data[index_ret_col] / 100
     benchmark_data['strategy_nav'] = (1 + benchmark_data['strategy_return']).cumprod()

     # 绩效指标
     all_metrics = []
     group_nav_dict = {}
     for group in groups:
         metrics = calculate_group_performance_metrics(group_results[group], group)
         all_metrics.append(metrics)
         group_nav_dict[group] = group_results[group]['strategy_nav']

     benchmark_metrics = calculate_group_performance_metrics(benchmark_data, '买入持有基准')
     all_metrics.append(benchmark_metrics)

     performance_df = pd.DataFrame(all_metrics).set_index('分组')
     group_nav_df = pd.DataFrame(group_nav_dict, index=analysis_data_clean.index)

     print("\n" + "=" * 90)
     print("所有分组策略表现汇总")
     print("=" * 90)
     display_df = performance_df.copy()
     for col in ['累计收益', '年化收益', '最大回撤', '胜率', '持仓占比']:
         if col in display_df.columns:
             display_df[col] = display_df[col].apply(lambda x: f'{x:.2f}%')
     if '夏普比率' in display_df.columns:
         display_df['夏普比率'] = display_df['夏普比率'].apply(lambda x: f'{x:.2f}')
     print(display_df)

     # 6. 净值对比图
     fig_nav, ax_nav = plt.subplots(figsize=(14, 8))
     nav_colors = colors[:len(groups)] + ['#888888']
     nav_labels = [f'{g}' for g in groups] + ['买入持有基准']

     for i, group in enumerate(groups):
         ax_nav.plot(group_nav_df.index, group_nav_df[group],
                     label=nav_labels[i], color=nav_colors[i], linewidth=2)

     ax_nav.plot(benchmark_data.index, benchmark_data['strategy_nav'],
                 label='买入持有基准', color='#888888', linewidth=2, linestyle='--')

     ax_nav.set_title(f'{factor_col} 各分组策略净值对比（持有{hold_period}期）', fontsize=16)
     ax_nav.set_xlabel('时间', fontsize=12)
     ax_nav.set_ylabel('净值（初始=1）', fontsize=12)
     ax_nav.legend(fontsize=11, loc='best')
     ax_nav.grid(alpha=0.3)
     fig_nav.tight_layout()

     # 7. 显示
     if plot:
         fig_bar.show()
         fig_nav.show()

     return {
         'group_stats': group_returns,
         'group_performance': performance_df,
         'group_nav': group_nav_df,
         'future_return_col': future_return_col,
         'fig_bar': fig_bar,
         'fig_nav': fig_nav,
     }
     ```

- [ ] **Step 4: 创建测试文件 `tests/test_timeseries_factor.py`**

  测试函数的核心逻辑：构造一个简单的时序 DataFrame，调用函数验证返回结构正确性。

  ```python
  """测试时序因子分组回测函数 backtest_timeseries_factor"""
  import sys
  import pandas as pd
  import numpy as np
  from datetime import datetime, timedelta

  sys.path.append(r'C:\Users\20561\Desktop\策略\因子回测')
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
          q=5, hold_period=5, plot=False
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
      assert len(result['group_nav'].columns) <= 5
      assert result['future_return_col'] == 'future_return_5d'

      # 验证 Figure
      assert result['fig_bar'] is not None
      assert result['fig_nav'] is not None


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
              q=5, hold_period=2, plot=False
          )
          assert False, "应抛出 ValueError"
      except ValueError:
          pass


  def test_backtest_timeseries_factor_future_return():
      """验证未来收益合成逻辑：常数收益下 future_return 应正确"""
      dates = pd.date_range('2020-01-01', periods=10, freq='D')
      # 每日 1% 收益，5 日复利 = (1.01^5 - 1) * 100 ≈ 5.101%
      analysis_data = pd.DataFrame({
          'factor': [1]*5 + [2]*5,
          'ret': [1.0]*10,  # 每天 1%
      }, index=dates)

      result = backtest_timeseries_factor(
          analysis_data, factor_col='factor', index_ret_col='ret',
          q=2, hold_period=5, plot=False
      )

      fr_col = result['future_return_col']
      # 前 5 行的 future_return_5d ≈ (1.01^5 - 1) * 100 ≈ 5.101
      expected = (1.01**5 - 1) * 100
      # 因为 qcut 被 factor=[1]*5+[2]*5 分成 2 组，G1 和 G2 各有 5 行
      # 每组内第 1-5 行 future_return 应该 ≈ expected
      assert abs(result['group_stats'].loc['G1', '平均收益(%)'] - expected) < 0.01, \
          f"G1 平均收益 {result['group_stats'].loc['G1', '平均收益(%)']} != {expected}"
  ```

- [ ] **Step 5: 运行测试确认通过**

  ```bash
  cd c:/Users/20561/Desktop/策略
  E:/working/anaconda3/envs/quant/bin/python -m pytest tests/test_timeseries_factor.py -v
  ```

  预期输出：
  ```
  tests/test_timeseries_factor.py::test_backtest_timeseries_factor_basic PASSED
  tests/test_timeseries_factor.py::test_backtest_timeseries_factor_sample_too_small PASSED
  tests/test_timeseries_factor.py::test_backtest_timeseries_factor_future_return PASSED
  ```

- [ ] **Step 6: 提交**

  ```bash
  cd c:/Users/20561/Desktop/策略
  git add 因子回测/alpha.py tests/test_timeseries_factor.py \
         docs/superpowers/specs/2026-07-28-timeseries-factor-backtest-design.md \
         docs/superpowers/plans/2026-07-28-timeseries-factor-backtest.md
  git commit -m "feat: add backtest_timeseries_factor for time-series factor group backtesting

  - New function backtest_timeseries_factor() added to alpha.py
  - Ports backtest_group_strategy and calculate_group_performance_metrics
    from 情绪因子 v2.ipynb as internal functions
  - Replaces nav.shift(-h)/nav-1 with rolling compound from index_ret_col
  - Tests cover basic structure, sample-too-small error, and future return
    calculation accuracy"
  ```
