# `analyze_factor` 多窗口设计

## 入口

```python
analyze_factor(
    data: pl.DataFrame,
    factor_col: str,
    ret_col: str = "daily_ret",
    ret_windows=(1, 3, 5),
    ic_windows=(1, 3, 5),
    group_num=5,
    plot=True,
    save_result=False,
)
```

`data` 是一张 Polars 长表，固定包含：

```text
trading_date | code | <factor_col> | <ret_col> | benchmark_ret(可选)
```

`ret_col[t]` 和 `benchmark_ret[t]` 都表示 `t-1` 收盘至 `t` 收盘的单期收益，
使用小数口径。调用方保证数据唯一性、类型和收益正确性。

## IC

- 复用 `add_future_return`，一次生成全部 `ic_windows` 的 `t+1...t+w` 累计收益。
- 每个日期、每个窗口计算截面 IC 和 RankIC。
- 按窗口分别计算累计 IC、累计 RankIC 和统计指标。
- IC 的多期未来收益允许相邻样本重叠，这是每日截面 IC 的正常口径。

## 分组回测

完全采用 FactorAna 的持仓口径：

1. 每日因子只计算一次截面等数量分组。
2. `ret_window=w` 表示每 `w` 个交易日调仓一次，而不是计算一笔稀疏的未来
   `w` 期收益。
3. 调仓日收盘确定股票组，下一交易日起生效；持仓组在两次调仓间保持不变。
4. 固定持仓组内每日等权，使用当日股票单期收益计算分组日收益。
5. 日收益连续复利生成净值，因此每个收益期只使用一次，净值时间轴也连续。

例如 `w=3`：第 0 日分组作用于第 1、2、3 日收益，第 3 日收盘重新分组，
新分组从第 4 日收益开始生效。

内部仅为分组回测把分组标签和日收益各 pivot 一次为 `T×N` NumPy 矩阵；
窗口内持仓用调仓锚点索引取得，不逐日循环股票。

## Benchmark

`benchmark_ret` 存在且有有效值时，从分析区间第二个交易日起连续复利，并在
每个分组净值子图中绘制同一条 benchmark 曲线。列缺失、全为 null 或 NaN 时
跳过 benchmark。

## 返回与绘图

返回 Polars 长表：`ic`、`ic_stats`、`group_returns`、`group_stats`、`nav`、
可选 `benchmark`，以及三个 Figure：

- 多窗口分组净值图
- 多窗口 IC / RankIC 图
- 多窗口累计 IC / 累计 RankIC 图

IC / RankIC 时序图参考 FactorAna 的层次设计：原始 IC 和 RankIC 用
低透明度细线保留每日波动，同时叠加同色粗线的滚动均值。公开参数
`ic_rolling_window` 默认为 30，IC 和 RankIC 使用相同窗口且
`min_periods=1`；现有均值虚线和累计 IC 图保持不变。

旧 Pandas 双宽表实现保留为 `analyze_factor_bak`，不在新入口中增加格式猜测或
兼容分支。
