# 分组回测时点修复计划

**目标：** 保持“t 日尾盘按收盘价决策并成交”的设定，修复两处确定问题：全样本分组未来函数，以及持仓收益错误包含 t 日已经发生的 close-to-close 收益。

## 不改的部分

- 不改情绪因子的计算：因子使用 t 日数据是允许的。
- 不改为分钟线、不引入开盘价或新数据。
- 不改分组数、持仓期参数和现有图表结构。
- 不要求分组结果与阈值 OR 组合策略数值相同。

## 改动 1：历史分位数分组，消除未来函数

文件：因子回测/alpha.py 中 backtest_timeseries_factor。

现状是对完整样本直接做：

~~~python
pd.qcut(analysis_data_clean[factor_col], q=q, ...)
~~~

这会让 t 日属于哪个组依赖 t 之后的因子值。

改为：对每个 t 日，用 factor[t-1] 及更早历史计算 q-1 个分位数边界，再用 factor[t] 与边界比较，得到 G1 到 Gq。

~~~python
history = factor.shift(1)
boundary_20 = history.expanding(min_periods=min_history).quantile(0.20)
boundary_40 = history.expanding(min_periods=min_history).quantile(0.40)
boundary_60 = history.expanding(min_periods=min_history).quantile(0.60)
boundary_80 = history.expanding(min_periods=min_history).quantile(0.80)
~~~

q 不固定为 5 时，循环计算 1/q 到 (q-1)/q 的边界。Notebook 已有 MIN_HISTORY=252；将它明确传给回测函数。预热期没有分组，不进入绩效。

## 改动 2：t 收盘成交，收益从 t+1 开始

文件：因子回测/alpha.py 中内部的 backtest_group_strategy。

交易口径：

~~~text
t 日收盘：得到因子、决定并按 close[t] 买入
t+1 到 t+h：持有并累计 h 个 close-to-close 日收益
t+h 日收盘：卖出
~~~

因此 h 日收益应为：

~~~text
close[t+h] / close[t] - 1
= Π(1 + ret[t+1] ... ret[t+h]) - 1
~~~

当前代码在信号日直接写 position[i]=1，并把 ret[i] 计入收益；应只把仓位写入下一行：

~~~python
for i, is_signal in enumerate(signal_arr[:-1]):
    if is_signal:
        remaining_days = hold_period

    if remaining_days > 0:
        position[i + 1] = 1
        remaining_days -= 1
~~~

若 t、t+1 都命中同一组，t+1 的新信号可以续期；但每一天 position 仍只能为 0 或 1，所以没有重复乘收益。

基准净值也从第一个可交易收益日，即首个有效分组日的 t+1 开始，保证横向可比。

## Notebook 改动

文件：因子回测/涨跌停情绪因子/择时增强_分组回测.ipynb。

只改两处：

1. 每个 backtest_timeseries_factor 调用加上：

~~~python
min_history=MIN_HISTORY,
~~~

2. 重启 Notebook 内核后从上至下重新执行，避免旧模块仍在内存中。

## 两个必须新增的测试

文件：tests/test_timeseries_factor.py。

1. **未来分组测试**：只修改样本末尾的因子值，断言此前日期的分组不变。
2. **收盘成交测试**：构造 t 日收益 99%、t+1 日收益 1% 的样本。若 t 日发出 G1 信号，G1 首个净值必须为 1.01，而不是 1.99。

## 验证命令

~~~powershell
& 'E:\working\anaconda3\envs\quant\python.exe' -m pytest -q -p no:cacheprovider tests\test_timeseries_factor.py
~~~

通过后，用新内核重跑分组 Notebook；预期分组回测的结果会变化，但这是消除未来分组与收益错位后的正常变化。

