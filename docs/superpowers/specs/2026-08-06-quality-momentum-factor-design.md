# 质量动量因子复现设计

## 目标

在 `因子回测/质量动量因子.ipynb` 中复现质量动量因子，并沿用
`因子回测/因子回测框架.ipynb` 的 Polars 长表与 `analyze_factor` 分析流程。
Notebook 同时计算并回测两个版本：示例代码版作为主因子，63 日半衰期版作为对照因子。

## 范围

- 修改 `因子回测/质量动量因子.ipynb`，保留现有因子说明并补全可执行代码。
- 新增 `tests/test_quality_momentum_factor_notebook.py`，用合成价格验证因子计算。
- 不修改 `my_utils/fun.py`、`因子回测/alpha.py` 或其他公共接口。
- 不引入新依赖；使用项目 `quant` 环境已有的 NumPy、Polars、nbformat 和 pytest。

## Notebook 结构

Notebook 由一个说明单元和三个代码单元组成：

1. **因子说明**：说明 26 个价格点、对数价格回归、年化收益、R² 与综合得分，明确两个权重版本的差异。
2. **环境与数据**：按参考 Notebook 的风格导入 `read_day_data`、`analyze_factor`、NumPy、Polars 和日期工具；集中声明分析起止日期、窗口、年化交易日和半衰期参数；额外向前读取 60 个自然日作为 26 日滚动窗口的预热数据。
3. **因子计算与分析数据**：定义计算函数，按股票和日期排序后生成六个因子结果列，再构造包含单期收益和等权基准收益的单张 Polars 长表。进入回测前过滤到正式分析起始日。
4. **双因子回测**：分别以 `momentum_score` 和 `momentum_score_halflife` 调用 `analyze_factor`，统一使用 1、3、5 日调仓窗口和未来收益窗口，并展示两套 IC 与分组统计。

## 因子定义

### 示例代码版（主因子）

每只股票在每个交易日使用截至当日的 26 个收盘价：

1. `y = log(prices)`，`x = [0, 1, ..., 25]`。
2. `weights = linspace(1, 2, 26)`。
3. 严格按示例调用 `np.polyfit(x, y, deg=1, w=weights)` 得到斜率和截距。
4. `annual_ret = exp(slope * 250) - 1`。
5. 按示例计算：
   - `ss_res = sum(weights * (y - y_fit) ** 2)`；
   - `ss_tot = sum(weights * (y - mean(y)) ** 2)`；
   - 当 `ss_tot <= 1e-10` 时令 `r2 = 0`，否则 `r2 = 1 - ss_res / ss_tot`。
6. `score = annual_ret * r2`。

输出列为：

- `momentum_annual_ret`
- `momentum_r2`
- `momentum_score`

该版本有意保留示例中 `np.polyfit` 权重与 R² 权重口径不完全一致的行为，确保“复刻示例代码”的结果可核对。

### 63 日半衰期版（对照因子）

价格窗口和年化公式与主因子一致，但采用标准加权最小二乘：

1. 观测权重为 `0.5 ** ((25 - x) / 63)`，最新价格权重为 1，越早的价格权重越低。
2. `np.polyfit` 接收观测权重的平方根，使最小化目标为标准的 `sum(weight * residual ** 2)`。
3. R² 的总平方和使用相同观测权重下的加权均值。
4. 常数价格窗口的 R² 设为 0；仅对浮点误差导致的轻微越界将 R² 限制在 `[0, 1]`。

输出列为：

- `momentum_annual_ret_halflife`
- `momentum_r2_halflife`
- `momentum_score_halflife`

## 计算实现

- Notebook 内提供单窗口计算函数，参数显式区分 `example` 与 `halflife` 两种模式。
- 按股票使用 NumPy 滑动窗口批量处理 26 日价格矩阵，避免对全市场每一行调用 Python 回调。
- Polars 负责排序、分组、结果拼接、收益构造和回测输入整理；不会转换整张全市场数据到 Pandas。
- 每个窗口必须包含 26 个有限且严格大于 0 的价格。数据不足、包含空值、非有限值或非正价格时，该日六个结果列保持空值。
- 因子包含当日收盘价；`daily_ret[t] = close[t] / pre_close[t] - 1`。现有 `analyze_factor` 会从下一期收益开始评价 t 日因子，避免同日收益前视。

## 回测数据流

```text
向前预热 60 个自然日读取日线
  → 按 code、trading_date 排序
  → 26 日滚动计算示例版与半衰期版
  → daily_ret 与当日股票池等权 benchmark_ret
  → 过滤到正式分析区间并删除所需列空值
  → analyze_factor(momentum_score)
  → analyze_factor(momentum_score_halflife)
```

两个回测使用完全相同的股票样本、收益、基准、分组数和窗口参数，保证差异只来自权重与 R² 口径。

## 异常与边界处理

- 窗口长度小于 2、半衰期不大于 0、未知权重模式时立即抛出 `ValueError`。
- 输入价格必须是一维数组；维度不正确时抛出 `ValueError`。
- 不足 26 条记录不会填充或缩短窗口，因子保持空值。
- 常数价格不会产生除零或 NaN，年化收益、R² 和得分均为 0。
- 不对极端得分做截面缩尾或中性化，因为目标定义未要求这些处理；`analyze_factor` 直接测试原始综合得分。

## 测试与验收

先新增失败测试，再写 Notebook 实现。测试覆盖：

1. 目标 Notebook 的普通 Python 代码在移除 IPython 魔法行后可以编译。
2. 固定的非线性 26 日价格序列得到预先核对的示例版年化收益、R² 和得分，防止误改示例权重语义。
3. 完全指数增长价格在两个版本中均得到已知斜率、`R² = 1` 和对应年化收益。
4. 常数价格得到年化收益、R²、得分全为 0。
5. 63 日半衰期权重满足最新权重为 1、最早权重为 `0.5 ** (25 / 63)`，并使用标准加权 R²。
6. 少于 26 条记录或窗口中含无效价格时结果为空；满 26 条有效记录后才产生首个因子值。
7. 合成多股票长表保持代码、日期和行数对齐，不发生跨股票滚动污染。

实现后使用 `E:\working\anaconda3\envs\quant\python.exe` 运行定向 pytest，并使用 nbformat 校验 Notebook 结构。环境和本地行情允许时，再用 `jupyter nbconvert --execute --to notebook --inplace` 自顶向下执行目标 Notebook；若全市场执行受资源限制，必须报告实际停止位置与可复现命令。
