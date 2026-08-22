# Alpha191 因子库 - 本地化适配

## 项目概述

将 WorldQuant 101 Alpha 因子和国泰君安 Alpha191 因子公式适配到本项目的本地数据系统和回测框架。支持 191 个因子的计算、IC/IR 分析和批量回测。

**文档速查：**
- [快速开始](#快速开始) —— 两分钟上手
- [如何查看项目](#如何查看项目) —— 项目结构和使用入口
- [模块说明](#模块说明) —— 各文件职责
- [已知限制](#已知限制) —— 未实现/需行业数据的因子
- [数据流](#数据流) —— 数据如何加载到因子计算
- [批量回测](#批量回测) —— 运行所有因子的回测

---

## 快速开始

```python
from 因子回测.alpha_191 import Alpha191Calculator

# 1. 初始化计算器
calc = Alpha191Calculator()
calc.load_data('2024-01-01', '2025-07-01')

# 2. 计算单个因子（最新日横截面值）
alpha_5 = calc.compute(5)          # pd.Series(index=股票代码, values=因子值)

# 3. 计算单个因子（全时段宽表）
alpha_5_df = calc.compute_df(5)    # pd.DataFrame(index=日期, columns=股票代码)

# 4. IC 分析（使用 因子回测/alpha.py 的 analyze_ic）
ic_result = calc.analyze_ic(5, return_periods=[1, 5, 10])

# 5. 极简版因子分析（宽表直接计算）
result = calc.analyze_factor(5, return_period=5, group_num=5)

# 6. 批量计算
results = calc.compute_all([1, 5, 10, 20])
```

---

## 如何查看项目

### 项目入口

| 入口 | 用途 | 推荐用户 |
|------|------|----------|
| [calculator.py](calculator.py) | 高层接口：数据加载→因子计算→回测分析 | 日常使用 ✅ |
| [batch_backtest.py](batch_backtest.py) | 批量回测所有 191 个因子，生成 HTML 报告 | 全量验证 |
| [study_alpha191.ipynb](study_alpha191.ipynb) | Jupyter Notebook 交互式研究 | 探索分析 |

### 数据流

```
┌─────────────────────────────────────────────────────────────┐
│  adapter.py                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ my_utils.fun │───→│ 本地 Parquet │───→│  宽表转换     │  │
│  │ read_day_data│    │ 日线数据     │    │ pivot_table   │  │
│  └──────────────┘    └──────────────┘    └───────┬───────┘  │
│  ┌──────────────┐                                │          │
│  │ 米筐 API 补充 │←───（缺字段时自动拉起）        │          │
│  │ (市值/行业)  │                                │          │
│  └──────────────┘                                │          │
└───────────────────────────────────┬──────────────┘          │
                                    │                         │
                                    ▼                         │
┌──────────────────────────────────────────────────┐          │
│  alpha_formulas.py / Alpha191Formulas             │          │
│  接收 data_dict → 提供 alpha_NNN() 和 alpha_NNN_df() │        │
│  191 个因子公式（约 176 个可直接计算）              │          │
└───────────────────────┬──────────────────────────┘          │
                        │                                     │
                        ▼                                     │
┌──────────────────────────────────────────────────┐          │
│  calculator.py / Alpha191Calculator               │          │
│  整合：load_data → compute → analyze_ic/analyze_factor │      │
└──────────────────────────────────────────────────┘          │
                                                               │
┌──────────────────────────────────────────────────┐          │
│  因子回测/alpha.py                                │          │
│  analyze_ic() / analyze_factor() ← 复用本地回测框架│         │
└──────────────────────────────────────────────────┘          │
```

### 文件说明

| 文件 | 职责 | 核心类/函数 |
|------|------|------------|
| [adapter.py](adapter.py) | 数据适配：从本地 Parquet + 米筐 API 读取数据，转换为宽表格式 | `LocalDataAdapter`, `load_factor_data()` |
| [alpha_formulas.py](alpha_formulas.py) | 191 个 Alpha 因子的公式计算（完整版） | `Alpha191Formulas` |
| [calculator.py](calculator.py) | 高层接口：整合数据加载、因子计算和本地回测分析 | `Alpha191Calculator` |
| [batch_backtest.py](batch_backtest.py) | 批量回测脚本：对所有因子计算 IC/IR，生成 HTML 报告并通过邮件发送 | `compute_ic()` |
| [batch_results.json](batch_results.json) | 批量回测结果（检查点，可断点续跑） | JSON 文件 |
| [alpha191_backtest_report.html](alpha191_backtest_report.html) | 批量回测 HTML 报告（成功/失败/IC分布/Top5） | HTML 报告 |
| [__init__.py](__init__.py) | 包入口，导出主要类和函数 | `Alpha191Calculator`, `Alpha191Formulas` |
| [study_alpha191.ipynb](study_alpha191.ipynb) | Jupyter Notebook 交互式演示 | 教程/示例 |
| [101 Formulaic Alphas.pdf](101%20Formulaic%20Alphas.pdf) | WorldQuant 101 Alpha 论文原版 PDF | 参考 |
| [国泰君安－基于短周期价量特征的多因子选股体系.pdf](国泰君安－基于短周期价量特征的多因子选股体系.pdf) | 国泰君安 Alpha191 研报 PDF | 参考 |

---

## 模块说明

### adapter.py — 数据适配器

负责从本地数据系统读取股票日线数据，转换为因子计算所需的宽表格式。

**数据来源优先级：**
1. 本地 Parquet 文件（`my_utils.fun.read_day_data()`）
2. 米筐 API 代理（本地缺字段时自动拉起，如市值、行业等）

**增强字段：**
- 基础价量：`open, high, low, close, pre_close, volume, amount`
- 衍生字段：`vwap, returns, adv5/10/20/30/40/50/60/120/180`
- 市值字段：`total_mv, circulation_mv, mv_A_free_float`（需本地 parquet 支持）
- 行业分类：`industry`（需调用 `load_with_industry()`）

**复用项目接口：**
| 接口 | 来源 |
|------|------|
| `read_day_data()` | `my_utils.fun` ✅ |
| `convert_code_format()` | `my_utils.mapping` ✅ |
| `stock_api.get_industry_list()` | `my_utils.stock_api` ✅ |
| `RQData.get_rq_data()` | `my_utils.rqdata` ✅ |

### alpha_formulas.py — 因子公式计算

提供 191 个 Alpha 因子公式的完整实现。

**计算模式：**
- `alpha_NNN(self)` → 返回最新日横截面 `pd.Series`
- `alpha_NNN_df(self)` → 返回全时段 `pd.DataFrame`

**辅助函数：**

| 函数 | 说明 |
|------|------|
| `ts_sum(df, window)` | 滚动求和 |
| `sma(df, window)` | 滚动移动平均 |
| `stddev(df, window)` | 滚动标准差 |
| `correlation(x, y, window)` | 滚动相关系数 |
| `covariance(x, y, window)` | 滚动协方差 |
| `ts_rank(df, window)` | 滚动时间序列排名 |
| `ts_argmax(df, window)` | 滚动 argmax |
| `ts_argmin(df, window)` | 滚动 argmin |
| `ts_min(df, window)` | 滚动最小值 |
| `ts_max(df, window)` | 滚动最大值 |
| `rank(df)` | 横截面百分比排名 |
| `delta(df, period)` | 差分算子 |
| `delay(df, period)` | 延迟算子 |
| `scale(df, k)` | 缩放算子 |
| `decay_linear(df, period)` | 线性加权移动平均 |
| `signed_power(df, power)` | 带符号幂函数 |
| `product(df, window)` | 滚动乘积 |

### calculator.py — 高层计算器

整合数据加载、因子计算和本地因子分析接口的统一入口。

**核心方法：**
- `load_data(start_date, end_date)` — 加载数据
- `compute(alpha_num)` — 计算单个因子最新日值
- `compute_df(alpha_num)` — 计算单个因子全时段值
- `compute_all(alpha_list)` — 批量计算多个因子
- `analyze_ic(alpha_num)` — 复用 `因子回测/alpha.analyze_ic()` 做 IC 分析
- `analyze_factor(alpha_num)` — 复用 `因子回测/alpha.analyze_factor()` 做因子分析

### batch_backtest.py — 批量回测脚本

对所有因子（1-191）逐一调用本地 `因子回测.alpha.analyze_factor()`。批量脚本只做
宽表转长表、公式调度与 HTML 组合；IC/RankIC、分组回测、净值、图表和绩效指标均由
统一因子框架返回。

**特性：**
- 默认运行全部 191 个因子，不再按 SLOW 名单静默跳过
- 报告按框架返回的 `rank_ic_mean` 降序，逐因子展示 IC/RankIC 表、分组统计和三类框架图
- 公式、行业分类或数据无法复现时，仍保留该因子的报告条目并说明原因
- 默认仅生成 `alpha191_backtest_report.html`；只有显式传入 `--send-email --receiver 邮箱` 才发送邮件

```powershell
E:\working\anaconda3\envs\quant\python.exe 因子回测\alpha_191\batch_backtest.py

# 只验证部分因子，便于排查数据或公式
E:\working\anaconda3\envs\quant\python.exe 因子回测\alpha_191\batch_backtest.py --alphas 1,5,10
```

---

## 回测结果解读

### 报告位置

- HTML 报告：`因子回测/alpha_191/alpha191_backtest_report.html`
- 报告内的图表以 Base64 内嵌，不依赖额外 PNG 文件
- 旧版 `batch_results.json` 是历史自算 IC 结果，不参与当前 `analyze_factor` 口径的批量报告

### 指标说明

| 指标 | 说明 | 参考标准 |
|------|------|----------|
| IC | 因子值与未来收益的 Pearson 相关系数 | \|IC\|>0.02 为有效 |
| IC_IR | IC 均值/IC 标准差，衡量稳定性 | >0.5 较好，>1.0 优秀 |
| RankIC | 因子排名与收益排名的 Spearman 相关系数 | 比 IC 更稳健 |
| IC>0% | IC 为正的天数占比 | >55% 为正向稳定 |
| 耗时 | 单个因子计算时间 | - |

### 已知结果（2021-01 ~ 2025-07, Top300 股票, 5日持仓）

见 `alpha191_backtest_report.html`。优胜因子集中在价量相关性类和趋势类。

---

## 已知限制

### 需行业数据的因子（19 个）

以下因子需调用 `load_factor_data_with_industry()` 提供 `industry_map` 才能计算：

| Alpha | 说明 | 当前状态 |
|-------|------|----------|
| 48, 56, 58, 59, 63, 67 | 含 IndNeutralize 的公式 | ✅ 有实现，需 industry |
| 87, 89, 90, 91, 93, 97, 100 | 需要 IndClass.industry/subindustry | ❌ 未实现 |
| 69, 70, 76, 79, 80, 82 | 复杂多层 IndNeutralize | ❌ 未实现 |

### 慢因子（已标记 SLOW，约 30 个）

计算窗口大（>180 天）或涉及复杂多层嵌套的因子已在 `batch_backtest.py` 中标记为 SLOW，默认跳过以节省时间。可修改 `SLOW` 集合取消跳过。

### 已修复的 Bug（2025-07-25）

| 问题 | 涉及 Alpha | 修复内容 |
|------|-----------|---------|
| `_df` 方法返回 Series 而非 DataFrame | 71, 73, 77, 88, 92, 96 | `pd.concat→max/min` 替换为 `np.maximum/np.minimum` |
| `pct_change()` FutureWarning | adapter.py | 添加 `fill_method=None` 参数 |
| polars `is_in` deprecation | adapter.py | 替换为 `semi join` |
| `alpha_082_df` 位置错乱 | alpha_formulas.py | 整理代码位置 |

---

## 依赖关系

```
因子回测/alpha_191/
├── adapter.py ───────────→ my_utils.fun.read_day_data
│                           my_utils.mapping.convert_code_format
│                           my_utils.stock_api.stock_api
│                           my_utils.rqdata.RQData
├── calculator.py ────────→ 因子回测/alpha.analyze_ic
│                           因子回测/alpha.analyze_factor
├── batch_backtest.py ────→ my_utils.email_fun.send_email
│                           因子回测/alpha_191.adapter.load_factor_data
│                           因子回测/alpha_191.alpha_formulas.Alpha191Formulas
└── alpha_formulas.py ────→ （纯公式计算，无项目依赖）
```

---

## 批量回测

```bash
# 在项目根目录运行
cd 策略/
E:/working/anaconda3/envs/quant/python.exe -m 因子回测.alpha_191.batch_backtest
```

这会：
1. 加载 2021-01-01 至 2025-07-01 的全量数据
2. 选取 TOP-300 股票
3. 逐个计算 191 个因子的 IC/IR
4. 生成 `alpha191_backtest_report.html` 报告
5. 通过邮件发送报告到 `2056123357@qq.com`
