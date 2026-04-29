# CLAUDE.md - A股量化交易策略项目助手指南

> 本文档专为 AI 助手设计，帮助快速理解项目结构和工作方式

## 项目概述

这是一个完整的 A 股量化交易策略研究与实盘系统，用于：
- 股票数据管理与更新
- 策略信号生成
- 回测分析（理论框架 + 真实资金框架）
- 因子研究
- QMT 实盘交易

主要策略：**涨停低开策略**（在 `回测demo.ipynb` 中完整实现）

---

## 目录结构详解

| 目录 | 用途 |
|------|------|
| **my_utils/** | 核心接口函数包：数据处理、回测框架、API接口、可视化、实盘交易 |
| **根目录** | 主 Notebook（策略研究与演示） |
| **任务/** | 生产任务脚本 - 数据更新、QMT 实盘交易、邮件提醒 |
| **因子回测/** | 因子研究框架和回测工具 |
| **my_backtester/** | 真实资金回测框架 |
| **信号文件/** | 交易信号 CSV 文件 |
| **信号交割复盘/** | 交割单 HTML 复盘文件 |
| **old/** | 废弃文件夹（原根目录文件备份） |
| **log/** | 各类日志文件 |

---

## 核心模块说明

### my_utils/ 接口函数包

所有核心工具函数已打包到 `my_utils/` 目录，通过 `from my_utils import xxx` 或 `from my_utils.xxx import xxx` 导入：

| 模块 | 功能描述 |
|------|----------|
| **my_utils/fun.py** | 核心工具库：本地数据接口(polars)、polars特征计算函数(涨停/炸板/断板标记、几天几板描述、均线计算)、日志设置 |
| **my_utils/trade_fun.py** | 理论回测框架：自定义交易逻辑函数、风控函数、并行回测处理 |
| **my_utils/pd_fun.py** | Pandas 版本的特征函数：涨停/炸板/断板标记、几天几板描述、均线计算 |
| **my_utils/stock_api.py** | 外部数据源接口：Tushare、掘金、AkShare，用于数据更新 |
| **my_utils/stock_db.py** | MySQL 数据库接口：连接池、数据读写（支持 Polars）较少使用 |
| **my_utils/mapping.py** | 数据清洗与转换：列名映射、股票代码格式转换、日期处理 |
| **my_utils/my_qmt.py** | QMT 实盘交易接口：连接 miniQMT、下单、持仓查询、交易回调 |
| **my_utils/email_fun.py** | 邮件通知系统：QQ 邮箱 SMTP、支持 HTML 和附件 |
| **my_utils/stock_plot.py** | 股票数据可视化函数 |

### 任务目录 (任务/)

| 文件 | 功能描述 |
|------|----------|
| **数据更新.py** | 利用外部数据更新本地数据库。每日数据更新：日线、分钟线、复权因子、市场数据 |
| **实盘信号.py** | 实盘信号生成脚本 |
| **my_strategy_buy.py** | 实盘买入策略执行 |
| **my_strategy_sell.py** | 实盘卖出策略执行 |
| **run_update_data.bat** | 数据更新批处理脚本 |
| **strategy_email.bat** | 邮件提醒批处理脚本 |

### 因子回测目录 (因子回测/)

| 文件 | 功能描述 |
|------|----------|
| **alpha.py** | 因子分析库：IC 分析、OLS 中性化、未来收益计算 |
| **因子分析结果/** | 因子分析结果：IC 序列、分组收益、净值图、CSV 报告 |

### 回测框架目录 (my_backtester/)

| 文件 | 功能描述 |
|------|----------|
| **my_backtester.py** | 真实资金回测引擎：Backtester 类，支持 T+1、手续费、滑点 |

### 关键 Notebook

| Notebook | 用途 |
|----------|------|
| **回测demo.ipynb** | （信号驱动）完整策略演示：涨停低开策略全流程 |
| **因子回测框架.ipynb** | 因子回测框架 |
| **国九条小市值策略.ipynb** | 组合式/周度调仓/有状态策略的完整策略演示 |

---

## 开发约定

- **Python 版本**: 3.9+
- **数据处理库**: 优先使用 Polars，其次 Pandas
- **数据存储**: 本地 Parquet 文件（`E:\working\stock_data`，在 `fun.py` 中有读取接口）
- **编码风格**: 遵循 PEP 8
- **Notebook**: 用于策略研究，生产代码用 .py 文件
- **日志文件**: 必须放在对应文件夹的 `log/` 子目录中
  - 例如：`因子回测/小市值策略.py` 的日志 → `因子回测/log/小市值策略.log`
  - 例如：`任务/数据更新.py` 的日志 → `任务/log/数据更新.log`

## 用户偏好

- **回复语言**: 默认使用中文回复，除非用户明确要求使用其他语言。
- **代码注释**: 生成或修改代码时，需要补充清晰、详细、面向后续维护的中文注释；注释应解释关键业务逻辑、数据口径、边界条件和非显然实现原因，避免只复述代码表面含义。

---

## 协作硬约束（给 AI 助手｜请严格遵守）

1. **先读再改**  
   涉及具体代码、路径、配置、数据格式、接口返回等内容时，必须先读取对应文件/样本/真实输出再下结论。信息不足时不要猜，改为提 1-2 个关键问题收敛范围。

2. **尽量测试并汇报**  
   修改代码后，尽量运行最相关的测试或最小可行验证，并在回复里说明：怎么验证的、结果如何。若确实无法运行测试（缺环境/权限/外部依赖/无数据等），需要说明原因，并给出替代验证方式或下一步建议。

3. **优先复用既有函数接口（避免另起炉灶）**  
   实现需求时优先使用项目中已定义的函数/方法/模块，避免重复实现一套相似逻辑。若确实缺能力，优先扩展现有函数（增加可选参数/保持向后兼容）而不是新增平行函数。

---

## 标准回测工作流（项目记忆）

本项目回测分为两阶段流水线。写新策略时务必先判断策略类型，选择合适的起点，不要自作主张另起一套。

### Stage 1：理论等权回测（信号驱动）

适用于单股/短线类策略（如涨停低开、连板、N 字等），参考 `回测demo.ipynb`：

```
polars 特征链（mark_limit_status / add_sma / cal_n_lowest 等）
  → pl.when(条件).then(1).otherwise(0).alias("signal")
  → 信号文件 = stock_data.filter(pl.col("signal")==1)
  → result_df, merged_df = cal_trade_info(信号文件, trade_fun=trade, start_date, end_date)
  → report_backtest_full(merged_df.to_pandas(), profit_col='profit', ...)
```

- `trade_fun`：无状态、单信号粒度（一日一股独立买卖）
- 产出：buy_time/buy_price/sell_time/sell_price/profit/holding_days 等
- `report_backtest_full`：胜率/盈亏比/年化/超额等绩效指标

### Stage 2：真实资金回测（订单驱动）

适用于组合式/周度调仓/有状态策略（如小市值、国九条等），参考 `因子回测/国九条小市值策略.ipynb`：

```
polars 特征 → select_stocks(daily_data) 返回目标股票列表
  → 按 trading_days 循环，每 N 日调仓，维护 holdings 状态
  → 输出 orders_df (datetime, code, direction, price, volume/cash_ratio)
  → Backtester(orders_df).run(start_time, end_time)
  → backtester.report(start_date, end_date)
```

- 体现资金约束、100 股取整、手续费、滑点与同日买卖顺序等执行细节
- `backtester.report()` 已封装净值/回撤/基准对比三联图，不要自行重写绘图

