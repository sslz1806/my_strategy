# DolphinDB 数据库表信息汇总

> 生成时间：2026-05-30  
> 探索范围：只读连接 DolphinDB，整理 DFS 数据库、表名、字段结构与少量已查询到的表级统计。  
> 重要说明：这是公司数据库，后续排查应优先只查 `getClusterDFSDatabases()`、`getTables()`、`schema(loadTable(...))` 等元信息，避免对大表执行全表 `count`、`count(distinct ...)`、`min/max` 或 `select *`。

## 本次查询说明

本次连接信息来自项目现有脚本：

- `任务/米筐数据更新.py`
- `my_utils/rqdata.py`

使用环境：

- Python：`E:\working\anaconda3\envs\quant\python.exe`
- DolphinDB Python 包版本：`3.0.4.2`
- DDB 地址：`10.140.5.44:8902`

本次确实做过部分高频/重查询，尤其尝试对 `dfs://common_day_hash.stock_tick` 做全表统计后，DDB 连接开始被服务端断开。因此本文档后续建议严格限制为元数据查询，不再探查具体数据内容。

## 数据库总览

当前可见 DFS 数据库共 14 个：

| 数据库 | 表数量 | 主要内容 |
|---|---:|---|
| `dfs://account_years_tsdb` | 17 | 账户、产品、基金净值、持仓、成交、滑点、行业/指数分布 |
| `dfs://autoInspection` | 7 | 自动巡检计划、报告、邮件发送历史 |
| `dfs://common_day_hash` | 1 | 股票 tick |
| `dfs://common_years_future_olap` | 2 | 期货日线、分钟线 |
| `dfs://common_years_future_tsdb` | 1 | 期货基础信息 |
| `dfs://common_years_olap` | 5 | 股票/指数日线、分钟线、写入日志 |
| `dfs://common_years_options_olap` | 3 | 期权日线、分钟线、tick |
| `dfs://common_years_options_tsdb` | 1 | 期权基础信息 |
| `dfs://common_years_tsdb` | 5 | 证券基础信息、交易日历、行业、港股相关表 |
| `dfs://rq_factor_years_tsdb` | 1 | 米筐因子日频数据 |
| `dfs://stock_years_tsdb` | 12 | 股票股本、ST、复权、财务、股东户数、陆股通、指数权重等 |
| `dfs://system_day_tsdb` | 1 | 系统信息 |
| `dfs://ths_factor_years_tsdb` | 1 | 同花顺因子日频数据 |
| `dfs://web_tsdb` | 0 | 空库 |
| `dfs://wind_factor_years_tsdb` | 1 | Wind 因子日频数据 |

## 核心 A 股行情库

### `dfs://common_years_olap`

| 表名 | 引擎 | 分区 | 字段数 | 说明 |
|---|---|---|---:|---|
| `day_kline` | OLAP | `date` RANGE | 20 | 日线行情，项目当前主要使用表 |
| `one_min_kline` | OLAP | `trade_time` RANGE | 8 | 1 分钟线，项目中用于聚合 15 分钟线 |
| `min_30_kline` | OLAP | `trade_time` RANGE | 8 | 30 分钟线，当前已查到为 0 行 |
| `insert_table` | OLAP | `update_time` RANGE | 3 | 插入记录 |
| `write_logs` | OLAP | `date` RANGE | 6 | 写入日志 |

`day_kline` 字段：

```text
order_book_id, date, open, close, high, low, volume, total_turnover,
prev_close, limit_up, limit_down, num_trades, open_interest,
settlement, prev_settlement, dominant_id, strike_price,
contract_multiplier, day_session_open, iopv
```

`one_min_kline` / `min_30_kline` 字段：

```text
order_book_id, trade_time, open, close, high, low, volume, total_turnover
```

已查询到的表级统计：

| 表名 | 行数 | 时间范围 |
|---|---:|---|
| `day_kline` | 22,747,864 | 2018-01-01 至 2026-05-29 |
| `one_min_kline` | 5,051,567,292 | 2005-01-04 09:31 至 2026-05-29 16:00 |
| `min_30_kline` | 0 | 无 |

### `dfs://common_day_hash`

| 表名 | 引擎 | 分区 | 字段数 | 说明 |
|---|---|---|---:|---|
| `stock_tick` | TSDB | `date, order_book_id` RANGE,HASH | 20 | 股票 tick 数据，重表 |

`stock_tick` 字段：

```text
date, exchange, order_book_id, trade_time, price, volume, amount,
open, high, low, pre_close, total_volume, total_turnover,
total_num_trades, ask_prices, ask_qtys, bid_prices, bid_qtys,
total_ask_nums, total_bid_nums
```

注意：本表非常重，不建议执行全表统计。后续只应查询 `schema` 或按具体日期/股票做极小范围抽样。

## 证券基础信息与交易日历

### `dfs://common_years_tsdb`

| 表名 | 引擎 | 分区 | 字段数 | 说明 |
|---|---|---|---:|---|
| `instrument_base` | TSDB | `listed_date` RANGE | 6 | A 股/证券基础信息 |
| `trade_date` | TSDB | `date` RANGE | 5 | A 股交易日历 |
| `industry` | TSDB | 无 | 4 | 行业映射 |
| `instrument_hk` | TSDB | `listed_date` RANGE | 15 | 港股基础信息 |
| `trade_date_hk` | TSDB | `date` RANGE | 5 | 港股交易日历 |

核心字段：

```text
instrument_base:
order_book_id, symbol, abbrev_symbol, type, listed_date, de_listed_date

trade_date:
date, trade_date, next_trade_date, last_trade_date, is_trade_date

industry:
date, mode, cs_industry, industry
```

已查询到的表级统计：

| 表名 | 行数 | 时间范围 |
|---|---:|---|
| `instrument_base` | 238,502 | listed_date：1990-12-01 至 2026-05-27 |
| `trade_date` | 9,862 | date：2000-01-01 至 2026-12-31 |
| `industry` | 61 | date：2024-08-02 |
| `instrument_hk` | 0 | 无 |

## 股票扩展、财务与事件库

### `dfs://stock_years_tsdb`

| 表名 | 引擎 | 分区 | 字段数 | 说明 |
|---|---|---|---:|---|
| `stock_shares` | TSDB | `date` RANGE | 6 | 股本数据 |
| `is_st_stock` | TSDB | `date` RANGE | 3 | ST 标记 |
| `ex_factor` | TSDB | `ex_date` RANGE | 6 | 复权因子 |
| `dividend` | TSDB | `date` RANGE | 6 | 分红 |
| `stock_pit` | TSDB | `quarter_date` RANGE | 6 | PIT 财务索引 |
| `stock_pit_financials` | TSDB | `date` RANGE | 7 | PIT 财务字段 |
| `holder_number` | TSDB | `info_date` RANGE | 8 | 股东户数 |
| `stock_connect` | TSDB | `date` RANGE | 5 | 陆股通持股 |
| `index_weights_ex` | TSDB | `date` RANGE | 4 | 指数成分权重 |
| `factor_returns` | TSDB | `date` RANGE | 3 | 风险因子收益 |
| `risk_covs` | TSDB | `date` RANGE | 4 | 风险协方差 |
| `shibor` | TSDB | `date` RANGE | 3 | Shibor |

核心字段摘录：

```text
stock_shares:
date, order_book_id, circulation_a, non_circulation_a, total_a, free_circulation

is_st_stock:
date, order_book_id, is_st

ex_factor:
ex_date, order_book_id, ex_factor, ex_cum_factor, announcement_date, ex_end_date

stock_pit_financials:
date, order_book_id, quarter, info_date, revenue, if_adjusted, net_profit

index_weights_ex:
date, order_book_id, part_order_book_id, weight
```

已查询到的表级统计：

| 表名 | 行数 | 时间范围 |
|---|---:|---|
| `stock_shares` | 18,043,281 | 2000-01-04 至 2026-05-29 |
| `is_st_stock` | 34,862,968 | 2000-01-04 至 2026-05-29 |
| `ex_factor` | 53,147 | 2000-01-05 至 2026-05-29 |
| `dividend` | 145,232 | 1991-03-03 至 2025-12-15 |
| `stock_pit` | 258,974 | 2016-03-31 至 2025-12-31 |
| `stock_pit_financials` | 115,674 | 2020-04-08 至 2026-05-27 |
| `holder_number` | 425,794 | 2000-01-08 至 2026-05-28 |
| `stock_connect` | 5,049,316 | 2017-03-17 至 2025-03-31 |
| `index_weights_ex` | 12,719,151 | 2005-01-31 至 2026-05-28 |
| `factor_returns` | 95,760 | 2017-01-03 至 2026-05-28 |
| `risk_covs` | 4,021,920 | 2017-01-03 至 2026-05-28 |
| `shibor` | 39,216 | 2006-10-08 至 2026-05-28 |

## 因子库

三个因子库结构一致：

| 数据库 | 表名 | 引擎 | 分区 | 字段 |
|---|---|---|---|---|
| `dfs://rq_factor_years_tsdb` | `day_factor` | TSDB | `date, factor` RANGE,VALUE | `date, symbol, factor, value` |
| `dfs://ths_factor_years_tsdb` | `day_factor` | TSDB | `date, factor` RANGE,VALUE | `date, symbol, factor, value` |
| `dfs://wind_factor_years_tsdb` | `day_factor` | TSDB | `date, factor` RANGE,VALUE | `date, symbol, factor, value` |

推测用途：

- `rq_factor_years_tsdb`：米筐因子
- `ths_factor_years_tsdb`：同花顺因子
- `wind_factor_years_tsdb`：Wind 因子

本次没有继续查询因子明细、因子名称列表或行数，避免对公司库造成压力。

## 期货与期权库

### 期货

| 数据库 | 表名 | 引擎 | 分区 | 字段数 | 说明 |
|---|---|---|---|---:|---|
| `dfs://common_years_future_olap` | `day_kline` | OLAP | `date` RANGE | 15 | 期货日线 |
| `dfs://common_years_future_olap` | `one_min_kline` | OLAP | `date` RANGE | 9 | 期货分钟线 |
| `dfs://common_years_future_tsdb` | `instrument_base` | TSDB | `listed_date` RANGE | 19 | 期货基础信息 |

### 期权

| 数据库 | 表名 | 引擎 | 分区 | 字段数 | 说明 |
|---|---|---|---|---:|---|
| `dfs://common_years_options_olap` | `day_kline` | OLAP | `date` RANGE | 17 | 期权日线 |
| `dfs://common_years_options_olap` | `one_min_kline` | OLAP | `date` RANGE | 9 | 期权分钟线 |
| `dfs://common_years_options_olap` | `tick` | OLAP | `date` RANGE | 19 | 期权 tick |
| `dfs://common_years_options_tsdb` | `instrument_base` | TSDB | `de_listed_date` RANGE | 5 | 期权基础信息 |

本次只整理了表结构，没有继续做行数和时间范围统计。

## 账户、产品与组合数据

### `dfs://account_years_tsdb`

| 表名 | 字段数 | 说明 |
|---|---:|---|
| `asset_order` | 18 | 账户委托 |
| `asset_position` | 17 | 账户持仓 |
| `asset_trade` | 17 | 账户成交 |
| `fund_asset` | 11 | 基金/产品资产 |
| `fund_barr_exposure` | 5 | Barra 暴露 |
| `fund_future_position` | 8 | 期货持仓 |
| `fund_index_distrubution` | 6 | 指数分布 |
| `fund_industry_distrubution` | 5 | 行业分布 |
| `fund_net` | 7 | 基金净值 |
| `fund_position_weight` | 5 | 持仓权重 |
| `fund_slippage` | 18 | 滑点与算法成交统计 |
| `fund_stats` | 7 | 组合统计 |
| `index_industry_distrubution` | 5 | 指数行业分布 |
| `lzq_option_vix` | 5 | 期权波动率/VIX 类数据 |
| `lzq_product_net` | 6 | 产品净值 |
| `product_cash` | 8 | 产品现金流水 |
| `product_net` | 6 | 产品净值 |

这些表均为 TSDB，主要按 `date` RANGE 分区。

## 自动巡检与系统库

### `dfs://autoInspection`

| 表名 | 说明 |
|---|---|
| `plans` | 巡检计划 |
| `planDetails` | 计划明细 |
| `reports` | 巡检报告 |
| `reportDetails` | 报告明细 |
| `metrics` | 巡检指标 |
| `emailHistory` | 邮件发送历史 |
| `updateHistory` | Web 版本更新历史 |

### `dfs://system_day_tsdb`

| 表名 | 字段 |
|---|---|
| `system_info` | `name, id, type, OLAPTable, OLAPCacheEngine, DFSMetadata, TSDBCacheEngine, TSDBLevelFileIndex, update_time` |

## 推荐的安全查询方式

以后如果只是看 DDB 里有什么数据，建议只执行以下轻量元信息查询：

```python
import dolphindb as ddb

s = ddb.session()
s.connect(host, port, user, password)

# 1. 查看 DFS 库
s.run("getClusterDFSDatabases()")

# 2. 查看某个库的表名
s.run('getTables(database("dfs://common_years_olap"))')

# 3. 查看某张表字段、分区、引擎等 schema
s.run('schema(loadTable("dfs://common_years_olap", "day_kline"))')

s.close()
```

应避免：

```sql
-- 大表上慎用/避免
select count(*) from loadTable(...)
select count(distinct order_book_id) from loadTable(...)
select min(date), max(date) from loadTable(...)
select * from loadTable(...)
```

如果必须看样本，建议明确限定小范围：

```sql
select top 5 *
from loadTable("dfs://common_years_olap", "day_kline")
where date = date(2026.05.29)
  and order_book_id = "000001.XSHE"
```

## 与当前项目关系

当前项目脚本已经使用的 DDB 表主要是：

| 项目用途 | DDB 表 |
|---|---|
| 股票池 | `dfs://common_years_tsdb.instrument_base` |
| 交易日 | `dfs://common_years_tsdb.trade_date` |
| 日线行情 | `dfs://common_years_olap.day_kline` |
| ST 标记 | `dfs://stock_years_tsdb.is_st_stock` |
| 股本/市值计算 | `dfs://stock_years_tsdb.stock_shares` |
| 复权因子 | `dfs://stock_years_tsdb.ex_factor` |
| 1 分钟线聚合 15 分钟线 | `dfs://common_years_olap.one_min_kline` |

这些表已经覆盖项目中米筐本地数据更新的核心需求。
