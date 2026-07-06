# 米筐分钟线 DDB 月度合成设计

日期：2026-07-06

## 背景

项目本地已有 `rq_stock_all_data`、`rq_adj`、`gm_stock_all_data` 的 2018-2021 日线数据补齐结果。GM 15 分钟历史补数受账号权限限制，普通权限只能拉取最近 180 个自然日分钟 Bar，无法用于 2018-2020 的历史分钟补齐。

DDB 中已有米筐来源的 `dfs://common_years_olap.one_min_kline` 一分钟线，字段为：

```text
order_book_id, trade_time, open, close, high, low, volume, total_turnover
```

用户确认：米筐分钟数据只需要右对齐口径，写入并覆盖 `rq_15min_stock_data_dir`，不需要保留左对齐版本。

## 目标

1. 在 `任务/米筐数据更新.py` 中支持从 DDB 一分钟线合成米筐 15 分钟线。
2. 查询频率限制为“每月一次分钟线查询”，避免按日查询压垮 DDB。
3. 输出口径对齐当前 GM 右对齐 15 分钟数据：
   - 每只股票每个完整交易日 18 根 Bar。
   - 时间戳为 `09:30, 09:45, ..., 11:30, 13:00, 13:15, ..., 15:00`。
   - `09:30`、`13:00` 是开盘 snapshot Bar。
4. `mode=update` 时覆盖目标日期范围内 `rq_15min_stock_data_dir` 的旧分区。
5. 先用一天样本和现有 GM 右对齐数据验证口径，再执行历史范围补数。

## 非目标

1. 不补 GM 15 分钟历史目录。
2. 不生成左对齐米筐分钟线。
3. 不使用 tick 表 `dfs://common_day_hash.stock_tick`。
4. 不做一次性多年分钟线全量查询。
5. 不改变已有日线和复权数据写入流程。

## 输出目录与 Schema

输出目录固定为：

```text
E:\working\stock_data\rq_15min_stock_data_dir
```

输出字段沿用 `my_utils.rq_fun.RQ_MIN_SCHEMA`：

```text
code: String
datetime: Datetime(us)
open: Float64
high: Float64
low: Float64
close: Float64
volume: Float64
trading_date: Date
```

代码格式转换：

```text
000001.XSHE -> SZSE.000001
600000.XSHG -> SHSE.600000
```

## 15 分钟右对齐口径

DDB 一分钟线 `trade_time` 是分钟 Bar 的结束时间。合成时按交易时段分别处理：

上午：

```text
09:31-09:45 -> 09:45
09:46-10:00 -> 10:00
...
11:16-11:30 -> 11:30
```

下午：

```text
13:01-13:15 -> 13:15
13:16-13:30 -> 13:30
...
14:46-15:00 -> 15:00
```

OHLCV 聚合规则：

```text
open   = bucket 内第一根 1min open
high   = bucket 内 high 最大值
low    = bucket 内 low 最小值
close  = bucket 内最后一根 1min close
volume = bucket 内 volume 求和
```

snapshot 规则对齐 `my_utils.stock_api.stock_api.gm_get_minute_data(align="right")`：

```text
09:30 snapshot 复制 09:45 完整 Bar 的 volume，
OHLC 全部设为 09:45 完整 Bar 的 open。

13:00 snapshot 复制 13:15 完整 Bar 的 volume，
OHLC 全部设为 13:15 完整 Bar 的 open。
```

这会使完整交易日每只股票输出 18 根 Bar。若某只股票某日缺少第一根完整 Bar，则不补对应 snapshot，避免凭空造数。

## DDB 查询策略

新增月度批次生成函数，按自然月切分用户指定日期范围。每个月只执行一次 `one_min_kline` 查询。

单月查询范围：

```text
trade_time >= timestamp(month_start 09:31:00)
trade_time <= timestamp(month_end 15:00:00)
```

查询条件：

1. 只查 `order_book_id like '%.XSHE' or order_book_id like '%.XSHG'`。
2. 查询返回后按 `instrument_base(type='CS')` 股票池过滤，剔除指数、基金、债券等非股票代码。
3. 只取合成所需字段，避免拉取无关列。

说明：如果后续实测单月全市场 1min 数据仍然过大，再增加只针对交易日范围的更细筛选或按半月降级。第一版遵守用户要求，以“每月一次”为默认且唯一自动执行策略。

## 写入流程

新增分钟更新流程与日线流程分离：

1. 获取交易日列表。
2. 生成月度批次。
3. 获取股票池一次并复用。
4. 对每个月：
   - 拉取该月全市场 1min 数据。
   - 本地合成右对齐 15min。
   - 仅保留用户指定日期范围内的交易日。
   - `mode=update` 时，先删除该月实际输出日期对应的旧分区，再写入新分区。
   - `mode=insert` 时，只写不存在的日期分区。
5. 记录每个月查询次数、输入行数、输出行数、输出交易日数。

写入目标仍使用已有 `write_partitioned`，保持 schema 对齐和分区结构一致。

## CLI 设计

为 `任务/米筐数据更新.py` 增加数据类型选择参数：

```text
--data-type day | min | all
```

默认值为 `day`，保持当前日常更新行为不变。

用法示例：

```powershell
E:\working\anaconda3\envs\quant\python.exe 任务\米筐数据更新.py `
  --start-date 2018-01-01 `
  --end-date 2020-12-31 `
  --mode update `
  --data-type min
```

`--batch-mode` 和 `--batch-size` 只作用于日线流程。分钟线固定按自然月批次，避免误配置成按日查询。

## 验证计划

### 单元测试

新增测试覆盖：

1. 构造一只股票一天 240 根 1min 数据，验证合成后时间戳为 18 根。
2. 验证 `09:30` 和 `13:00` snapshot 的 OHLC 全等于对应第一根完整 15min Bar 的 open，volume 复制对应完整 Bar。
3. 验证 `09:45`、`13:15` 等完整 Bar 的 OHLCV 聚合规则。
4. 验证自然月批次生成不会按日切分。

### 样本对齐测试

执行一次极小范围 DDB 查询：

```text
order_book_id = 000001.XSHE
trade_date = 2021-01-04
```

将 DDB 1min 合成结果与本地 GM 右对齐样本：

```text
E:\working\stock_data\15min_stock_data_right_dir\trading_date=2021-01-04
```

对比内容：

1. 时间戳集合。
2. `SZSE.000001` 的 18 根 OHLCV。
3. 行数是否为 18。

样本对齐通过后，再运行目标历史月份补数。

### 补数后审计

对 `rq_15min_stock_data_dir` 统计：

1. 分区日期最小值、最大值、交易日数量。
2. 总行数、股票数。
3. 每个完整交易日每只股票最多 18 根。
4. 2018-2020 是否有目标分区。

## 风险与保护

1. DDB 单月全市场 1min 数据量大：只允许按月执行，不做并发多月查询。
2. DDB 查询失败：该月不删除旧分区，不写半成品。
3. 合成结果为空：不删除旧分区。
4. schema 不匹配：写入前 cast 到 `RQ_MIN_SCHEMA`。
5. 口径偏差：必须先通过一天样本对齐测试。

## 实施边界

本设计只修改：

```text
任务/米筐数据更新.py
tests/test_historical_backfill_scripts.py
```

如果测试过程中发现 `my_utils.rq_fun.write_partitioned` 的行为不足以支持分钟覆盖，再以最小兼容方式扩展现有函数，不新增平行写入框架。
