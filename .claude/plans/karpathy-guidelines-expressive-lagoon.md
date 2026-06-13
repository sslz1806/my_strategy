# 米筐数据更新脚本重构计划

## Context

当前 `任务/米筐数据更新.py` 混合了两种更新模式：
- **按天更新**（日线）：每天循环，获取当天股票池并拉取数据
- **按股票分批更新**（分钟线/复权因子）：预获取股票池，再按股票分批拉取

用户要求统一为**按天模式**，删除按股票分批的代码，让脚本干净简洁。

## 改动范围（本次仅修改数据更新脚本）

| 文件 | 操作 |
|------|------|
| `任务/米筐数据更新.py` | **重写** — 删除按股票更新逻辑，统一按天循环 |
| `my_utils/rq_fun.py` | **删除** `split_batches()`（仅被 `batch_update` 使用） |

策略文件（`回测demo.ipynb`）和交易函数（`trade_fun.py`）本次不动。

## 步骤

### Step 1: 重构 `任务/米筐数据更新.py`

**删除的代码**（4个函数 + main 中的预获取逻辑）：

| 删除项 | 行号 | 原因 |
|--------|------|------|
| `_probe_trading_days()` | 第229-251行 | 用少量股票探测交易日，脆弱；改为从已有本地数据获取交易日 |
| `_fetch_adj_single()` | 第143-164行 | 单天除权事件拉取 adj_factor 不准确，整合到通用按天拉取逻辑 |
| `fetch_adj_data()` | 第280-302行 | 按股票分批拉复权因子，属于按股票更新模式 |
| `fetch_minute_batch()` | 第259-277行 | 按股票分批拉分钟线，替换为按天版本 |
| `batch_update()` | 第310-346行 | **核心删除项** — 按股票批量的通用更新函数 |
| `update_day_data_rq()` | 第172-221行 | 不再独立存在，合并到统一更新函数 |
| main() 中 end_pool 预获取 | 第389-398行 | "都说了按天更新数据，还获取股票池干什么" |
| `split_batches` / `ensure_rq_codes` 导入 | 第14行 | 不再使用 |

**保留的代码**：

| 保留项 | 理由 |
|--------|------|
| `get_stock_universe()` | 获取当天全市场股票池，按天模式的核心函数 |
| `fetch_day_full()` | 按天拉取日线数据，模式正确，仅签名微调 |
| `normalize_day_data()` 等 rq_fun 函数 | 数据处理层，与更新模式无关 |
| `parse_args()` | CLI 参数解析，仅简化参数列表 |
| `write_partitioned()` | 分区写入函数，保持不变 |

**新增的代码**：

1. **`get_trading_days(api, start_date, end_date) -> list[dt.date]`**
   - 优先尝试 RQ API 获取交易日（如 `get_rq_data("get_trading_dates", ...)` 或 `get_rq_data("get_calendar", ...)`）
   - API 不支持时，用单只稳定股票（`000001.XSHE`）的 `get_price` 探测交易日
   - 不依赖 GM/TS 等本地数据

2. **`fetch_minute_full(api, rq_codes, trade_date) -> pl.DataFrame`**
   - 按天拉取分钟线。传入当天全市场股票池 + 单天日期，拉取15分钟线

3. **`update_all(api, start_date, end_date, mode)` — 统一更新入口**
   ```
   交易日列表 = get_trading_days(api, start_date, end_date)
   循环每个交易日:
     股票池 = get_stock_universe(api, trade_date)  # 每天获取，不预取
     日线 = fetch_day_full(api, 股票池, trade_date)  → 写入
     # 分钟 = fetch_minute_full(api, 股票池, trade_date)  → 写入  (测试阶段注释掉)
     复权因子 = 从日线中提取 adj_factor 列 → 写入 rq_adj
   ```

4. **简化 `main()`** — 移除 end_pool 预获取，直接调用 `update_all()`

参数简化：移除 `--batch-size`、`--codes` 参数。

**测试策略**：先注释掉分钟数据更新代码，只跑日线验证脚本功能正常，后续再放开。

### Step 2: 修改 `my_utils/rq_fun.py`

删除 `split_batches()`（第142-146行）—— 唯一调用者 `batch_update()` 将被删除。

### Step 3: 验证

1. 运行 `E:\working\anaconda3\envs\quant\python.exe 任务/米筐数据更新.py` 确认无报错
2. 检查 `rq_stock_all_data` / `rq_15min_stock_data_dir` / `rq_adj` 目录数据是否正常写入
3. 运行 `pytest tests/test_rq_update_data.py` 确认单元测试通过
