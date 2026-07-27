# 简化 build_daily_sentiment_factors

## Context

用户指出 `build_daily_sentiment_factors` 中 `completed_events` 这一步多余：
- 次日收益的兑现日对齐用 `shift(1)` 即可，不需要单独 group_by + join
- `up_ret_count` 和 `limit_up_count` 在实践中 >99.9% 一致（极少涨停次日停牌）
- 整个 factor pipeline 应该一次 group_by 搞定

## 改动文件

**因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb** — cell `46497cd1`，仅函数 `build_daily_sentiment_factors`

## 简化方案

### 当前流程

```
prepared
  → group_by(trading_date)       → daily_counts（1. 计数）
  → filter + group_by(next_market_date)  → completed_events（2. 收益聚合）
  → calendar.join(daily_counts).join(completed_events)  （3. 两步 join）
  → rolling_sum + rolling_max     → 6个因子
```

### 简化后流程

```
prepared
  → group_by(trading_date) + shift(1)  → 计数+收益（一次搞定）
  → calendar.join(daily_counts)        → 一次 join
  → rolling_sum + rolling_max          → 6个因子
```

### 具体改动

1. **砍掉 `completed_events`**：收益（sum of event_next_ret）直接跟计数放在同一个 `group_by("trading_date")` 里
2. **`shift(1)` 替代 `next_market_date` 对齐**：t 日事件的次日在 t+1，对收益列做 `shift(1)` 即可，效果等价于原版按 `next_market_date` 分组
3. **砍掉 `up_ret_count`**：直接用 `n_up.shift(1)` 做收益分母（二者在数据层面差不到 0.1%）
4. **`select()` 里直出因子**：不保留中间列，滚动窗口 + 除法在 `select()` 的表达式里完成

### 完整函数代码

```python
def build_daily_sentiment_factors(
    prepared: pl.DataFrame,
    calendar: pl.DataFrame,
    window: int = 5,
) -> pl.DataFrame:
    """
    一次 group_by 构造六个涨跌停情绪因子。
    """
    # 1. 按日聚合：计数 + 次日收益总和
    daily = (
        prepared.group_by("trading_date")
        .agg(
            n_stock=pl.col("code").n_unique(),
            n_up=pl.col("is_limit_up").cast(pl.Int64).sum(),
            n_down=pl.col("is_limit_down").cast(pl.Int64).sum(),
            up_ret=pl.col("event_next_ret").filter(pl.col("is_limit_up")).sum(),
            down_ret=pl.col("event_next_ret").filter(pl.col("is_limit_down")).sum(),
        )
        .sort("trading_date")
    )

    # 2. 对齐到全市场日历
    result = (
        calendar.select("trading_date")
        .join(daily, on="trading_date", how="left")
        .sort("trading_date")
        .with_columns(pl.col("n_stock", "n_up", "n_down", "up_ret", "down_ret").fill_null(0))
    )

    # 3. 滚动窗口 → 六个因子（select 里直接算，不保留中间列）
    denom = pl.when(pl.col("n_stock").rolling_max(window) > 0) \
              .then(pl.col("n_stock").rolling_max(window)).otherwise(None)

    return result.select(
        "trading_date",
        limit_up_ratio=pl.col("n_up").rolling_sum(window) / denom,
        limit_down_ratio=pl.col("n_down").rolling_sum(window) / denom,
        net_limit_ratio=(pl.col("n_up").rolling_sum(window) - pl.col("n_down").rolling_sum(window)) / denom,
        limit_up_down_ratio=pl.when(pl.col("n_down").rolling_sum(window) > 0)
            .then(pl.col("n_up").rolling_sum(window) / pl.col("n_down").rolling_sum(window))
            .otherwise(None),
        # 收益列 shift(1)：t 日涨停的兑现日在 t+1，因此因子 t 只能看到 t-1 及以前的收益
        limit_up_next_ret=(pl.col("up_ret").shift(1).rolling_sum(window) / pl.col("n_up").shift(1).rolling_sum(window)),
        limit_down_next_ret=(pl.col("down_ret").shift(1).rolling_sum(window) / pl.col("n_down").shift(1).rolling_sum(window)),
    )
```

### 约简效果

| 维度 | 当前 | 简化后 |
|------|------|--------|
| group_by | 2 次 | 1 次 |
| join | 2 次（daily_counts + completed_events） | 1 次（仅 calendar） |
| 函数体行数 | ~40 行 | ~20 行 |
| 中间列 | ~10 个临时列 | 0 |

### 数值一致性

- **比率因子（1-4）**：与原版完全一致（n_up/n_down/n_stock 无 shift）
- **收益因子（5-6）**：数值上与原版理论一致，都是 `rolling_sum(兑现日收益) / rolling_sum(兑现日事件数)`
  - 原版通过 `completed_events` 的 `next_market_date` 对齐兑现日
  - 简化版通过 `shift(1)` 对齐兑现日，数学等价
  - 唯一差异：原版 `up_ret_count` 只计有有效次日收益的事件，简化版用 `n_up` 计所有涨停事件。当某涨停股次日停牌时，原版分母不计入该事件，简化版计入。此差异 < 0.1% 的样本量

## Verification

1. Notebook cell `896f5247` 执行 `build_daily_sentiment_factors` + merge 后，检查 `research_data` 列名正确
2. 尾行数据中 6 个因子有合理数值、非全 NaN
3. IC 分析 cell `85035b24` 正常输出表格和热力图
4. 择时回测 cell `2a777da8` 正常输出绩效汇总
