# 米筐因子数据缓存增量更新设计

## 背景

`米筐官方因子收益率_风格趋势.ipynb` 每次运行都会调用米筐 API 拉取风险因子收益数据（v1/v2 两套模型）。
当前使用全量 parquet 缓存 + 存在即加载的策略，导致 `WINDOW_END` 改为动态日期后，
缓存文件一旦生成就不再更新，用户需要手动删除才能拿到最新数据。

本设计解决：**省时间（利用缓存）和数据新鲜（自动增量更新）** 的矛盾。

## 设计

### 缓存结构

每个模型（v1 / v2）对应两个文件，存放在 `{NB_DIR}/saved_data/` 目录下：

| 文件 | 用途 |
|------|------|
| `rq_factor_return_v1.parquet` | v1 模型因子收益数据（完整历史） |
| `rq_factor_return_v1_meta.json` | v1 元数据：缓存覆盖的日期范围 |
| `rq_factor_return_v2.parquet` | v2 模型因子收益数据 |
| `rq_factor_return_v2_meta.json` | v2 元数据 |

元数据格式：

```json
{
  "last_date": "2026-07-01",
  "window_start": "2025-05-20",
  "created_at": "2026-07-01T10:23:45"
}
```

用独立元数据文件而非直接读 parquet 来获取 `last_date`，是因为读取整个 parquet 查最后一行
比读一个 200 字节的 JSON 文件慢得多——虽然数据量不大，但原则上有意分离。

### 加载逻辑

```python
def load_factor_return(model, start_date, end_date):
    cache_path = saved_data / f"rq_factor_return_{model}.parquet"
    meta_path  = saved_data / f"rq_factor_return_{model}_meta.json"

    if cache_exists_and_valid(cache_path, meta_path, start_date, end_date):
        # 元数据 last_date >= end_date → 直接用缓存
        return pd.read_parquet(cache_path)

    if cache_exists_but_stale(cache_path, meta_path, start_date, end_date):
        # 元数据 last_date < end_date → 增量拉取缺失区间
        fr = pd.read_parquet(cache_path)
        new_data = rq.get_factor_return(last_date + 1d, end_date, model=model)
        if new_data is not None and not new_data.empty:
            fr = pd.concat([fr, new_data])
            # 完整写回 parquet（内存已增量，写入覆盖旧文件）
            fr.to_parquet(cache_path)
            _write_meta(meta_path, last_date=end_date, window_start=start_date)
        return fr

    # 缓存不存在或 start_date 变化 → 全量重拉（会覆盖旧文件）
    fr = rq.get_factor_return(start_date, end_date, model=model)
    fr.to_parquet(cache_path)
    _write_meta(meta_path, last_date=end_date, window_start=start_date)
    return fr
```

### 流程图

```
                    ┌──────────────┐
                    │ 检查缓存存在  │
                    └──────┬───────┘
                           │
                 ┌─────────┴─────────┐
                 │                   │
             不存在               存在
                 │                   │
           读 meta.json ────→  比对 start_date 和 last_date
                 │                   │
         ┌───────┴────────┐  ┌──────┴──────┐
         │                │  │             │
     start_date 变了   start_date 一致  last_date ≥ end_date
         │                │  │             │
     全量从 start_date ──┘  │        直接用缓存
     拉到 end_date          │        (跳过网络)
                            │
                      last_date < end_date
                            │
                     只拉 (last_date + 1d, end_date]
                     追加到缓存 DataFrame
                     写回 parquet + meta
```

### 边界处理

| 场景 | 行为 |
|------|------|
| 缓存不存在 | 全量拉取 |
| 缓存覆盖到昨天，今天还未拉取 | 只拉今天一天，追加 |
| 元数据 `last_date` 与 parquet 实际不符 | 降级全量重拉（安全但罕见） |
| 用户修改了 `WINDOW_START`（与元数据不一致） | 删除旧缓存，全量重拉 |
| API 在增量区间内某天返回空（非交易日） | 跳过，不追加空行 |
| parquet 文件损坏 | 捕获异常，降级全量重拉 |

### 不做的

- 不做多线程/异步拉取（数据量小，不值得）
- 不做跨 session 的缓存一致性校验（只在每个 cell 运行时校验一次）
- 不拆分 v1/v2 为独立类（现在只是 notebook 里的一个辅助函数）

## 位置

增量逻辑封装在 `米筐官方因子收益率_风格趋势.ipynb` 的 Part 1 中，替换现有 `if-else` 缓存判断，
作为 cell-4 内的一个辅助函数 `load_factor_return_with_cache()`。

如果 `barra_cne5_风格归因.ipynb` 后续也用到同样逻辑，再考虑抽取到 `my_utils/rqdata.py`。

## 验收标准

1. 首次运行（无缓存）：全量拉取，生成 parquet + meta，控制台打印"全量拉取完成"
2. 次日再运行：只拉最新一天，控制台打印"增量追加 X 条"
3. 当天的第二次运行：不走网络，控制台打印"缓存已最新"
4. 删除缓存文件后运行：回到行为 1
