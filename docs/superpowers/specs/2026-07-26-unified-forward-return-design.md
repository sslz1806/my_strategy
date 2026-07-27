# alpha.py 统一未来收益函数设计

## 1. 目标

在 `因子回测/alpha.py` 中提供一个统一的未来收益计算入口 `add_future_return`，同时支持 Polars 和 Pandas 输入，替代项目中散落的同类手写实现。

本次聚焦：
- 替代 `因子回测/alpha.py` 中旧的 `cal_next_return`；
- 替代 `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb` 中手写的未来收益计算；
- 不改动 `my_utils/stock_api.py` 中的 `get_all_date_data_with_future`（留待后续统一）。

## 2. 设计决策

| 决策项 | 选择 | 说明 |
|---|---|---|
| 函数位置 | `因子回测/alpha.py` | 与现有 `analyze_factor`、`ols_neutralize` 等因子分析函数放在一起 |
| 函数名 | `add_future_return` | 直观表达“添加未来收益列” |
| 后端支持 | Polars + Pandas，自动分发 | 一个函数名，内部按输入类型走不同分支 |
| 数据形态 | 面板数据 / 时序数据自动识别 | 有 `code` 列则按股票分组，否则按单序列计算 |
| 收益来源 | `ret_col` 优先，回退到 `price_col / pre_close_col - 1` | 默认 `ret_col="pct"` |
| 输出列名 | `future_{ret_col}_{n}d` | 例如 `future_pct_5d`、`future_close_1d` |

## 3. API 设计

```python
from typing import Sequence

def add_future_return(
    df,
    ret_col: str = "pct",
    price_col: str = "close",
    pre_close_col: str = "pre_close",
    horizons: Sequence[int] = (1, 5, 10, 20),
    date_col: str = "trading_date",
    code_col: str = "code",
):
    """
    为输入 DataFrame 添加未来 n 日累计收益列。

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        输入数据，必须包含 date_col 所指定的日期列。
    ret_col : str, default "pct"
        日收益列名。如果该列存在，则直接用它的值作为日收益（小数形式，
        例如 0.01 表示 1%）。如果该列不存在，则使用 price_col / pre_close_col - 1
        计算日收益。
    price_col : str, default "close"
        收盘价列名，仅在 ret_col 不存在时用于计算日收益。
    pre_close_col : str, default "pre_close"
        昨收列名，仅在 ret_col 不存在时用于计算日收益。
    horizons : Sequence[int], default (1, 5, 10, 20)
        未来收益周期列表，每个元素 n 会生成一列 future_{ret_col}_{n}d。
    date_col : str, default "trading_date"
        日期列名，用于内部排序。
    code_col : str, default "code"
        股票代码列名。如果 df 中存在该列，则按股票分组计算未来收益；
        否则按单一时序计算。

    Returns
    -------
    pl.DataFrame | pd.DataFrame
        与输入同类型的 DataFrame，新增 future_{ret_col}_{n}d 列。
        返回结果按 date_col（时序）或 [code_col, date_col]（面板）排序。

    Notes
    -----
    - 未来收益从 t 日收盘后开始计算，不包含 t 日当日收益。
    - 累计收益使用复利方式：prod(1 + r_{t+1}, ..., 1 + r_{t+n}) - 1。
    - 样本尾部不足 n 个交易日的行，对应 future_{ret_col}_{n}d 为 NaN / null。
    """
```

## 4. 实现要点

### 4.1 日收益获取

```python
# 伪代码
if ret_col in df.columns:
    daily_ret = ret_col
else:
    daily_ret = computed_ret_col  # price_col / pre_close_col - 1
```

### 4.2 未来收益计算

对日收益序列 `r_t`，未来 n 日累计收益为：

```
future_ret_{n}d(t) = prod(1 + r_{t+1}, ..., 1 + r_{t+n}) - 1
```

- **Polars 面板数据**：使用 `pl.col(daily_ret).add(1).rolling_product(n, min_samples=n).shift(-n).over(code_col)`。
- **Polars 时序数据**：去掉 `.over(code_col)`。
- **Pandas 面板数据**：使用 `groupby(code_col)` + `rolling(n).apply(np.prod, raw=True).shift(-n)`。
- **Pandas 时序数据**：去掉 `groupby(code_col)`。

### 4.3 排序

内部先排序再计算，确保 rolling/shift 语义正确：

- 面板数据：按 `[code_col, date_col]` 排序；
- 时序数据：按 `[date_col]` 排序。

返回排序后的 DataFrame。

### 4.4 列名生成

无论日收益是直接使用 `ret_col` 还是回退计算得到，输出列名统一使用：

```python
f"future_{ret_col}_{n}d"
```

这样当 `ret_col="pct"` 时，即使回退到 `close / pre_close - 1`，输出列也是 `future_pct_1d`、`future_pct_5d` 等，命名保持一致。

## 5. 迁移范围

| 文件 | 操作 |
|---|---|
| `因子回测/alpha.py` | 新增 `add_future_return`；删除 `cal_next_return` |
| `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb` | 删除/替换手写未来收益计算，改为 `from 因子回测.alpha import add_future_return` |

## 6. 验证方案

1. **后端一致性**：用同一组合成数据分别生成 Polars 和 Pandas 输入，比较输出结果是否一致；
2. **面板 vs 时序**：验证多股票输入能按股票分组，单序列输入不报错；
3. **收益来源回退**：分别测试 `ret_col` 存在、不存在两种情况，确认回退计算正确；
4. **输出列名**：确认列名符合 `future_{ret_col}_{n}d`；
5. **尾部空值**：确认样本尾部不足 horizon 的行返回 NaN / null；
6. **无残留引用**：全局搜索 `cal_next_return` 和旧版手写未来收益实现，确认已替换干净。

## 7. 不纳入范围

- 不修改 `my_utils/stock_api.py` 中的 `get_all_date_data_with_future`；
- 不修改 `alpha.py` 中的 `analyze_ic`、`analyze_factor` 等其他函数；
- 不新增情绪择时专用的阈值、信号、回测函数。
