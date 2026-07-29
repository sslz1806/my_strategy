# 时序因子分组回测函数设计文档

> 将 `情绪因子 v2.ipynb` 中的分组测试与分组策略回测代码包装为 `alpha.py` 的可复用函数

## 1. 背景

`情绪因子 v2.ipynb` 中有一段分组测试与分组回测的代码，包含两个核心内部函数 `backtest_group_strategy` 和 `calculate_group_performance_metrics`，逻辑清晰、职责分明。需求是将该段代码包装为 `alpha.py` 的可复用函数 `backtest_timeseries_factor`，保持原代码结构基本不动，仅做必要的参数化改造。

## 2. 函数签名

```python
def backtest_timeseries_factor(
    analysis_data: pd.DataFrame,
    factor_col: str,
    index_ret_col: str,
    q: int = 5,
    hold_period: int = 5,
    plot: bool = True,
) -> dict:
```

## 3. 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `analysis_data` | pd.DataFrame | — | 时序因子数据。**index 为时间轴**（交易日/15分钟时间戳/任意有序时间戳均可，函数不假定频率粒度）。必须含 `factor_col` 和 `index_ret_col` 两列。 |
| `factor_col` | str | — | 因子值列名 |
| `index_ret_col` | str | — | 单期收益率列名（**%单位**，如 0.5 表示 0.5%）。NaN 在内部被填充为 0。 |
| `q` | int | 5 | 等分位组数 |
| `hold_period` | int | 5 | 持仓期数（与 index 粒度一致：日线传天数，分钟线传分钟期数） |
| `plot` | bool | True | 是否自动 plt.show() 显示图表。图表 Figure 始终通过返回值返回。 |

### 对 `index_ret_col` 的单位约定

`index_ret_col` 的值是百分比数值，5 表示 5%（而非 0.05）。这一约定与 notebook 中 `daily_return` 列的语义一致。函数内部使用 `ret / 100` 转为小数计算。

## 4. 返回值

```python
{
    'group_stats': pd.DataFrame,
        # index=factor_group (G1~Gq)
        # columns=['平均收益(%)', '收益标准差', '样本数']
        # 分组的未来收益描述统计

    'group_performance': pd.DataFrame,
        # index=group_name (G1~Gq + '买入持有基准')
        # columns=['累计收益(%)', '年化收益(%)', '夏普比率', '最大回撤(%)', '胜率(%)', '持仓占比(%)']
        # 每组作为独立策略的绩效指标

    'group_nav': pd.DataFrame,
        # columns=各分组名 (G1~Gq), index=analysis_data.index
        # 每组策略净值（不含基准，基准在绘图中单独画虚线）

    'future_return_col': str,
        # 内部生成的未来收益列名，格式 f'future_return_{hold_period}d'

    'fig_bar': plt.Figure | None,
        # 分组平均收益柱状图 Figure 对象

    'fig_nav': plt.Figure | None,
        # 分组策略净值对比图 Figure 对象
}
```

## 5. 内部流程

### 5.1 数据准备

```python
analysis_data = analysis_data.copy()
analysis_data = analysis_data.sort_index()  # 确保时序有序
```

### 5.2 未来收益合成（唯一的结构性改动）

```python
future_return_col = f'future_return_{hold_period}d'
ret_series = analysis_data[index_ret_col].fillna(0)  # NaN → 0
gross = 1 + ret_series / 100
future_gross = gross.rolling(hold_period).apply(np.prod, raw=True).shift(-hold_period)
analysis_data[future_return_col] = (future_gross - 1) * 100
```

**说明**：原 notebook 从 `nav` 列 `shift(-h)/nav-1` 计算未来收益，改为从单期收益率滚动复利合成。数学等价，但不再依赖 `nav` 列。

### 5.3 分组

```python
analysis_data_clean = analysis_data.dropna(subset=[factor_col]).copy()
analysis_data_clean['factor_group'] = pd.qcut(
    analysis_data_clean[factor_col],
    q=q,
    labels=[f'G{i+1}' for i in range(q)],
    duplicates='drop'
)
```

### 5.4 分组未来收益统计 + 柱状图

- `groupby('factor_group')[future_return_col].agg(['mean', 'std', 'count'])`
- 柱状图 + 数值标签 + y=0 参考线

### 5.5 分组策略回测（保留 notebook 原版实现）

#### `backtest_group_strategy(data, group_col, target_group, hold_period)`

```
signal = (data[group_col] == target_group)
→ 持仓期循环（numpy 实现）
→ strategy_return = position * daily_return / 100
→ strategy_nav = (1 + strategy_return).cumprod()
```

#### `calculate_group_performance_metrics(data, group_name)`

```
→ 累计收益 / 年化收益 / 夏普比率 / 最大回撤 / 胜率 / 持仓占比
```

原 notebook 中 `strategy_return = position * daily_return / 100` 中的 `daily_return` 即为参数 `index_ret_col`，函数内统一使用 `index_ret_col` 列名。

#### 基准

买入持有基准：满仓 `index_ret_col` 的复利净值。放在 `group_performance` 最后一行。在净值图中用虚线区分。

### 5.6 净值对比图

- q 条实线 + 1 条虚线基准
- 图例：`G1`~`Gq` + `'买入持有基准'`

## 6. 边缘情况处理

| 场景 | 处理 |
|------|------|
| `factor_col` 含 NaN | `dropna(subset=[factor_col])` |
| `index_ret_col` 含 NaN | `fillna(0)` 后参与复利计算 |
| 尾部不够 `hold_period` 长度 | `shift(-hold_period)` 产生 NaN，最终 `dropna` 丢弃 |
| `pd.qcut` 重复值过多导致实际分组 < q | 保留 `duplicates='drop'`，打印警告 |
| 有效样本数 < q | 抛 ValueError |

## 7. 放置位置

函数定义在 `因子回测/alpha.py`，位于 `analyze_factor()` 之后，用注释分隔：

```python
# ============================================================
# 时序因子分组回测
# ============================================================

def backtest_timeseries_factor(...):
    ...
```

## 8. 与 notebook 的差异对照

| 项目 | notebook 原版 | 函数版 |
|------|-------------|--------|
| 因子列名 | 硬编码 `'sentiment_factor'` | 参数 `factor_col` |
| 收益率列名 | `'daily_return'` | 参数 `index_ret_col` |
| 分组数 | 硬编码 5 | 参数 `q` |
| 持仓天数 | 硬编码 3 | 参数 `hold_period` |
| 未来收益来源 | `nav.shift(-h)/nav-1` | 从 `index_ret_col` 复利合成 |
| `analysis_data` | notebook 全局变量 | 参数传入 |
| 图表显示 | 固定 `plt.show()` | `plot` 控制，始终返回 Figure |
| NaN 处理 | `dropna()` 透传 | `index_ret_col` 先 fillna(0) |
| 量化精度 | 符合回测需求 | 与 notebook 完全一致 |
