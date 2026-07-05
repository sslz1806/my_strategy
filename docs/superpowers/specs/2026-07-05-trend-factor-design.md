# 趋势因子评分卡设计文档

> 用于涨停低开策略的信号过滤场景，量化衡量股票趋势的"好"与"稳"

## 一、背景与目标

### 1.1 动机

现有涨停低开策略已使用 SMA_7 均线位置、乖离率、30日最低价位置等作为趋势过滤条件。但这些条件偏经验规则，缺乏对趋势"方向强度"和"走势稳定性"的统一定量度量。本设计旨在构建一个结构化的趋势评分卡，提供：

- 一个综合的、可解释的 "趋势好且稳" 得分
- 可作为现有策略的额外信号过滤层
- 可独立用于选股或因子研究

### 1.2 使用场景

**信号过滤**（主要）：从涨停低开候选股票池中，筛选出"趋势良好且运行稳健"的股票，过滤掉超跌反弹、趋势反转、剧烈波动等不可持续情形。

### 1.3 设计原则

- **简洁可解释**：每个子因子有明确的经济含义，方便调参与验证
- **与项目现有代码兼容**：基于 Polars 实现，融入 `fun.py` 特征链
- **模块化可独立测试**：各因子可单独计算、单独回测
- **快速迭代**：参数（窗口、权重、门槛）可通过配置调整，不需要改代码

## 二、因子框架

### 2.1 核心维度

| 一级维度 | 权重 | 含义 | 核心度量 |
|---------|------|------|---------|
| 趋势强度 | 50% | "趋势好不好" | 回归标准化斜率 + 价格相对位置 |
| 趋势稳定性 | 50% | "趋势稳不稳" | 回归R² + 波动率倒数 + 最大回撤 |

### 2.2 因子结构

#### 趋势强度子因子

| 子因子 | 权重 | 计算方法 |
|--------|------|---------|
| 多窗口回归斜率 | 60% | 20日/60日/120日三个窗口滚动线性回归的标准化斜率，加权合成 |
| 价格相对位置 | 20% | 收盘价 / 60日均线，控制在 0.8~1.2 区间（过高过热，过低趋势弱） |
| N日动量 | 20% | 过去20日累计收益，与斜率互补（斜率反映趋势方向，动量反映近期力度） |

#### 趋势稳定性子因子

| 子因子 | 权重 | 计算方法 |
|--------|------|---------|
| 回归R² | 40% | 60日滚动线性回归的拟合优度，直接衡量"沿着一条直线走得直不直" |
| EWMA波动率倒数 | 25% | `1 / ewma_volatility`，波动越小越稳定 |
| 60日最大回撤倒数 | 25% | `1 - low_60min / high_60max`，回撤越小越稳 |
| 上涨日占比 | 10% | 60日内上涨天数占比，平稳上行市场该值在 0.5~0.7 区间 |

### 2.3 窗口参数

| 窗口 | 趋势强度权重 | 趋势稳定性权重 | 定位 |
|------|------------|--------------|------|
| 20日 | 20% | — | 短期边际变化，对涨停低开事件敏感 |
| 60日 | 50% | 100% | 主力窗口，平衡响应速度与稳健性 |
| 120日 | 30% | — | 长期趋势方向确认，过滤纯超跌反弹 |

### 2.4 评分合成流程

```
原始因子值
  → 截面 rank 标准化（0-1 百分位）
  → 子维度内加权合成 → 强度分 + 稳定性分
  → 硬性门槛过滤（R² < 0.6 淘汰、最大回撤 > 20% 淘汰）
  → 综合分 = 强度分 × 0.5 + 稳定性分 × 0.5
  → 最终得分范围 [0, 1]
```

## 三、核心算法

### 3.1 滚动线性回归斜率与 R²

对每只股票在第 t 天，以过去 W 天收盘价为因变量 y，时间索引 [0, 1, ..., W-1] 为自变量 x，做一元 OLS。

```
斜率(slope) = cov(x, y) / var(x) = r(x,y) × σy / σx
R² = r(x,y)²
标准化斜率 = slope / close_mean（消除股价量纲）
```

由于 x 为等差数列，var(x) 和 σx 对给定 W 为常数：
```
σx = sqrt(W × (W+1) / 12)
```

因此只需滚动计算 `corr(x, y)` 和 `σy`，即可得到斜率和 R²。

### 3.2 Polars 实现

使用 `pl.corr` 的滚动窗口计算相关系数，纯 Polars 无外部依赖。

```python
def add_trend_slope_rsq(df: pl.DataFrame, window: int = 60) -> pl.DataFrame:
    """为每只股票添加滚动回归的标准化斜率与R²"""
    t_col = f"t_{window}"
    r_col = f"r_{window}"
    slope_col = f"trend_slope_{window}"
    rsq_col = f"trend_rsq_{window}"
    std_close_col = f"std_close_{window}"
    
    df = df.with_columns(
        pl.int_range(pl.len()).over("code").alias(t_col)
    )
    # 滚动相关系数
    df = df.with_columns(
        pl.corr(t_col, "close")
        .rolling(window, min_periods=window)
        .over("code")
        .alias(r_col)
    )
    # 滚动标准差
    df = df.with_columns(
        pl.col("close")
        .rolling_std(window)
        .over("code")
        .alias(std_close_col)
    )
    std_t = (window * (window + 1) / 12) ** 0.5
    # 斜率 = r * σy / σx
    df = df.with_columns(
        ((pl.col(r_col) * pl.col(std_close_col) / std_t) / 
         pl.col("close").rolling_mean(window).over("code"))
        .alias(slope_col)
    )
    # R²
    df = df.with_columns(
        (pl.col(r_col) ** 2).alias(rsq_col)
    )
    return df
```

## 四、策略集成

### 4.1 集成位置

在现有信号条件末尾追加趋势过滤条件：

```python
trend_filters = (
    (pl.col("trend_composite") > 0.5) &
    (pl.col("trend_rsq_60") > 0.6) &
    (pl.col("stability_maxdd_60") < 0.15)
)

signal = (原有的全部条件) & trend_filters
```

### 4.2 参数可配置

```python
class TrendFilterConfig:
    composite_threshold: float = 0.5    # 综合得分门槛
    rsq_min: float = 0.6                # 最低R²要求
    # 注意：stability_maxdd_60 是回撤倒数(=1/(1-low/high))，值越大回撤越小
    # 断板低开票60日回撤普遍40-68%，所以此条件不纳入默认过滤
```

## 五、因子报告输出

为了方便后续单因子回测，所有因子值将输出为 Parquet 格式到 `因子回测/趋势因子/` 目录：

| 输出文件 | 内容 | 频率 |
|---------|------|------|
| `factor_values.parquet` | 每只股票每日的完整因子值 | 日频 |
| `factor_summary.csv` | 各因子的截面均值、标准差、IC等统计 | 日频 |
| `趋势因子分析报告.md` | 因子分组收益、IC序列分析、使用建议 | 一次性 |

### 5.1 因子列

最终输出的因子数据集包含以下列：

| 列名 | 说明 | 所属维度 |
|------|------|---------|
| `trend_slope_20/60/120` | 标准化回归斜率 | 强度 |
| `trend_rsq_20/60/120` | 回归R² | 稳定性 |
| `trend_composite` | 综合得分 | 综合 |
| `trend_strength` | 强度分 | 强度 |
| `trend_stability` | 稳定性分 | 稳定性 |
| `price_position_60` | 价格/60日线 | 强度 |
| `stability_ewmvol_60` | EWMA波动率倒数 | 稳定性 |
| `stability_maxdd_60` | 60日最大回撤 | 稳定性 |
| `stability_up_ratio_60` | 上涨日占比 | 稳定性 |

### 5.2 因子评估

使用 `因子回测/alpha.py` 中的现有工具进行：
- IC 分析（截面 Rank IC、ICIR）
- 分组收益（分5组的多空收益）
- 因子相关性（与已有因子如 RSTR、波动率的相关性矩阵）

## 六、代码组织结构

```
my_utils/fun.py  → 新增：
  ├── add_trend_slope_rsq(df, window)     # 回归斜率+R²
  ├── add_trend_slope_multi(df, windows)  # 多窗口合成
  ├── add_stability_factors(df, window)   # 稳定性补充因子
  ├── add_trend_composite_score(df)       # 综合评分
  └── TrendFilterConfig                   # 可配置参数

因子回测/趋势因子/
  ├── run_trend_factor_report.py          # 运行因子计算的脚本
  ├── factor_values.parquet               # 因子值输出
  ├── factor_summary.csv                  # 因子统计摘要
  └── 趋势因子分析报告.md                  # 分析报告

回测demo.ipynb  → 修改：
  └── 在信号条件中追加 trend_filters
```

## 七、待验证问题

1. **窗口参数的最佳选择**：20/60/120 是经验值，可能需要通过 IC 衰减曲线优化
2. **R²门槛的敏感性**：0.6 是初步值，需要用回测验证不同门槛下的策略表现
3. **趋势强度与稳定性的最优权重**：0.5/0.5 是等权起点，可用 IC 加权或优化器调整
4. **与其他因子的正交性**：需要验证趋势评分与现有 RSTR、波动率因子的相关性，避免共线性
5. **极端行情适应性**：2024年初的流动性危机中，趋势因子可能全部失效，需要考虑熔断等特殊处理
