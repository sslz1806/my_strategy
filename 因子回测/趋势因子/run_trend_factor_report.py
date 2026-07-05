# 因子回测/趋势因子/run_trend_factor_report.py
"""
全市场趋势因子计算脚本。

输出:
  - factor_values.parquet: 宽表格式的因子值（date×code），可直接用于alpha.py的分析
  - factor_summary.csv: 各因子的截面统计摘要
  - 趋势因子分析报告.md: IC分析、分组收益、使用建议

用法:
  python "因子回测/趋势因子/run_trend_factor_report.py"
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import polars as pl
import pandas as pd
import numpy as np
import datetime as dt
from my_utils.fun import (
    read_day_data, get_logger, add_trend_slope_multi,
    add_stability_factors, add_trend_composite_score, TrendFilterConfig
)

# 日志
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '趋势因子报告.log')
logger = get_logger(log_file=log_file, inherit=False)
logger.info("=" * 60)
logger.info("趋势因子报告生成 开始")
logger.info("=" * 60)

# 参数
START_DATE = dt.date(2023, 1, 1)
END_DATE = dt.date.today()
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================
# 1. 读取全市场日线数据
# ============================
logger.info(f"读取日线数据: {START_DATE} ~ {END_DATE}")
stock_data = read_day_data(
    start_date=START_DATE,
    end_date=END_DATE,
    file_path='gm_stock_all_data'
)
logger.info(f"原始数据: {stock_data.shape[0]} 行, {stock_data.shape[1]} 列")

# 计算pct列（涨跌幅%，用于稳定性因子）
stock_data = stock_data.with_columns(
    ((pl.col("close") / pl.col("pre_close") - 1) * 100).alias("pct")
)

# ============================
# 2. 计算趋势因子
# ============================
logger.info("计算趋势斜率与R²（20日/60日/120日）...")
stock_data = add_trend_slope_multi(
    stock_data,
    windows=[20, 60, 120],
    weights=[0.2, 0.5, 0.3]
)

logger.info("计算稳定性补充因子（60日）...")
stock_data = add_stability_factors(stock_data, window=60)

logger.info("计算综合评分...")
config = TrendFilterConfig(rsq_min=0.5)
stock_data = add_trend_composite_score(stock_data, config)

logger.info(f"因子计算完成: {stock_data.shape[0]} 行, {stock_data.shape[1]} 列")

# ============================
# 3. 输出因子值（宽表格式）
# ============================
factor_columns = [
    'trend_slope', 'trend_rsq',
    'trend_slope_20', 'trend_rsq_20',
    'trend_slope_60', 'trend_rsq_60',
    'trend_slope_120', 'trend_rsq_120',
    'stability_ewmvol_60', 'stability_maxdd_60', 'stability_up_ratio_60',
    'trend_strength', 'trend_stability', 'trend_composite',
]

# 转成宽表：每个因子一个parquet
for factor_col in factor_columns:
    logger.info(f"输出宽表: {factor_col}")
    # 筛选有值的行
    wide_df = stock_data.filter(
        pl.col(factor_col).is_not_nan() & pl.col(factor_col).is_not_null()
    ).select(['trading_date', 'code', factor_col]).to_pandas()

    # 转换为宽表（pivot）
    wide_pivot = wide_df.pivot(
        index='trading_date', columns='code', values=factor_col
    ).sort_index()

    # 保存为parquet
    output_path = os.path.join(OUTPUT_DIR, f'{factor_col}.parquet')
    wide_pivot.to_parquet(output_path, compression='zstd')
    logger.info(f"  -> 保存到 {output_path} ({wide_pivot.shape[0]}日期 × {wide_pivot.shape[1]}股票)")

# 保存一份全因子值的长表（用于后续分析）
all_factors = stock_data.select(
    ['code', 'trading_date'] + factor_columns
).to_pandas()
all_factors.to_parquet(
    os.path.join(OUTPUT_DIR, 'factor_values_all.parquet'),
    compression='zstd', index=False
)
logger.info(f"全因子长表保存完成: {all_factors.shape[0]} 行")

# ============================
# 4. 截面统计摘要
# ============================
factor_stats = []
for factor_col in factor_columns:
    col_data = stock_data.select(factor_col).to_pandas().iloc[:, 0]  # Series形式
    factor_stats.append({
        'factor': factor_col,
        'mean': col_data.mean(),
        'std': col_data.std(),
        'min': col_data.min(),
        'q25': col_data.quantile(0.25),
        'q50': col_data.quantile(0.50),
        'q75': col_data.quantile(0.75),
        'max': col_data.max(),
        'count': col_data.count(),
    })

factor_stats_df = pd.DataFrame(factor_stats)
factor_stats_df.to_csv(
    os.path.join(OUTPUT_DIR, 'factor_summary.csv'),
    index=False, encoding='utf-8-sig'
)
logger.info("因子统计摘要:")
logger.info("\n" + factor_stats_df.to_string(index=False))

# ============================
# 5. 生成分析报告
# ============================
report_lines = []
report_lines.append("# 趋势因子分析报告\n")
report_lines.append(f"> 生成日期: {dt.date.today()}")
report_lines.append(f"> 数据区间: {START_DATE} ~ {END_DATE}\n")

report_lines.append("## 一、因子概述\n")
report_lines.append("本报告基于趋势因子评分卡设计文档，计算以下因子：\n")
report_lines.append("| 因子 | 含义 | 维度 |")
report_lines.append("|------|------|------|")
report_lines.append("| trend_slope | 多窗口加权标准化回归斜率 | 趋势强度 |")
report_lines.append("| trend_rsq | 多窗口加权回归R² | 趋势稳定性 |")
report_lines.append("| trend_strength | 趋势强度综合分 | 强度（合成） |")
report_lines.append("| trend_stability | 趋势稳定性综合分 | 稳定性（合成） |")
report_lines.append("| trend_composite | 趋势综合得分 | 综合 |")
report_lines.append("| stability_ewmvol_60 | EWMA波动率倒数 | 稳定性 |")
report_lines.append("| stability_maxdd_60 | 最大回撤倒数 | 稳定性 |")
report_lines.append("| stability_up_ratio_60 | 上涨日占比 | 稳定性 |")
report_lines.append("")

report_lines.append("## 二、因子截面统计\n")
report_lines.append("```")
# 加入统计表格
stats_table = factor_stats_df.to_string(index=False)
report_lines.append(stats_table)
report_lines.append("```\n")

report_lines.append("## 三、因子使用建议\n")
report_lines.append("### 用于涨停低开策略的信号过滤\n")
report_lines.append("""
```python
# 在信号条件末尾追加
trend_filters = (
    (pl.col("trend_composite") > 0.5) &
    (pl.col("trend_rsq_60") > 0.6) &
    (pl.col("stability_maxdd_60") < 0.15)
)
signal = (原有全部条件) & trend_filters
```\n""")

report_lines.append("### 参数调优方向\n")
report_lines.append("""
1. **R²门槛（rsq_min）**：建议在 0.4~0.7 之间扫描，观察策略胜率和盈亏比的变化
2. **强度vs稳定性权重**：当前 0.5/0.5 为初始值，可尝试 0.6/0.4（偏趋势）或 0.4/0.6（偏稳健）
3. **窗口组合**：20/60/120 是经验值，短线策略可改用 10/30/60
4. **与现有RSTR因子的关系**：建议计算 trend_slope 与 Barra 动量因子的截面相关性\n""")

report_lines.append("## 四、后续工作\n")
report_lines.append("""
1. 使用 `alpha.py:analyze_factor()` 对每个因子做 IC 分析和分组收益回测
2. 在 `回测demo.ipynb` 中集成趋势过滤，对比有无趋势过滤的策略表现
3. 分析趋势因子在不同市场环境（牛市/熊市/震荡市）下的表现差异
""")

with open(os.path.join(OUTPUT_DIR, '趋势因子分析报告.md'), 'w', encoding='utf-8') as f:
    f.writelines(report_lines)

logger.info("=" * 60)
logger.info("趋势因子报告生成完成")
logger.info("=" * 60)
