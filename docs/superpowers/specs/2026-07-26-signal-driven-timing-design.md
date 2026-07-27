# 日频情绪择时回测：信号驱动续期设计

> 日期：2026-07-26  
> 背景：将 `sentiment_factors_5d_research.ipynb` 中的 `run_timing` 从固定步长非重叠调仓，改为信号触发即开仓、中途触发即续期的连续持仓模式。

---

## 1. 设计目标

- 用 `signal_factor` 标记每日是否触发信号；
- 信号在 T 日收盘后生成，仓位从 T+1 日开始生效；
- 触发后至少持有 `horizon` 个交易日；
- 若在持有期内再次触发，持仓期自动延长至最新信号日 + `horizon`；
- 空仓日仓位为 0，但逐日收益表中仍保留 0 收益行，保证净值曲线连续；
- 预热期信号为 NaN 时不进入样本，不改变当前仓位状态；
- IC 分析保持独立口径：仍用因子值与固定 `horizon` 日未来收益计算。

---

## 2. 关键边界

| 场景 | 处理 |
|------|------|
| T 日信号=1 | T+1 日开始持仓，初始持有到 T+horizon |
| T+k 日（k<horizon）信号=1 | 续期，持有到 T+k+horizon |
| T+k 日信号=0 | 不提前平仓，仍按当前 holding_until 持有 |
| 空仓日 | position=0，strategy_daily_ret=0，保留 daily row |
| 预热期信号=NaN | 不更新 holding_until，不改变当前状态 |
| 数据末尾不完整持有期 | 默认不进入 blocks 统计，避免未来函数 |

---

## 3. 实现方案

采用显式状态机（方案 2）：

```python
holding_until_idx = -1
for t_idx in range(len(ordered) - 1):
    signal = ordered.iloc[t_idx][signal_column]
    if pd.notna(signal) and signal == 1:
        holding_until_idx = t_idx + horizon

    position = 1.0 if t_idx + 1 <= holding_until_idx else 0.0
    # 记录 t_idx+1 日收益
```

绩效统计由固定 block 改为连续持仓段：
- block 开始：当日有仓位且前一日无仓位；
- block 结束：当日有仓位且下一日无仓位；
- 只统计完整到期的 block（可通过参数关闭）。

---

## 4. 验证计划

1. 构造合成样本：T 触发、T+4 触发，验证持仓延长到 T+8；
2. 验证空仓日 position=0 且 strategy_daily_ret=0；
3. 验证预热期 NaN 不触发开仓；
4. 运行现有测试，确保未破坏既有功能。

---

## 5. 影响范围

- `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb`
  - `run_timing` 函数
  - `summarize_timing` 函数
- `tests/test_sentiment_factors_5d_notebook.py` 或新增测试文件
