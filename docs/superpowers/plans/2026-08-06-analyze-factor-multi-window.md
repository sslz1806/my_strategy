# `analyze_factor` 实施清单

- [x] 保留旧宽表函数为 `analyze_factor_bak`，新函数只接收单张 Polars 长表。
- [x] 复用 `add_future_return` 计算多窗口 IC、RankIC 和累计 IC。
- [x] 按 FactorAna 的调仓锚点延续持仓组，次日生效并按每日收益连续计算净值。
- [x] 支持可选 `benchmark_ret` 和三类多窗口图。
- [x] 用手算样本验证无未来函数、无重叠收益、累计 IC、benchmark 和 G10 排序。
- [x] 运行相关回归测试并检查最终差异。
