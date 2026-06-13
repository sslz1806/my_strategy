# Barra CNE5 与 Fama-French 学习笔记 — 设计文档

> 日期：2026-06-13　目标目录：`因子回测/learn/`
> 目的：在学习项目中实现 `barra_use3`、`fama_french`、`fama_french5` 三个教学 notebook，
> 沿用既有 `CAPM.ipynb`/`马科维兹理论.ipynb` 的渐进式教学风格。

## 通用约定（三个 notebook 共享）

- **数据源**：统一用米筐 `rqdatac`（封装 `my_utils.rqdata.RqData`，license 已内置）。
  理由：基本面(PB/ROE/资产增速)、申万行业、指数成分、官方 Barra 暴露都在米筐，单源保证截面对齐。
  本地 `my_utils.fun.read_day_data` / 掘金作为备选源在注释中说明。
- **股票池**：默认中证800 成分（`rqdatac.index_components('000906.XSHG', date)`），剔除 ST/停牌/上市不足 1 年。
- **区间**：2021-01 ~ 2024-12（FF 月频；Barra 截面取若干代表日演示）。
- **路径处理**：复用 CAPM 笔记的 `PROJECT_ROOT` 自动定位 + `sys.path` 注入片段。
- **风格**：理论↔代码映射表 → 数据准备 → 因子构建 → 回归/检验 → 可视化 → 小结；详细中文注释。

## ① barra_use3.ipynb — Barra CNE5 多因子风险模型

核心方程（截面）：`r_i = Σ_k X_ik · f_k + u_i`，WLS 截面回归反解因子收益 `f`。

10 个 CNE5 风格因子（教学版：主描述子为主，子描述子做加权简化并注明口径）：

| 因子 | 口径 | 米筐字段/算法 |
|------|------|----------------|
| Size | 总市值对数 | `ln(market_cap_3)` |
| Beta | 个股对市场 252 日回归 β | 价格回归 |
| Momentum (RSTR) | 过去 12 月剔除最近 1 月累计对数收益 | 价格 |
| Residual Volatility | Beta 回归残差波动 + 日收益波动 | 价格 |
| Non-linear Size | Size³ 对 Size 正交化后的残差 | 派生 |
| Book-to-Price | 1 / PB | `1/pb_ratio_ttm` |
| Liquidity | 月/季/年换手率对数加权 | `turnover` 或本地换手率 |
| Earnings Yield | EP 盈利市值比等 | `1/pe_ratio_ttm` 等 |
| Growth | 盈利/营收增长 | `total_asset_growth_ratio_ttm` 等 |
| Leverage | 市场杠杆/账面杠杆/负债资产比 | 财务派生 |

流程：构建描述子 → 标准化(行业均值填缺 → MAD 去极值 → 市值加权 z-score) →
暴露矩阵 X=[国家因子|申万行业哑变量|10风格] → WLS(权重=√流通市值) 估因子收益(行业市值加权和=0 约束) →
**验证：自建暴露 vs 米筐官方 `get_factor_exposure` 逐因子相关性**；因子收益累计净值/bar/相关性热图。

## ② fama_french.ipynb — FF 经典三因子

方程：`R_i - r_f = α + β_MKT·MKT + β_SMB·SMB + β_HML·HML + ε`。

- 2×3 分组：Size 按市值中位数 S/B；BP 按 30/40/70 分位 L/M/H → 6 组合。
- SMB = 小盘组合均值 − 大盘组合均值；HML = 高 BP 均值 − 低 BP 均值；MKT = 市场超额收益。
- 因子表现（累计净值/年化/t 检验）→ 时序回归测试资产(规模×价值组合/个股)→ α 显著性、R²、暴露解读。
- 点明 CAPM 是 FF3 的 β_SMB=β_HML=0 特例；小结含 GRS 检验思路。

## ③ fama_french5.ipynb — FF 五因子（新建文件）

FF3 基础上加：
- RMW（盈利）：营业利润率/ROE 排序 robust − weak。
- CMA（投资）：总资产增速排序 conservative − aggressive（增速低=保守）。
- 独立 2×3 排序构建；Part 4 对比 FF3 vs FF5 解释力提升，复现"FF5 中 HML 可能冗余"。

## 验证标准（目标驱动）

- Barra：自建 10 因子暴露与米筐官方暴露逐因子相关性 > 0.6（教学简化版门槛）即视为方向正确。
- FF：因子月度均值符号符合经典预期(SMB/HML/RMW/CMA 多为正)、回归 R² 显著高于 CAPM、α 多不显著。
- 三个 notebook 端到端执行无报错、图正常输出。

## 备选方案（已否决）

- B：直接用米筐现成因子暴露做归因 —— 学不到因子构建过程。
- C：只自建少数代表因子 —— 与"完整 CNE5"目标不符。
