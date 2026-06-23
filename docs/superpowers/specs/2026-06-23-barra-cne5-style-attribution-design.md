# Barra CNE5 全市场风格归因 学习 Notebook — 设计文档

> 日期：2026-06-23
> 类型：学习 Notebook + 可复用模块
> 参考：`因子回测/learn/barra_use3.ipynb`（CNE5 风险模型构建 + 等权组合风险归因）

## 一、目标

barra_use3 已经把 CNE5 的「10 风格 + 行业 + 国家」风险模型从零搭好，并在结尾对**等权组合**做了一次风险归因（拆方差）。本项目"更近一步"，产出一本**综合学习 notebook**，用同一套 Barra 因子对 **A 股全市场（也可切指定指数）** 做完整的**风格归因**，三段式：

1. **风格画像（暴露追踪）**：目标组合的风格"长相"随时间怎么变。
2. **收益归因（绝对 + 主动）**：把目标组合 1 年的累计收益拆成 国家 + 行业 + 10 风格 + 特质 各自贡献多少。
3. **风险归因（绝对 + 主动）**：把目标组合的年化波动拆成各因子 + 特质贡献。

绝对 = 拆目标组合自身收益/风险；主动 = 目标组合相对全市场基准的超额收益/风险来自哪些风格。

## 二、已确认决策

| 决策点 | 选择 | 说明 |
|---|---|---|
| 归因类型 | 综合（画像 + 收益 + 风险） | 三段式 Part 化 |
| 基准口径 | 绝对 + 主动 都做 | 主动基准 = 全市场市值加权 |
| 回归域（估因子收益） | **中证全指 `000985.XSHG`** 期初快照 | 代表"全市场"，约 5000+ 只，已按指数规则剔 ST/次新/极差流动性 |
| 截面频率 | **日频** | 1 年≈245 个截面点，因子协方差可用；标准 Barra 口径 |
| 归因目标 `TARGET` | 默认 `whole_market_cap`，可切 `whole_market_eq` / 指数代码 | 见 §四 目标组合构建 |
| 代码组织 | **B：抽公共模块** `因子回测/learn/barra_core.py` | 新 notebook import 它 |
| barra_use3 | **不动** | 保留其逐格教实现的学习价值；barra_core 与其内联代码有一份合理重复 |

## 三、架构

### 3.1 `因子回测/learn/barra_core.py`（新建，可复用模型机件）

只装"Barra 模型机件"——即 barra_use3 Part 2~4 的逻辑，抽成不依赖全局变量的纯函数，供新 notebook（以及未来其它 barra 应用）import。**不**包含归因逻辑（归因是新 notebook 的教学正文，内联写出）。

```python
STYLES = ['size', 'beta', 'momentum', 'residual_volatility', 'non_linear_size',
          'book_to_price', 'liquidity', 'earnings_yield', 'growth', 'leverage']

def winsorize_mad(s, n=3.0) -> pd.Series
    """MAD 去极值，迁移自 barra_use3。"""

def standardize(s, ind, mcap) -> pd.Series
    """三步标准化：行业中位数填缺 → MAD 去极值 → 市值加权 z-score。迁移自 barra_use3。"""

def compute_style_descriptors(close, volume, free_circ, mktcap_d, pb_d, pe_d,
                              leverage_d, profit_growth_d, rev_growth_d,
                              asset_growth_d, idx_close, win=252) -> dict
    """向量化滚动计算所有风格描述子面板（date×code），返回 dict：
       {'size_d','beta_d','momentum_d','dailyvol_d','btop_d','liquidity_d',
        'etop_d','leverage_d','mktcap_d','profit_growth_d','rev_growth_d',
        'asset_growth_d'}。迁移自 barra_use3 Part 2。"""

def build_exposures(d, desc, industry_map) -> (Z: DataFrame[N×10], mcap: Series, ind: Series)
    """单截面日 d 的 10 因子暴露矩阵。desc 为 compute_style_descriptors 输出。
       含 non_linear_size / residual_volatility / growth 的正交化与复合。迁移自 barra_use3 Part 3。"""

def cs_factor_returns(d, fwd_ret, desc, industry_map, min_n=100) -> (out: Series, resid: Series) | None
    """单期截面 WLS：设计矩阵 [行业哑变量 | 10 风格]，权重 √mcap，
       返回 10 风格收益 + country（行业市值加权均值）以及特质收益。迁移自 barra_use3 Part 4。"""
```

迁移原则：函数体与 barra_use3 等价，仅把原来依赖的 module 级全局（`size_d`、`beta_d`…）改为通过 `desc` 字典传入，消除隐式全局依赖。行为对齐，便于 barra_use3 的口径 caveats 直接继承。

### 3.2 `因子回测/learn/barra_cne5_风格归因.ipynb`（新建，学习正文）

顶部统一配置区（一处改全局）：

```python
UNIVERSE_INDEX = '000985.XSHG'      # 回归域 = 中证全指
MARKET_INDEX   = '000985.XSHG'      # Beta 的市场基准 R_t（与回归域一致）
DATA_START     = '2024-05-01'       # 数据起点：归因窗口前移 ≥252 交易日（留足余量，避免窗口初期滚动因子算不满）
WINDOW_START   = '2025-06-23'       # 归因窗口起（最近 1 年）
WINDOW_END     = '2026-06-20'       # 归因窗口止（取实际最后交易日）
TARGET         = 'whole_market_cap' # 'whole_market_cap' | 'whole_market_eq' | '000300.XSHG' 等指数代码
FAST_MODE      = False              # True 时缩小 universe/缩短窗口，快速试跑
```

实际日期在运行时按交易日历对齐；`FAST_MODE` 用于在全市场+日频较重时先跑通流程。

## 四、Notebook 结构（Part 化，对齐 barra_use3 风格）

| Part | 标题 | 内容 | 与 barra_use3 关系 |
|---|---|---|---|
| 0 | 环境准备 | 依赖 + 米筐 init + 中文字体 + 项目根目录 + `from my_utils.rqdata import RqData` + `import barra_core` | 复用，新增 import barra_core |
| 1 | 全市场数据准备 | 中证全指成分**期初快照**；`close`(后复权)/`volume`(不复权)/自由流通股本/基本面 7 字段；市场基准 = 中证全指收盘 | 扩到全市场；三种复权口径坑沿用 |
| 2 | 风格描述子 | 调 `barra_core.compute_style_descriptors(...)`；标注耗时/内存 | 逻辑入模块 |
| 3 | 暴露矩阵 | 调 `barra_core.build_exposures(cs_dates[-1], desc, industry_map)` demo 展示 | 逻辑入模块 |
| 4 | **日频截面 WLS 求因子收益** | 逐**交易日**调 `barra_core.cs_factor_returns`，存 `style_ret`(T×11)、`spec_ret`(T×N)；画风格累计收益（"这一年谁在赚钱"）+ 月均/t 值表 | 月频→日频，最重循环 |
| 5 | 官方暴露对照验证 | 取一个全市场截面与米筐 `get_factor_exposure` 逐因子求横截面相关 | 复用 |
| 6 | **目标组合 + 风格画像（暴露追踪）** | 按 `TARGET` 构建日频权重 `w_i(t)`；日频组合暴露 `x_k(t)=Σ w_i(t)X_ik(t)` 与主动暴露 `x_k^act(t)`；折线图。教学点：市值加权全市场 `x_k≈0`、指数有持续偏离 | **新增** |
| 7 | **收益归因（绝对 + 主动）** | 日链式 `贡献_k(t)=x_k(t)·f_k(t)`；累计柱状 + 堆叠面积；绝对口径每日恒等闭合到组合收益，主动口径 country 抵消、看超额来源 | **新增** |
| 8 | **风险归因（绝对 + 主动）** | 日频因子收益估 `F`(×252) 与特质方差(×252)；`x'Fx` 拆因子贡献 + 特质；主动版本用 `x^act` | 升级 barra_use3 Part 6-7 到真实目标 |
| 9 | 学习小结 | 技术版 + 通俗版（对齐 barra_use3 结尾风格） | 对齐 |

## 五、目标组合构建与归因数学

### 5.1 目标权重 `w_i(t)`（日频）

- `whole_market_cap`：`w_i(t) = mcap_i(t) / Σ_{j∈universe} mcap_j(t)`（全市场即时市值加权，"市场组合"定义）。
- `whole_market_eq`：`w_i(t) = 1/N(t)`（等权，自然超配小盘，风格暴露非零）。
- 指数代码（如 `000300.XSHG`）：取该指数成分**期初快照**（必 ⊆ 中证全指），在成分内按自由流通市值加权 `w_i(t) = mcap_i(t) / Σ_{j∈index} mcap_j(t)`。
  - **简化**：用自有市值加权近似真实指数权重（不含官方调整因子/权重上限）；成分用期初快照（survivorship 简化）。

主动基准 = `whole_market_cap`；主动权重 `w_i^act(t) = w_i^target(t) − w_i^mkt(t)`。

### 5.2 收益归因（精确恒等，算术日链）

时点约定：暴露取 `t`，因子收益取 `t→t+1` 的前向收益反解出的 `f_k(t)`。组合 `t→t+1` 收益：

```
r_p(t) = Σ_i w_i(t) r_i(t→t+1)
       = Σ_k x_k(t) f_k(t) + Σ_i w_i(t) u_i(t)          （每日精确成立，u_i 为回归残差）
其中 x_k(t) = Σ_i w_i(t) X_ik(t)
```

- 因子贡献 = `x_k(t) f_k(t)`（含 country 与各行业聚合项）；特质贡献 = `Σ_i w_i(t) u_i(t)`。
- 累计 = 各日贡献**算术求和**；每日恒等 ⇒ 累计 `Σ因子贡献 + 特质 ≡ 累计组合算术收益`（验证点，残差应 ~1e-12）。
- 几何复利的多期 linking 残差会在文字中说明（教学简化，不做 Carino/Menchero 平滑）。
- 主动口径：用 `w^act` 代入，country 暴露 `=1−1=0` 自动抵消，呈现"超额来自哪些风格/行业"。

### 5.3 风险归因（点时风险预测，对齐 barra_use3 Part 6-7）

- `F` = 日频因子收益（styles + country）样本协方差 ×252；`σ²(u_i)` = 日频残差方差 ×252。
- 取最后截面日的暴露 `x`（point-in-time 风险快照）：因子方差 `= x'Fx`，各因子贡献 `= x ⊙ (Fx)`；特质方差 `= Σ w_i² σ²(u_i)`。
- 主动风险：用 `x^act` 与 `w^act` 同式计算。
- **简化说明**沿用 barra_use3：未做 Newey-West / VRA / Eigenfactor / 贝叶斯收缩，只做朴素样本协方差年化。

## 六、口径 caveats

继承 barra_use3：动量 231 日窗口（非原文 504）、growth 用 ttm 同比代理（缺分析师预测）、earnings_yield/leverage 仅主描述子、residual_volatility 多正交了 size、全程用原始收益未扣无风险利率。

本项目新增：
- universe / 指数成分用**期初快照**（survivorship 简化，未做成分动态调整）。
- 指数目标用**自有自由流通市值加权**近似真实成分权重（无官方调整因子/权重上限）。
- 多期收益归因用**算术日链**（每日恒等精确闭合；几何复利 linking 残差仅文字说明）。
- 全市场 + 日频数据量大：Part 顶部标注预计耗时/内存，提供 `FAST_MODE`。

## 七、验证计划

在 conda `quant` 环境跑通整本（需联网 rqdatac，解释器 `E:\working\anaconda3\envs\quant\python.exe`）。检查点：

1. 数据：中证全指成分数、交易日数、面板形状非空且合理。
2. 因子收益序列：`style_ret` 形状 ≈ 245×11，无整列 NaN。
3. 官方暴露对照：主因子（size/beta/book_to_price/liquidity）横截面相关 >0.7。
4. **收益归因恒等式**：每日 `Σ因子贡献 + 特质 − 组合算术收益` 绝对值 < 1e-8（关键正确性验证）。
5. **风险归因**：因子方差 + 特质方差 ≈ 组合总方差；占比合理。
6. 教学点自检：`TARGET='whole_market_cap'` 时各风格暴露 `|x_k|` 接近 0、归因近乎全 country；切 `000300.XSHG` 后 size 等出现明显非零偏离。

若全市场 + 日频在本机过重，先以 `FAST_MODE=True`（缩 universe/窗口）跑通验证恒等式，再放大。

## 八、交付物

1. `因子回测/learn/barra_core.py` — 可复用 Barra 模型机件模块。
2. `因子回测/learn/barra_cne5_风格归因.ipynb` — 综合学习 notebook（9 个 Part）。
3. barra_use3.ipynb **不改动**。
