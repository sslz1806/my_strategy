# alpha.py 统一未来收益函数实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `因子回测/alpha.py` 中实现一个同时支持 Polars/Pandas、面板/时序的未来收益函数 `add_future_return`，并用它替代 `cal_next_return` 和情绪 Notebook 中的手写实现。

**Architecture:** 在 `alpha.py` 中新增一个公共入口 `add_future_return`，根据输入类型分发到 Polars/Pandas 私有实现；输入若含 `code` 列则按股票分组，否则按单一时序计算。`ret_col` 优先使用，不存在时回退到 `price_col / pre_close_col - 1`。Notebook 删除原有 `add_forward_returns` 函数，直接调用 `alpha.py` 中的新函数并同步更新下游列名引用。

**Tech Stack:** Python 3.9+, Polars, Pandas, NumPy, pytest, Jupyter Notebook

## Global Constraints

- 使用项目指定的 `quant` Python 环境：`E:\working\anaconda3\envs\quant\python.exe`
- 所有收益率统一使用小数形式（1% = 0.01）
- 输出列名统一为 `future_{ret_col}_{n}d`
- `ret_col` 默认 `"pct"`，不存在时回退到 `price_col / pre_close_col - 1`
- 面板数据按 `code_col`（默认 `"code"`）分组，时序数据不分组
- 返回结果按 `[code_col, date_col]` 或 `[date_col]` 排序
- 不改动 `my_utils/stock_api.py`

---

## File Structure

| 文件 | 责任 |
|---|---|
| `因子回测/alpha.py` | 新增 `add_future_return` 及 Polars/Pandas 私有实现；删除 `cal_next_return` |
| `tests/test_add_future_return.py` | 新增单元测试，覆盖两种后端、两种数据形态、收益来源回退、列名、尾部空值 |
| `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb` | 删除手写 `add_forward_returns`；改用 `from 因子回测.alpha import add_future_return`；同步更新 `future_return_*` 列名引用 |

---

### Task 1: 在 `alpha.py` 中实现 `add_future_return`

**Files:**
- Modify: `因子回测/alpha.py`

**Interfaces:**
- Consumes: 无（新建函数）
- Produces: `add_future_return(df, ret_col="pct", price_col="close", pre_close_col="pre_close", horizons=(1, 5, 10, 20), date_col="trading_date", code_col="code")`

- [ ] **Step 1: 读取当前 `alpha.py` 的导入区，确认已有 `polars as pl`、`pandas as pd`、`numpy as np`**

  Run: `Read 因子回测/alpha.py`

- [ ] **Step 2: 在 `cal_next_return` 之前（或文件靠前位置）添加私有 Polars 实现 `_add_future_return_pl`**

  ```python
  def _add_future_return_pl(
      df: pl.DataFrame,
      ret_col: str,
      price_col: str,
      pre_close_col: str,
      horizons: Sequence[int],
      date_col: str,
      code_col: str,
  ) -> pl.DataFrame:
      is_panel = code_col in df.columns

      if is_panel:
          sorted_df = df.sort([code_col, date_col])
      else:
          sorted_df = df.sort(date_col)

      if ret_col in sorted_df.columns:
          gross_expr = pl.col(ret_col) + 1.0
      else:
          gross_expr = pl.col(price_col) / pl.col(pre_close_col)

      exprs = []
      for h in horizons:
          future_expr = gross_expr.rolling_product(window_size=h, min_samples=h).shift(-h)
          if is_panel:
              future_expr = future_expr.over(code_col)
          exprs.append(future_expr.alias(f"future_{ret_col}_{h}d"))

      return sorted_df.with_columns(exprs)
  ```

- [ ] **Step 3: 在同一区域添加私有 Pandas 实现 `_add_future_return_pd`**

  ```python
  def _add_future_return_pd(
      df: pd.DataFrame,
      ret_col: str,
      price_col: str,
      pre_close_col: str,
      horizons: Sequence[int],
      date_col: str,
      code_col: str,
  ) -> pd.DataFrame:
      is_panel = code_col in df.columns

      if is_panel:
          sorted_df = df.sort_values([code_col, date_col]).copy()
      else:
          sorted_df = df.sort_values(date_col).copy()

      if ret_col in sorted_df.columns:
          gross = 1.0 + sorted_df[ret_col].astype(float)
      else:
          gross = sorted_df[price_col].astype(float) / sorted_df[pre_close_col].astype(float)

      sorted_df["__gross_ret__"] = gross

      for h in horizons:
          if is_panel:
              future = sorted_df.groupby(code_col)["__gross_ret__"].transform(
                  lambda s: s.rolling(h, min_periods=h).apply(np.prod, raw=True).shift(-h) - 1.0
              )
          else:
              future = (
                  sorted_df["__gross_ret__"]
                  .rolling(h, min_periods=h)
                  .apply(np.prod, raw=True)
                  .shift(-h)
                  - 1.0
              )
          sorted_df[f"future_{ret_col}_{h}d"] = future

      return sorted_df.drop(columns=["__gross_ret__"])
  ```

- [ ] **Step 4: 添加公共分发函数 `add_future_return`**

  ```python
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

      支持 Polars 和 Pandas 两种后端。若存在 code_col 列则按股票分组计算，
      否则按单一时序计算。ret_col 列存在时直接使用，否则用 price_col / pre_close_col - 1
      计算日收益。输出列名为 future_{ret_col}_{n}d。
      """
      if isinstance(df, pl.DataFrame):
          return _add_future_return_pl(
              df, ret_col, price_col, pre_close_col, horizons, date_col, code_col
          )
      if isinstance(df, pd.DataFrame):
          return _add_future_return_pd(
              df, ret_col, price_col, pre_close_col, horizons, date_col, code_col
          )
      raise TypeError(f"Unsupported DataFrame type: {type(df)}")
  ```

- [ ] **Step 5: 删除 `cal_next_return` 函数**

  Remove lines:
  ```python
  def cal_next_return(stock_data: pl.DataFrame, days=5) -> pl.DataFrame:
      stock_data = stock_data.sort(['code','trading_date'])
      stock_data = stock_data.with_columns([
          ((pl.col('close').shift(-days) - pl.col('close')) / pl.col('close')*100).over('code').alias(f'return_{days}d')
      ])
      return stock_data
  ```

- [ ] **Step 6: 运行语法检查**

  Run: `E:/working/anaconda3/envs/quant/python.exe -m py_compile 因子回测/alpha.py`
  Expected: 无输出（表示语法正确）

- [ ] **Step 7: Commit**

  ```bash
  git add 因子回测/alpha.py
  git commit -m "feat(alpha): add add_future_return with Polars/Pandas support and remove cal_next_return"
  ```

---

### Task 2: 为 `add_future_return` 添加单元测试

**Files:**
- Create: `tests/test_add_future_return.py`

**Interfaces:**
- Consumes: `add_future_return` from `因子回测.alpha`
- Produces: 无

- [ ] **Step 1: 创建测试文件并写入测试代码**

  ```python
  from __future__ import annotations

  import sys
  from datetime import date
  from pathlib import Path

  import numpy as np
  import pandas as pd
  import polars as pl
  import pytest

  PROJECT_ROOT = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(PROJECT_ROOT))

  from 因子回测.alpha import add_future_return


  def make_pd_series():
      return pd.DataFrame(
          {
              "trading_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
              "pct": [0.10, 0.20, -0.10, 0.25],
          }
      )


  def make_pl_series():
      return pl.DataFrame(
          {
              "trading_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
              "pct": [0.10, 0.20, -0.10, 0.25],
          }
      )


  def make_pd_panel():
      return pd.DataFrame(
          {
              "code": ["A", "A", "A", "B", "B", "B"],
              "trading_date": pd.to_datetime(
                  ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-02", "2024-01-03", "2024-01-04"]
              ),
              "pct": [0.10, 0.20, -0.10, 0.05, 0.15, -0.05],
          }
      )


  def make_pl_panel():
      return pl.DataFrame(
          {
              "code": ["A", "A", "A", "B", "B", "B"],
              "trading_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)] * 2,
              "pct": [0.10, 0.20, -0.10, 0.05, 0.15, -0.05],
          }
      )


  def test_pandas_time_series():
      df = make_pd_series()
      result = add_future_return(df, horizons=(1, 3))

      assert np.isclose(result.loc[0, "future_pct_1d"], 0.20)
      assert np.isclose(
          result.loc[0, "future_pct_3d"],
          (1.20 * 0.90 * 1.25) - 1.0,
      )
      assert pd.isna(result.loc[3, "future_pct_1d"])
      assert pd.isna(result.loc[2, "future_pct_3d"])


  def test_polars_time_series():
      df = make_pl_series()
      result = add_future_return(df, horizons=(1, 3))

      assert np.isclose(result["future_pct_1d"][0], 0.20)
      assert np.isclose(
          result["future_pct_3d"][0],
          (1.20 * 0.90 * 1.25) - 1.0,
      )
      assert result["future_pct_1d"][3] is None
      assert result["future_pct_3d"][2] is None


  def test_pandas_panel_groups_by_code():
      df = make_pd_panel()
      result = add_future_return(df, horizons=(1, 2))

      row_a0 = result[(result["code"] == "A") & (result["trading_date"] == "2024-01-02")].iloc[0]
      row_b0 = result[(result["code"] == "B") & (result["trading_date"] == "2024-01-02")].iloc[0]

      assert np.isclose(row_a0["future_pct_1d"], 0.20)
      assert np.isclose(row_b0["future_pct_1d"], 0.15)
      assert np.isclose(row_a0["future_pct_2d"], (1.20 * 0.90) - 1.0)
      assert np.isclose(row_b0["future_pct_2d"], (1.15 * 0.95) - 1.0)


  def test_polars_panel_groups_by_code():
      df = make_pl_panel()
      result = add_future_return(df, horizons=(1, 2))

      a0 = result.filter((pl.col("code") == "A") & (pl.col("trading_date") == date(2024, 1, 2))).row(0, named=True)
      b0 = result.filter((pl.col("code") == "B") & (pl.col("trading_date") == date(2024, 1, 2))).row(0, named=True)

      assert np.isclose(a0["future_pct_1d"], 0.20)
      assert np.isclose(b0["future_pct_1d"], 0.15)
      assert np.isclose(a0["future_pct_2d"], (1.20 * 0.90) - 1.0)
      assert np.isclose(b0["future_pct_2d"], (1.15 * 0.95) - 1.0)


  def test_ret_col_fallback_to_price():
      df = pd.DataFrame(
          {
              "trading_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
              "close": [11.0, 12.0, 10.8],
              "pre_close": [10.0, 11.0, 12.0],
          }
      )
      result = add_future_return(df, ret_col="pct", horizons=(1, 2))

      # pct 列不存在，应回退到 close / pre_close - 1
      assert np.isclose(result.loc[0, "future_pct_1d"], 12.0 / 11.0 - 1.0)
      assert np.isclose(result.loc[0, "future_pct_2d"], (12.0 / 11.0) * (10.8 / 12.0) - 1.0)


  def test_output_column_names_use_ret_col():
      df = pd.DataFrame(
          {
              "trading_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
              "market_daily_ret": [0.01, 0.02],
          }
      )
      result = add_future_return(df, ret_col="market_daily_ret", horizons=(1,))

      assert "future_market_daily_ret_1d" in result.columns
  ```

- [ ] **Step 2: 安装/确认测试依赖并运行测试**

  Run: `E:/working/anaconda3/envs/quant/python.exe -m pytest tests/test_add_future_return.py -v`
  Expected: 7 tests PASS

- [ ] **Step 3: Commit**

  ```bash
  git add tests/test_add_future_return.py
  git commit -m "test(alpha): add unit tests for add_future_return"
  ```

---

### Task 3: 在情绪 Notebook 中替换手写未来收益计算

**Files:**
- Modify: `因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb`

**Interfaces:**
- Consumes: `add_future_return` from `因子回测.alpha`
- Produces: 更新后的 Notebook，删除 `add_forward_returns` 函数并同步列名

- [ ] **Step 1: 读取 Notebook 确认 `add_forward_returns` 函数所在 cell 和调用位置**

  Search:
  - `def add_forward_returns` 所在 cell
  - `add_forward_returns(market_daily` 调用位置
  - `future_return_` 所有引用位置

- [ ] **Step 2: 在 Notebook 的导入 cell 中新增 import**

  在现有 `from my_utils.fun import read_day_data` 附近添加：

  ```python
  from 因子回测.alpha import add_future_return
  ```

- [ ] **Step 3: 删除 `def add_forward_returns(...)` 所在 cell**

  该 cell 包含完整的 `add_forward_returns` 函数定义及 docstring。

- [ ] **Step 4: 替换调用语句**

  将：
  ```python
  market_with_forward = add_forward_returns(market_daily, horizons=HORIZONS)
  ```
  替换为：
  ```python
  market_with_forward = add_future_return(
      market_daily, ret_col="market_daily_ret", horizons=HORIZONS
  )
  ```

- [ ] **Step 5: 同步更新下游列名引用**

  在 Notebook 中搜索所有 `future_return_` 字符串，将其替换为 `future_market_daily_ret_`。具体位置包括：
  - `analyze_ic` 函数内的 `target = f"future_return_{horizon}d"`
  - `compute_rolling_ic` 函数内的 `target = f"future_return_{horizon}d"`
  - 相关 docstring 中的描述文字

  替换后示例：
  ```python
  target = f"future_market_daily_ret_{horizon}d"
  ```

- [ ] **Step 6: 运行 Notebook 测试或执行 Notebook 验证无报错**

  Run: `E:/working/anaconda3/envs/quant/python.exe -m pytest tests/test_sentiment_factors_5d_notebook.py -v`
  Expected: 相关测试 PASS；若测试未覆盖此改动，则使用 `NotebookEdit` 或 `jupyter nbconvert --execute` 执行 Notebook 相关章节。

- [ ] **Step 7: Commit**

  ```bash
  git add 因子回测/涨跌停情绪因子/sentiment_factors_5d_research.ipynb
  git commit -m "refactor(notebook): replace add_forward_returns with add_future_return from alpha.py"
  ```

---

### Task 4: 全局检查无残留引用

**Files:**
- 全项目

**Interfaces:**
- Consumes: 无
- Produces: 无

- [ ] **Step 1: 搜索 `cal_next_return` 残留**

  Run: `grep -r "cal_next_return" --include="*.py" --include="*.ipynb" .`
  Expected: 只剩设计文档/plan 中的历史描述，无实际代码引用

- [ ] **Step 2: 搜索情绪 Notebook 中 `add_forward_returns` 残留**

  Run: `grep -r "add_forward_returns" --include="*.py" --include="*.ipynb" 因子回测/涨跌停情绪因子/`
  Expected: 无匹配

- [ ] **Step 3: 运行新增测试和已有相关测试**

  Run: `E:/working/anaconda3/envs/quant/python.exe -m pytest tests/test_add_future_return.py tests/test_sentiment_factors_5d_notebook.py -v`
  Expected: 全部 PASS

- [ ] **Step 4: Commit（如需要）**

  如无额外改动，此步骤可跳过；如有修复则提交。

---

## Self-Review

**1. Spec coverage:**
- Polars/Pandas 双后端：Task 1 Step 2/3/4 实现，Task 2 测试覆盖。
- 面板/时序自动识别：Task 1 Step 2/3 中通过 `code_col in df.columns` 判断。
- `ret_col` 回退到价格：Task 1 Step 2/3 和 Task 2 `test_ret_col_fallback_to_price` 覆盖。
- 输出列名 `future_{ret_col}_{n}d`：Task 1 Step 2/3 和 Task 2 `test_output_column_names_use_ret_col` 覆盖。
- 删除 `cal_next_return`：Task 1 Step 5。
- Notebook 替换：Task 3。
- 不改动 `my_utils/stock_api.py`：明确列为不纳入范围。

**2. Placeholder scan:**
- 无 TBD/TODO/"implement later"/"add appropriate error handling" 等占位符。
- 所有步骤包含可执行代码或命令。

**3. Type consistency:**
- `add_future_return` 签名在 Task 1 和 Task 2 导入处一致。
- `_add_future_return_pl` 和 `_add_future_return_pd` 参数与公共函数一致。
- Notebook 中调用使用 `ret_col="market_daily_ret"`，与输出列名 `future_market_daily_ret_{n}d` 一致。
