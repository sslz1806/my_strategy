# 策略项目简述

用于A股选股信号生成与回测的小型研究仓库，含信号筛选、交易撮合、风控和可视化示例。

## 目录概览
- 回测与信号：回测demo.ipynb(我的第一个策略回测)、策略信号查看.ipynb、N字策略.ipynb
- 因子研究：因子回测/ 内多份 Notebook 与 alpha.py
- 核心脚本：fun.py（数据/特征）、trade_fun.py（回测撮合）、mapping.py（字段/代码）、pd_fun.py（pandas 版）、stock_api.py（数据抓取）
- 产出与日志：信号文件/（csv）、信号交割复盘/（html）、回测.log、function_output.log

## 依赖与数据
- Python 3.10+；主要库：polars、pandas、numpy、plotly、tqdm、scipy、akshare、gm.api、tinyshare。
- 数据根目录在 fun.py 的 DATA_ROOT_DIR（默认 E:/working/stock_data），需准备 parquet：ts_stock_all_data、ts_daily_basic、ts_adj、15min_stock_data_dir。

## 快速开始（回测demo.ipynb）
1) 安装依赖并放好数据到 DATA_ROOT_DIR。
2) 运行 Notebook：读取数据 → 生成特征（涨停/断板、均线、最低价等）→ 设置 params_dict 低开筛选 → 调用 cal_trade_info 生成交割单（示例含权重/买点下移风控）→ report_backtest_full 输出净值、回撤、夏普、胜率并画图。
3) 输出查看：信号文件/ 下的 csv。

## 关键入口
- 数据与特征：fun.py
- 回测撮合：trade_fun.py
- 代码/日期格式：mapping.py
- 因子研究：因子回测/ 目录

## 备注
内部研究用途，未设置开源许可证。
