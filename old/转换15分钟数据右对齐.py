"""
一次性脚本：将左对齐的15分钟历史数据转换为右对齐，写入新目录（源目录保持不动）。

源数据口径（左对齐）：bar 时间戳 = 开始时间(bob)，每天 09:30...11:15 + 13:00...14:45 共 16 根
真实 bar，另外在 11:30 和 15:00 各补了一根 OHLC 全为 close 的快照 bar，合计 18 行/股/天。

目标口径（右对齐，与左对齐完全镜像）：bar 时间戳 = 结束时间(eob)，每天 09:45...11:30 +
13:15...15:00 共 16 根真实 bar，另外在 09:30 和 13:00 各补一根 OHLC 全为该时段第一根 bar
的 open 的快照 bar（镜像左对齐在时段末尾补 close 快照的逻辑），合计 18 行/股/天。

转换规则：
1. 删除 11:30 和 15:00 两根 close 快照行——右对齐后这两个时间点由真实 bar 占据。
2. 其余行时间戳整体 +15 分钟（开始时间 -> 结束时间），得到右对齐真实 bar。
3. 对每个时段第一根 bar（源数据中 09:30 / 13:00 开始的 bar）额外复制一行开盘快照：
   时间戳保持 09:30 / 13:00，OHLC 全部 = 该 bar 的 open，其余列（volume 等）原样复制
   （与左对齐快照保留原 bar volume 的处理一致）。

源目录:   E:\\working\\stock_data\\15min_stock_data_dir
目标目录: E:\\working\\stock_data\\15min_stock_data_right_dir

支持断点续跑：目标目录中已存在的 trading_date 分区会跳过，可安全重复执行。
（注意：如修改过转换逻辑需要重建，应先清空目标目录再运行。）
"""
import os
import polars as pl

SRC = r'E:\working\stock_data\15min_stock_data_dir'
DST = r'E:\working\stock_data\15min_stock_data_right_dir'

os.makedirs(DST, exist_ok=True)

src_parts = sorted(p for p in os.listdir(SRC) if p.startswith('trading_date='))
done_parts = {p for p in os.listdir(DST) if p.startswith('trading_date=')}
todo = [p for p in src_parts if p not in done_parts]
print(f'源分区 {len(src_parts)} 个, 目标已存在 {len(done_parts)} 个, 待转换 {len(todo)} 个', flush=True)

t = pl.col('datetime')
# close 快照行判定：时间戳恰为 11:30 或 15:00（源数据中真实 bar 不可能落在这两个时间点）
is_close_snap = ((t.dt.hour() == 11) & (t.dt.minute() == 30)) | (
    (t.dt.hour() == 15) & (t.dt.minute() == 0)
)
# 时段首根 bar 判定：源数据中开始时间恰为 09:30 或 13:00 的真实 bar，用于生成开盘快照
is_session_open = ((t.dt.hour() == 9) & (t.dt.minute() == 30)) | (
    (t.dt.hour() == 13) & (t.dt.minute() == 0)
)

total_in, total_out = 0, 0
for i, part in enumerate(todo, 1):
    df = pl.read_parquet(os.path.join(SRC, part))
    real = df.filter(~is_close_snap)
    # 开盘快照：时间戳保持 09:30/13:00 不动，OHLC 全为该 bar 的 open
    open_snap = real.filter(is_session_open).with_columns(
        pl.col('open').alias('high'),
        pl.col('open').alias('low'),
        pl.col('open').alias('close'),
    )
    # 真实 bar：时间戳 +15 分钟（开始时间 -> 结束时间）
    real = real.with_columns((t + pl.duration(minutes=15)).alias('datetime'))
    out = pl.concat([real, open_snap]).sort(['code', 'datetime'])
    out.write_parquet(DST, partition_by=['trading_date'])
    total_in += df.height
    total_out += out.height
    if i % 100 == 0 or i == len(todo):
        print(f'[{i}/{len(todo)}] {part} 完成, 累计输入 {total_in} 行 -> 输出 {total_out} 行', flush=True)

print('转换完成', flush=True)
