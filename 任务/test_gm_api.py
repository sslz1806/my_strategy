"""
测试掘金API接口 gm_get_daily_data
"""
import sys
sys.path.append("C://Users/20561/Desktop/策略")

import pandas as pd
import datetime

print("=" * 70)
print("掘金API接口测试脚本")
print("=" * 70)

# 测试1: 检查导入
try:
    print("\n[测试1] 导入stock_api模块...")
    from my_utils.stock_api import stock_api
    print("[OK] 导入成功")
except Exception as e:
    print(f"[FAIL] 导入失败: {e}")
    sys.exit(1)

# 测试2: 初始化API
try:
    print("\n[测试2] 初始化stock_api...")
    api = stock_api()
    print("[OK] 初始化成功")
except Exception as e:
    print(f"[FAIL] 初始化失败: {e}")
    sys.exit(1)

# 测试3: 获取单日数据
try:
    print("\n[测试3] 获取单日数据 (2025-04-09)...")
    df = api.gm_get_daily_data_multi_dates('2025-04-01', '2025-04-09')

    if df is not None and not df.empty:
        print(f"[OK] 数据获取成功!")
        print(f"  - 记录数: {len(df)}")
        print(f"  - 字段数: {len(df.columns)}")
        print(f"\n  字段列表:")
        for i, col in enumerate(df.columns, 1):
            non_null = df[col].notna().sum()
            print(f"    {i:2d}. {col:20s} (非空: {non_null:5d}/{len(df)})")

        print(f"\n  前3条数据预览:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.head(3).to_string())

        print(f"\n  数据统计:")
        print(f"    - 唯一股票数: {df['code'].nunique()}")
        print(f"    - 日期范围: {df['trading_date'].min()} ~ {df['trading_date'].max()}")
        if 'limit_up' in df.columns:
            print(f"    - 涨停价范围: {df['limit_up'].min():.2f} ~ {df['limit_up'].max():.2f}")
        if 'is_st' in df.columns:
            print(f"    - ST股票数量: {df['is_st'].sum()}")
        if 'mv_A_free_float' in df.columns:
            print(f"    - 流通市值范围: {df['mv_A_free_float'].min()/1e8:.2f}亿 ~ {df['mv_A_free_float'].max()/1e8:.2f}亿")
    else:
        print("[FAIL] 数据获取失败或返回空数据")
        sys.exit(1)

except Exception as e:
    print(f"[FAIL] 获取数据时出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 批量获取多日数据
print("\n[测试4] 批量获取多日数据 (最近3个交易日)...")
try:
    # 获取最近3个交易日
    today = datetime.date.today()
    dates = []
    for i in range(10):  # 往前找10天
        d = today - datetime.timedelta(days=i)
        if d.weekday() < 5:  # 周一到周五
            dates.append(d.strftime('%Y-%m-%d'))
        if len(dates) >= 3:
            break

    print(f"  准备获取日期: {dates}")

    df_multi = api.gm_get_daily_data_multi_dates(dates[-1], dates[0])

    if df_multi is not None and not df_multi.empty:
        print(f"[OK] 批量数据获取成功!")
        print(f"  - 总记录数: {len(df_multi)}")
        print(f"  - 交易日数量: {df_multi['trading_date'].nunique()}")
        print(f"  - 每日股票数量统计:")
        daily_counts = df_multi.groupby('trading_date').size()
        for date, count in daily_counts.items():
            print(f"      {date}: {count} 只")
    else:
        print("[FAIL] 批量数据获取失败")

except Exception as e:
    print(f"[FAIL] 批量获取时出错: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成!")
print("=" * 70)
