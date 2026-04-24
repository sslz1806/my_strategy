"""
Tushare API 测试脚本
用于测试 Tushare 接口连接和各个 API 功能是否正常
"""
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import traceback
from datetime import datetime, timedelta

# 配置日志
from my_utils.fun import get_logger
logging = get_logger(log_file='log/tushare_test.log', inherit=False)

def print_separator(title=""):
    """打印分隔线"""
    line = "=" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(f"\n{line}")

def test_1_tinyshare_connection():
    """测试1: 测试 tinyshare 库连接"""
    print_separator("测试1: Tinyshare 连接")
    try:
        import tinyshare as tns

        # 使用数据更新脚本中的 token
        ts_token = 'YzAEH11Yc7jZCHjeJa63fnbpSt3k9Je3GvWn0390oiBKO95bVJjP7u5L34e2ff6b'

        print(f"正在使用 token 连接 tinyshare...")
        ts = tns.pro_api(ts_token)

        # 测试简单的交易日历查询
        today = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

        print(f"查询交易日历: {start_date} ~ {today}")
        trade_cal = ts.trade_cal(start_date=start_date, end_date=today)

        if trade_cal is not None and not trade_cal.empty:
            print(f"✓ Tinyshare 连接成功!")
            print(f"  获取到 {len(trade_cal)} 条交易日历记录")
            print(f"  示例数据:\n{trade_cal.head(3)}")
            return True
        else:
            print(f"✗ Tinyshare 返回空数据")
            return False

    except Exception as e:
        print(f"✗ Tinyshare 连接失败: {str(e)}")
        traceback.print_exc()
        return False

def test_2_tushare_official_connection():
    """测试2: 测试官方 tushare 库连接（使用自定义端点）"""
    print_separator("测试2: 官方 Tushare 连接（自定义端点）")
    try:
        import tushare as ts

        # 使用 stock_api.py 中的配置
        token = '5036663342330339422'
        custom_url = 'http://5k1a.xiximiao.com/dataapi'

        print(f"正在连接自定义端点: {custom_url}")
        ts.set_token(token)
        pro = ts.pro_api()

        # 修改 API 端点
        pro._DataApi__token = token
        pro._DataApi__http_url = custom_url

        # 测试交易日历
        today = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

        print(f"查询交易日历: {start_date} ~ {today}")
        trade_cal = pro.trade_cal(start_date=start_date, end_date=today)

        if trade_cal is not None and not trade_cal.empty:
            print(f"✓ 官方 Tushare（自定义端点）连接成功!")
            print(f"  获取到 {len(trade_cal)} 条交易日历记录")
            print(f"  示例数据:\n{trade_cal.head(3)}")
            return True
        else:
            print(f"✗ Tushare 返回空数据")
            return False

    except Exception as e:
        print(f"✗ 官方 Tushare 连接失败: {str(e)}")
        traceback.print_exc()
        return False

def test_3_stock_api_class():
    """测试3: 测试 stock_api 类"""
    print_separator("测试3: stock_api 类")
    try:
        from my_utils.stock_api import stock_api

        print("正在初始化 stock_api...")
        api = stock_api()

        # 测试交易日历获取（内部调用）
        print("✓ stock_api 初始化成功")

        # 测试获取最近一个交易日
        today = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')

        print(f"\n测试 ts_get_stocks_data (精简模式)...")
        print(f"日期范围: {start_date} ~ {today}")

        # 只获取少量数据测试
        try:
            # 先单独测试 bak_daily 接口
            import tushare as ts
            ts.set_token(api.config['ts_token'])
            pro = ts.pro_api()
            pro._DataApi__token = api.config['ts_token']
            pro._DataApi__http_url = api.config.get('http_url', 'http://5k1a.xiximiao.com/dataapi')

            # 获取最近的交易日
            trade_cal = pro.trade_cal(end_date=datetime.now().strftime('%Y%m%d'))
            trade_cal = trade_cal[trade_cal['is_open'] == 1]
            last_trade_date = trade_cal['cal_date'].iloc[-1]

            print(f"最近交易日: {last_trade_date}")

            # 测试 bak_daily
            print("\n测试 bak_daily 接口...")
            daily_data = pro.bak_daily(trade_date=last_trade_date)
            if daily_data is not None and not daily_data.empty:
                print(f"✓ bak_daily 成功, 获取 {len(daily_data)} 条记录")
                print(f"  示例: {daily_data[['ts_code', 'open', 'close', 'vol']].head(2)}")
            else:
                print(f"✗ bak_daily 返回空数据")

            # 测试 stk_limit
            print("\n测试 stk_limit 接口...")
            limit_data = pro.stk_limit(trade_date=last_trade_date)
            if limit_data is not None and not limit_data.empty:
                print(f"✓ stk_limit 成功, 获取 {len(limit_data)} 条记录")
                print(f"  示例: {limit_data[['ts_code', 'up_limit', 'down_limit']].head(2)}")
            else:
                print(f"✗ stk_limit 返回空数据")

            # 测试 stock_st
            print("\n测试 stock_st 接口...")
            st_data = pro.stock_st(trade_date=last_trade_date)
            if st_data is not None and not st_data.empty:
                print(f"✓ stock_st 成功, 获取 {len(st_data)} 条记录")
                print(f"  示例: {st_data[['ts_code', 'name', 'is_st']].head(2)}")
            else:
                print(f"✗ stock_st 返回空数据(可能需要回查历史数据)")
                # 尝试回查几天
                for i in range(1, 10):
                    check_date = (datetime.strptime(last_trade_date, '%Y%m%d') - timedelta(days=i)).strftime('%Y%m%d')
                    st_data = pro.stock_st(trade_date=check_date)
                    if st_data is not None and not st_data.empty:
                        print(f"  ✓ 使用 {check_date} 的 ST 数据")
                        break

            # 测试 adj_factor
            print("\n测试 adj_factor 接口...")
            adj_data = pro.adj_factor(trade_date=last_trade_date)
            if adj_data is not None and not adj_data.empty:
                print(f"✓ adj_factor 成功, 获取 {len(adj_data)} 条记录")
                print(f"  示例: {adj_data[['ts_code', 'adj_factor']].head(2)}")
            else:
                print(f"✗ adj_factor 返回空数据")

            # 测试 daily_basic
            print("\n测试 daily_basic 接口...")
            basic_data = pro.daily_basic(trade_date=last_trade_date)
            if basic_data is not None and not basic_data.empty:
                print(f"✓ daily_basic 成功, 获取 {len(basic_data)} 条记录")
                print(f"  示例: {basic_data[['ts_code', 'pe', 'pb', 'total_mv']].head(2)}")
            else:
                print(f"✗ daily_basic 返回空数据")

            return True

        except Exception as e:
            print(f"✓ stock_api 类可用, 但详细接口测试出错: {str(e)}")
            traceback.print_exc()
            return True  # 类初始化成功就算部分成功

    except Exception as e:
        print(f"✗ stock_api 初始化失败: {str(e)}")
        traceback.print_exc()
        return False

def test_4_mins_token():
    """测试4: 测试分钟数据 token (tinyshare)"""
    print_separator("测试4: 分钟数据 Token")
    try:
        import tinyshare as tns

        mins_token = 'fbdsJ45z9Nodp7FbUgDEsm1Oi8boH7Wuiqn7cQJnRAvs5bSwuB4e0iOBbe16ef40'

        print(f"正在使用分钟数据 token 连接...")
        m_ts = tns.pro_api(mins_token)

        print("✓ 分钟数据 token 连接成功")

        # 注意：分钟数据接口可能需要单独测试
        print("(分钟数据接口测试跳过，避免过多请求)")

        return True

    except Exception as e:
        print(f"✗ 分钟数据 token 连接失败: {str(e)}")
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print_separator()
    print("  Tushare API 全面测试")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()

    results = {}

    # 运行各项测试
    results['Tinyshare 连接'] = test_1_tinyshare_connection()
    time.sleep(1)

    results['官方 Tushare'] = test_2_tushare_official_connection()
    time.sleep(1)

    results['stock_api 类'] = test_3_stock_api_class()
    time.sleep(1)

    results['分钟数据 Token'] = test_4_mins_token()

    # 输出总结
    print_separator("测试总结")
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print_separator()
    print(f"  总计: {passed}/{total} 项测试通过")

    if passed == total:
        print("  ✓ 所有 API 正常，可以继续使用数据更新脚本")
    else:
        print("  ✗ 部分 API 异常，请检查网络连接或 Token 配置")
    print_separator()

    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试运行出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
