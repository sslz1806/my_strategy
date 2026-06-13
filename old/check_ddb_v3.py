"""
v3: 完全模仿 米筐数据更新.py 的连接模式和查询模式。
先测简单查询，再逐步尝试 schema / sample。
"""
import dolphindb as ddb
import time

HOST = "10.140.5.44"
PORT = 8902
USER = "admin"
PASSWORD = "123456"


def test_one(query_sql, label, timeout=60):
    """独立连接执行一次查询，如果失败则返回 None。"""
    s = None
    try:
        s = ddb.session()
        s.connect(HOST, PORT, USER, PASSWORD)
        # 给连接一点时间
        time.sleep(0.5)
        result = s.run(query_sql)
        print(f"[OK] {label} => 类型={type(result).__name__}")
        return result
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        return None
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


# === 阶段 1: 测试连接是否恢复 ===
print("=== 阶段 1: 连接测试 ===\n")

# 尝试 1: 最简单的运算
test_one("1+1", "简单运算 1+1")

# 尝试 2: 看一下 getClusterDFSDatabases
test_one("getClusterDFSDatabases()", "数据库列表")

# 如果上面都失败了，等待 30 秒再试
time.sleep(5)

# 尝试 3: 用与 米筐数据更新.py 完全相同的查询模式
test_one(
    "select order_book_id, symbol from loadTable('dfs://common_years_tsdb', 'instrument_base') where type = 'CS' limit 3",
    "instrument_base 简单查询（与米筐数据更新.py 相同模式）"
)

# === 阶段 2: 如果连接恢复，查询因子 schema ===
print("\n=== 阶段 2: 因子表 schema（最轻量） ===\n")

test_one(
    'schema(loadTable("dfs://rq_factor_years_tsdb", "day_factor"))["colDefs"]',
    "米筐因子表 colDefs"
)

test_one(
    'schema(loadTable("dfs://ths_factor_years_tsdb", "day_factor"))["colDefs"]',
    "同花顺因子表 colDefs"
)

test_one(
    'schema(loadTable("dfs://wind_factor_years_tsdb", "day_factor"))["colDefs"]',
    "Wind 因子表 colDefs"
)

# === 阶段 3: stock_pit ===
print("\n=== 阶段 3: stock_pit 表信息 ===\n")

test_one(
    'schema(loadTable("dfs://stock_years_tsdb", "stock_pit"))["colDefs"]',
    "stock_pit colDefs"
)

test_one(
    'schema(loadTable("dfs://stock_years_tsdb", "stock_pit_financials"))["colDefs"]',
    "stock_pit_financials colDefs"
)

# === 阶段 4: 采样（如果 schema 成功） ===
print("\n=== 阶段 4: 数据采样 ===\n")

test_one(
    'select top 5 * from loadTable("dfs://rq_factor_years_tsdb", "day_factor") '
    'where date = date(2026.05.29) and symbol = "000001.XSHE"',
    "米筐因子采样 (000001, 20260529)"
)

test_one(
    'select top 3 * from loadTable("dfs://stock_years_tsdb", "stock_pit") '
    'where order_book_id = "000001.XSHE"',
    "stock_pit 采样 (000001)"
)

print("\n全部完成。")
