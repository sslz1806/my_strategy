"""
临时脚本 v2：每个查询使用独立 DDB 连接，避免连接断开问题。
"""
import dolphindb as ddb
import sys

HOST = "10.140.5.44"
PORT = 8902
USER = "admin"
PASSWORD = "123456"


def run_ddb(label, query, print_result=True):
    """使用独立连接执行一次 DDB 查询。"""
    s = ddb.session()
    try:
        s.connect(HOST, PORT, USER, PASSWORD)
        result = s.run(query, clearMemory=True)
        if print_result:
            print(f"\n=== {label} ===")
            if hasattr(result, 'to_string'):
                print(result.to_string())
            else:
                print(result)
        return result
    except Exception as e:
        print(f"\n=== {label} ===")
        print(f"查询失败: {e}")
        return None
    finally:
        s.close()


# 逐个查询，每个独立连接
queries = [
    (
        "米筐因子列表",
        'select distinct factor from loadTable("dfs://rq_factor_years_tsdb", "day_factor")'
    ),
    (
        "同花顺因子列表",
        'select distinct factor from loadTable("dfs://ths_factor_years_tsdb", "day_factor")'
    ),
    (
        "Wind 因子列表",
        'select distinct factor from loadTable("dfs://wind_factor_years_tsdb", "day_factor")'
    ),
    (
        "stock_pit schema",
        'schema(loadTable("dfs://stock_years_tsdb", "stock_pit"))'
    ),
    (
        "stock_pit_financials schema",
        'schema(loadTable("dfs://stock_years_tsdb", "stock_pit_financials"))'
    ),
]

results = {}
for label, query in queries:
    results[label] = run_ddb(label, query)

# 汇总因子库中的财务相关因子
print("\n\n" + "=" * 70)
print("汇总：各因子库中包含的因子")
print("=" * 70)

for label in ["米筐因子列表", "同花顺因子列表", "Wind 因子列表"]:
    r = results.get(label)
    if r is not None and not r.empty:
        factors = sorted(r["factor"].dropna().tolist())
        print(f"\n{label}: 共 {len(factors)} 个因子")
    else:
        print(f"\n{label}: 无数据或查询失败")

print("\n完成。")
