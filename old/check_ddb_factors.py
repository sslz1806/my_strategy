"""
临时脚本：探查 DDB 因子库和 stock_pit 中的财务指标数据。
仅做只读元信息查询，不查大表数据。
"""
import dolphindb as ddb

HOST = "10.140.5.44"
PORT = 8902
USER = "admin"
PASSWORD = "123456"


def main():
    s = ddb.session()
    s.connect(HOST, PORT, USER, PASSWORD)
    print("DDB 连接成功\n")

    # ============================================================
    # 1. 米筐因子库
    # ============================================================
    print("=" * 70)
    print("1. 米筐因子库 (dfs://rq_factor_years_tsdb.day_factor)")
    print("=" * 70)
    try:
        rq = s.run(
            'select distinct factor '
            'from loadTable("dfs://rq_factor_years_tsdb", "day_factor")'
        )
        print(f"因子总数: {len(rq)}\n")
        factors = sorted(rq["factor"].dropna().tolist())
        for f in factors:
            print(f"  {f}")
    except Exception as e:
        print(f"查询失败: {e}")

    print()

    # ============================================================
    # 2. 同花顺因子库
    # ============================================================
    print("=" * 70)
    print("2. 同花顺因子库 (dfs://ths_factor_years_tsdb.day_factor)")
    print("=" * 70)
    try:
        ths = s.run(
            'select distinct factor '
            'from loadTable("dfs://ths_factor_years_tsdb", "day_factor")'
        )
        print(f"因子总数: {len(ths)}\n")
        factors = sorted(ths["factor"].dropna().tolist())
        for f in factors:
            print(f"  {f}")
    except Exception as e:
        print(f"查询失败: {e}")

    print()

    # ============================================================
    # 3. Wind 因子库
    # ============================================================
    print("=" * 70)
    print("3. Wind 因子库 (dfs://wind_factor_years_tsdb.day_factor)")
    print("=" * 70)
    try:
        wind = s.run(
            'select distinct factor '
            'from loadTable("dfs://wind_factor_years_tsdb", "day_factor")'
        )
        print(f"因子总数: {len(wind)}\n")
        factors = sorted(wind["factor"].dropna().tolist())
        for f in factors:
            print(f"  {f}")
    except Exception as e:
        print(f"查询失败: {e}")

    print()

    # ============================================================
    # 4. stock_pit 完整 schema
    # ============================================================
    print("=" * 70)
    print("4. stock_pit 完整 schema (dfs://stock_years_tsdb.stock_pit)")
    print("=" * 70)
    try:
        schema = s.run(
            'schema(loadTable("dfs://stock_years_tsdb", "stock_pit"))'
        )
        print(schema)
    except Exception as e:
        print(f"查询失败: {e}")

    print()

    # ============================================================
    # 5. stock_pit_financials 完整 schema
    # ============================================================
    print("=" * 70)
    print("5. stock_pit_financials 完整 schema "
          "(dfs://stock_years_tsdb.stock_pit_financials)")
    print("=" * 70)
    try:
        schema = s.run(
            'schema(loadTable("dfs://stock_years_tsdb", "stock_pit_financials"))'
        )
        print(schema)
    except Exception as e:
        print(f"查询失败: {e}")

    s.close()
    print("\n完成。")


if __name__ == "__main__":
    main()
