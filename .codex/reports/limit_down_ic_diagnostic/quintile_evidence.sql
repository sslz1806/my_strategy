-- 图表聚合值来自 sentiment_factors_5d_research.ipynb 的 2026-07-21 同频复算。
-- 本查询只为便携报告提供可执行、可审计的图表数据层，不替代原始 Notebook 来源。
SELECT 'Q1（最低）' AS quintile, -0.00075930 AS mean_return, -0.00408163 AS median_return, 37 AS n_obs
UNION ALL
SELECT 'Q2', 0.00380389, 0.00920621, 36
UNION ALL
SELECT 'Q3', 0.00892007, 0.00847409, 36
UNION ALL
SELECT 'Q4', -0.00052197, 0.00185492, 36
UNION ALL
SELECT 'Q5（最高）', 0.00887495, 0.01566696, 36;
