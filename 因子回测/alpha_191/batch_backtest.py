"""
因子回测/alpha_191/batch_backtest.py - Alpha191 批量回测脚本

对所有 191 个因子运行 2021-2025 回测，采集 IC/IR 等指标，
汇总为 HTML 报告并通过邮件发送。
"""

import sys, os, time, json, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from 因子回测.alpha_191.adapter import load_factor_data
from 因子回测.alpha_191.alpha_formulas import Alpha191Formulas
from my_utils.email_fun import send_email

START_DATE = '2021-01-01'
END_DATE = '2025-07-01'
RETURN_PERIOD = 5
MIN_RECORDS = 500
MAX_STOCKS = 300
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(OUT_DIR, 'batch_results.json')
REPORT_FILE = os.path.join(OUT_DIR, 'alpha191_backtest_report.html')

SLOW = {4, 7, 17, 25, 26, 31, 35, 38, 39, 43, 48, 52, 57, 58, 59, 61,
        66, 68, 71, 72, 73, 75, 77, 81, 84, 85, 86, 88, 91, 92, 94, 95,
        96, 98, 108, 115, 117, 118, 119, 120, 138, 139}

t_start = time.time()
print(f"Alpha191 批量回测  {START_DATE}~{END_DATE}  持仓{RETURN_PERIOD}天")

# 加载全量数据
raw_data = load_factor_data(START_DATE, END_DATE, min_records=MIN_RECORDS)
n_dates = len(raw_data['close'])
print(f"全量: {n_dates}天, {len(raw_data['close'].columns)}只")

# 选TOP-N股票
cnt = raw_data['close'].notna().sum(axis=0).sort_values(ascending=False)
sel = cnt.head(MAX_STOCKS).index.tolist()
print(f"选取TOP-{MAX_STOCKS}", end='')

# 缩小每个DataFrame
data = {}
for k, v in raw_data.items():
    if isinstance(v, pd.DataFrame):
        data[k] = v[sel].copy()
    else:
        data[k] = v
close_df = data['close']
print(f", 最终{len(sel)}只")

# 创建新的 formulas（！！！关键：用缩小后的数据，不是全量的）
formulas = Alpha191Formulas(data)
print(f"交易日: {n_dates}, 股票: {len(sel)}")
print()

# 加载检查点
completed = set()
results = []
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
    completed = {r['alpha'] for r in results if r['status'] == 'ok'}
    print(f"检查点: {len(completed)}个已完成\n")

def compute_ic(n):
    """计算单个因子IC"""
    t0 = time.time()
    try:
        # 使用局部formulas对象计算
        fdf = getattr(formulas, f'alpha_{n:03d}_df')()
        if fdf is None or fdf.isna().all().all():
            return {'alpha': n, 'status': 'skip', 'reason': '全空', 'time': round(time.time()-t0, 1)}

        valid = fdf.notna().sum(axis=1) >= 30
        fv = fdf.loc[valid]
        cv = close_df.loc[fv.index]
        cols = fv.columns.intersection(cv.columns)
        fv, cv = fv[cols], cv[cols]

        ret5 = cv.shift(-RETURN_PERIOD) / cv - 1
        idx = fv.index.intersection(ret5.dropna(how='all').index)
        fv, ret5 = fv.loc[idx], ret5.loc[idx]

        if len(fv) < 5:
            return {'alpha': n, 'status': 'skip', 'reason': f'天数<5({len(fv)})', 'time': round(time.time()-t0, 1)}

        ic = fv.corrwith(ret5, axis=1).dropna()
        ric = fv.rank(axis=1).corrwith(ret5.rank(axis=1), axis=1).dropna()

        if len(ic) < 5:
            return {'alpha': n, 'status': 'skip', 'reason': f'IC天数<5', 'time': round(time.time()-t0, 1)}

        im, istd = ic.mean(), ic.std()
        rm, rstd = ric.mean(), ric.std()
        return {
            'alpha': n, 'status': 'ok',
            'ic_mean': float(im), 'ic_ir': float(im/istd) if istd > 0 else None,
            'rank_ic_mean': float(rm), 'rank_ic_ir': float(rm/rstd) if rstd > 0 else None,
            'ic_pos_ratio': float((ic > 0).mean()),
            'rank_ic_pos_ratio': float((ric > 0).mean()),
            'n_obs': len(ic), 'time': round(time.time()-t0, 1),
        }
    except Exception as e:
        return {'alpha': n, 'status': 'error', 'reason': str(e)[:100], 'time': round(time.time()-t0, 1)}

# 主循环
for n in range(1, 192):
    if n in completed:
        print(f"  [{n}/191] a_{n:03d} 缓存跳过")
        continue
    if n in SLOW:
        results.append({'alpha': n, 'status': 'skip', 'reason': '慢因子', 'time': 0})
        print(f"  [{n}/191] a_{n:03d} ⏭️ 慢因子")
        continue

    r = compute_ic(n)
    results.append(r)
    s = r['status']
    if s == 'ok':
        print(f"  [{n}/191] a_{n:03d} ✅ IC={r['ic_mean']:.4f} IR={r['ic_ir']:.2f} ({r['time']}s)")
    elif s == 'skip':
        print(f"  [{n}/191] a_{n:03d} ⏭️ {r.get('reason','')} ({r['time']}s)")
    else:
        print(f"  [{n}/191] a_{n:03d} ❌ {r.get('reason','')} ({r['time']}s)")

    if n % 10 == 0:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        ok_n = sum(1 for r in results if r['status'] == 'ok')
        print(f"  --- CP: {ok_n}ok/{n}done/{time.time()-t_start:.0f}s ---")

with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 汇总
ok_r = [r for r in results if r['status'] == 'ok']
sk_r = [r for r in results if r['status'] == 'skip']
er_r = [r for r in results if r['status'] == 'error']
elapsed = time.time() - t_start
print(f"\n完成! OK={len(ok_r)} Skip={len(sk_r)} Err={len(er_r)} 耗时={elapsed:.0f}s")

ok_r.sort(key=lambda x: x['ic_mean'] or 0, reverse=True)
print("\n--- Top10 ---")
for r in ok_r[:10]:
    print(f"  a{r['alpha']:03d}: IC={r['ic_mean']:.4f} IR={r['ic_ir']}")
print("\n--- Bottom10 ---")
for r in reversed(ok_r[-10:]):
    print(f"  a{r['alpha']:03d}: IC={r['ic_mean']:.4f} IR={r['ic_ir']}")

# HTML
def _fmt(v, f='.4f'):
    return f'{v:{f}}' if v is not None else '<span class=na>N/A</span>'
def _c(v):
    if v is None: return ''
    if v > 0.02: return 'color:#27ae60;font-weight:bold'
    if v > 0.01: return 'color:#2ecc71'
    if v > 0: return 'color:#7f8c8d'
    return 'color:#e74c3c'

ic_v = [r['ic_mean'] for r in ok_r if r['ic_mean'] is not None]
ir_v = [r['ic_ir'] for r in ok_r if r['ic_ir'] is not None]

rows = ''.join(
    f'<tr><td>{r["alpha"]}</td><td style="{_c(r["ic_mean"])}">{_fmt(r["ic_mean"])}</td>'
    f'<td>{_fmt(r["ic_ir"],".2f")}</td><td>{_fmt(r["rank_ic_mean"])}</td>'
    f'<td>{_fmt(r["rank_ic_ir"],".2f")}</td><td>{_fmt(r["ic_pos_ratio"],".0%")}</td>'
    f'<td>{r["time"]}s</td></tr>\n' for r in ok_r
)
skip_h = '<br>'.join(f'<span class=skip>a{r["alpha"]:03d}</span> <small>{r.get("reason","")}</small>' for r in sk_r)
err_h = '<br>'.join(f'<span class=err>a{r["alpha"]:03d}</span> <small>{r.get("reason","")}</small>' for r in er_r)

dists = [
    ('>0.03', sum(1 for v in ic_v if v > 0.03)),
    ('0.02~0.03', sum(1 for v in ic_v if 0.02 < v <= 0.03)),
    ('0.01~0.02', sum(1 for v in ic_v if 0.01 < v <= 0.02)),
    ('0~0.01', sum(1 for v in ic_v if 0 < v <= 0.01)),
    ('-0.01~0', sum(1 for v in ic_v if -0.01 < v <= 0)),
    ('<=-0.01', sum(1 for v in ic_v if v <= -0.01)),
]
dist = ''.join(f'<tr><td>{l}</td><td>{c}</td><td>{c/len(ic_v)*100:.0f}%</td></tr>\n' for l, c in dists)

ir_p = sum(1 for v in ir_v if v > 1)
ir_m = sum(1 for v in ir_v if 0.5 < v <= 1)
ir_w = sum(1 for v in ir_v if 0 < v <= 0.5)
ir_n = sum(1 for v in ir_v if v <= 0)

html = f'''<!DOCTYPE html>
<html lang=zh-CN>
<head><meta charset=UTF-8><title>Alpha191 回测报告</title>
<style>
body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#f5f6fa;color:#2c3e50}}
h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}}
.sb{{display:flex;gap:15px;flex-wrap:wrap;margin:20px 0}}
.sc{{background:#fff;border-radius:8px;padding:15px 20px;box-shadow:0 2px 4px #0000001a;flex:1;min-width:140px;text-align:center}}
.sc .n{{font-size:28px;font-weight:700;color:#3498db}} .sc .l{{font-size:12px;color:#7f8c8d}}
.sc.g .n{{color:#27ae60}} .sc.r .n{{color:#e74c3c}} .sc.o .n{{color:#f39c12}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 4px #0000001a;margin:15px 0}}
th{{background:#3498db;color:#fff;padding:10px 12px;text-align:left;font-size:13px}}
td{{padding:6px 12px;border-bottom:1px solid #ecf0f1;font-size:13px}}
tr:hover{{background:#f8f9fa}}
.na{{color:#bdc3c7}} .skip{{color:#f39c12}} .err{{color:#e74c3c}}
.footer{{margin-top:40px;padding:15px;background:#ecf0f1;border-radius:8px;font-size:12px;color:#7f8c8d;text-align:center}}
.badges{{display:flex;gap:15px;flex-wrap:wrap;margin:15px 0}}
.badge{{background:#fff;border-left:4px solid #27ae60;padding:10px 15px;border-radius:4px;box-shadow:0 1px 3px #0000001a;flex:1;min-width:180px}}
.badge.neg{{border-color:#e74c3c}}
.badge .nm{{font-weight:700;font-size:16px}} .badge .mt{{font-size:12px;color:#7f8c8d}}
</style></head>
<body>
<h1>📊 Alpha191 批量回测报告</h1>
<p>{START_DATE} ~ {END_DATE} | 持仓{RETURN_PERIOD}天 | 股票{len(sel)}只 | 交易日{n_dates}天</p>
<div class=sb>
<div class="sc g"><div class=n>{len(ok_r)}</div><div class=l>成功</div></div>
<div class="sc o"><div class=n>{len(sk_r)}</div><div class=l>跳过</div></div>
<div class="sc r"><div class=n>{len(er_r)}</div><div class=l>失败</div></div>
<div class=sc><div class=n>{np.mean(ic_v):.4f}</div><div class=l>平均IC</div></div>
<div class=sc><div class=n>{sum(1 for v in ic_v if v>0)}/{len(ic_v)}</div><div class=l>正向IC</div></div>
<div class=sc><div class=n>{elapsed/60:.0f}min</div><div class=l>耗时</div></div>
</div>

<h2>🏆 Top5</h2>
<div class=badges>''' + ''.join(
    f'<div class=badge><div class=nm>a{r["alpha"]:03d}</div><div class=mt>IC={r["ic_mean"]:.4f} IR={r["ic_ir"]:.2f}</div></div>\n'
    for r in ok_r[:5]
) + '''</div>

<h2>📉 Bottom5</h2>
<div class=badges>''' + ''.join(
    f'<div class="badge neg"><div class=nm>a{r["alpha"]:03d}</div><div class=mt>IC={r["ic_mean"]:.4f} IR={r["ic_ir"]:.2f}</div></div>\n'
    for r in reversed(ok_r[-5:])
) + '''</div>

<h2>📈 IC分布</h2>
<table><tr><th>区间</th><th>数量</th><th>占比</th></tr>''' + dist + '''</table>

<h2>📊 IR分布</h2>
<table><tr><th>IR区间</th><th>数量</th></tr>
<tr><td>IR>1.0</td><td>''' + str(ir_p) + '''</td></tr>
<tr><td>0.5<IR≤1.0</td><td>''' + str(ir_m) + '''</td></tr>
<tr><td>0<IR≤0.5</td><td>''' + str(ir_w) + '''</td></tr>
<tr><td>IR≤0</td><td>''' + str(ir_n) + '''</td></tr>
</table>

<h2>📋 所有因子</h2>
<table><tr><th>#</th><th>IC</th><th>IC_IR</th><th>RankIC</th><th>RankIC_IR</th><th>IC>0%</th><th>耗时</th></tr>
''' + rows + '''</table>

<h2>⏭️ 跳过</h2><p>''' + (skip_h or '无') + '''</p>
<h2>❌ 失败</h2><p>''' + (err_h or '无') + '''</p>
<div class=footer>
<p>生成: ''' + datetime.now().strftime('%Y-%m-%d %H:%M') + ''' | Alpha191本地化 | IC直接计算</p>
</div></body></html>'''

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nHTML: {REPORT_FILE}")

print("发邮件...")
try:
    send_email(
        subject=f"Alpha191 回测报告 ({START_DATE}~{END_DATE})",
        body=html, body_type='html',
        receiver_emails=['2056123357@qq.com'],
    )
    print("✅ 邮件发送成功!")
except Exception as e:
    print(f"❌ 邮件失败: {e}")

print("全部完成!")
