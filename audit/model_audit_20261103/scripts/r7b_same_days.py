"""R7b: r7 as-of 결과와 운영 연도별 방식을 같은 날짜에서 비교.
주의: r7_asof.py의 분기 창은 q + QuarterBegin(1)(3·6·9·12월 시작 기준)이라 1–2·4–5·7–8·10–11월만 덮었다(3·6·9·12월 제외).
그래서 r7 결과는 그 날짜들(일별 193, 1일 191, 3일 198)에서만 유효하며, 여기서 같은 날짜의 연도별(운영) 성적과 짝지어 비교한다."""
from h import *
df = load(); o = obs_mask(df); A = pd.read_csv(OUT + '/r7_asof_preds.csv', index_col=0, parse_dates=True)
wf = pd.read_csv(OUT + '/r0_daily_wf.csv', parse_dates=['date']).set_index('date').wf_ops
hc = pd.read_csv(OUT + '/r1_hindcast_ops.csv', index_col=0, parse_dates=True)
rows = []; bt = []
for task, key, ref in (('일별', 'daily', wf), ('1일 관측만', 'h1', hc.h1_obs.shift(1, freq='D')), ('3일+완벽예보', 'h3', hc.h3_rain.shift(3, freq='D'))):
    k = A[f'{key}_L0'].dropna().index; y = df.SS[k].values; r = ref.reindex(k).values
    rows.append({'task': task, 'scheme': '연도별(운영)', **metrics(y, r)})
    for L in (0, 60, 120):
        p = A[f'{key}_L{L}'].reindex(k).values; rows.append({'task': task, 'scheme': f'분기 as-of L={L}', **metrics(y, p)})
        for fn, fnn in ((logr2, 'logR2'), (mae, 'MAE')):
            d, lo, hi = block_boot(y, r, p, k, fn, B=1000); bt.append({'task': task, 'L': L, 'metric': fnn, 'diff_vs_yearly': d, 'lo95': lo, 'hi95': hi})
R = pd.DataFrame(rows); R.to_csv(OUT + '/r7b_same_days.csv', index=False); pd.DataFrame(bt).to_csv(OUT + '/r7b_boot.csv', index=False)
pd.set_option('display.width', 200)
print(R[['task', 'scheme', 'n', 'logR2', 'R2', 'MAE', 'MBE', 'ratio_ge25']].round(3).to_string()); print(pd.DataFrame(bt).round(3).to_string())
