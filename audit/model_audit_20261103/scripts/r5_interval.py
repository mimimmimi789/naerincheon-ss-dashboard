"""R5: 80% 예측구간 점검. 운영 구간(고정 로그 잔차 분위수, 2019–26 같은 표본에서 산정) vs 시간순 보정(연도 Y 구간은 2019..Y-1 전진검증 잔차로만).
조건별 포함률(농도, 강우, 선행기간). 평가 2020–26(시간순 보정은 이전 연도 잔차가 있어야 함)."""
from h import *
df = load(); o = obs_mask(df)
wf = pd.read_csv(OUT + '/r0_daily_wf.csv', parse_dates=['date']).set_index('date').wf_ops
hc = pd.read_csv(OUT + '/r1_hindcast_ops.csv', parse_dates=[0], index_col=0)
static = json.loads(open(REPO + '/data/base/forecast_static.json', encoding='utf-8').read())
rows = []
def cover(name, T, y, p, lo_fix, hi_fix, P):
    r = np.log(y) - np.log(p); s = pd.Series(r, index=T)
    lo_t = pd.Series(np.nan, index=T); hi_t = lo_t.copy()
    for Y in range(2020, 2027):
        prev = s[(s.index.year < Y)]; m = s.index.year == Y
        if len(prev) >= 30: lo_t[m], hi_t[m] = prev.quantile(.1), prev.quantile(.9)
    k = (T.year >= 2020) & np.isfinite(r)
    for tag, lo, hi in (('운영 고정구간', np.full(len(r), lo_fix), np.full(len(r), hi_fix)), ('시간순 보정', lo_t.values, hi_t.values)):
        inside = (r >= lo) & (r <= hi)
        conds = {'전체': k, 'SS<10': k & (y < 10), '10≤SS<25': k & (y >= 10) & (y < 25), 'SS≥25': k & (y >= 25),
                 '당일/기간 강우≥10mm': k & (P >= 10), '강우<10mm': k & (P < 10)}
        for c, m in conds.items():
            m = m & np.isfinite(lo)
            rows.append({'target': name, 'band': tag, 'cond': c, 'n': int(m.sum()), 'coverage': float(inside[m].mean()) if m.any() else np.nan,
                         'below': float((r[m] < lo[m]).mean()) if m.any() else np.nan, 'above': float((r[m] > hi[m]).mean()) if m.any() else np.nan,
                         'mean_width_ratio': float(np.mean(np.exp(hi[m] - lo[m]))) if m.any() else np.nan})
y = df.SS[wf.index].values; cover('일별 추정', wf.index, y, wf.values, model.LOG_LO, model.LOG_HI, df.P0[wf.index].values)
obs = df.SS[o & (df.index >= '2019-01-01')]
for h in fcm.H:
    for tag in ('obs', 'rain'):
        D = obs.index - pd.Timedelta(days=h); p = hc[f'h{h}_{tag}'].reindex(D).values; k = np.isfinite(p)
        Pw = df.P0.rolling(h + 1, min_periods=1).sum().reindex(obs.index).values   # D..D+h 강우
        lo, hi, _ = static['bands'][f'h{h}_{tag}']
        cover(f'h{h}_{tag}', obs.index[k], obs.values[k], p[k], lo, hi, Pw[k])
R = pd.DataFrame(rows); R.to_csv(OUT + '/r5_interval.csv', index=False); pd.set_option('display.width', 250)
print(R.round(3).to_string())
