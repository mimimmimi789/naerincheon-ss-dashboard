"""R8: 운영 일별 전진검증·1일 예측의 농도 구간별·연도별 성능 (추가 학습 없음)."""
from h import *
r = pd.read_read = None
d = pd.read_csv(OUT + '/r3_daily_preds.csv', index_col=0, parse_dates=True)
y = d.ss.values; p = d['운영: XGB10+Ridge 0.5'].values; pX = d.XGB10.values; pR = d.Ridge.values
q90 = np.quantile(y, .9); rows = []
for nm, m in (('전체', y > 0), ('SS<10', y < 10), ('SS≥10', y >= 10), ('SS≥25', y >= 25), (f'상위10%(≥{q90:.1f})', y >= q90)):
    for mn, pp in (('운영 혼합', p), ('XGB10', pX), ('Ridge', pR)):
        e = pp[m] - y[m]
        rows.append({'구간': nm, 'model': mn, 'n': int(m.sum()), 'MAE': np.mean(np.abs(e)), 'MBE': np.mean(e), 'median_ratio': np.median(pp[m] / y[m]),
                     'logMBE': np.mean(np.log(pp[m]) - np.log(y[m])), 'logRMSE': np.sqrt(np.mean((np.log(pp[m]) - np.log(y[m])) ** 2))})
B = pd.DataFrame(rows); B.to_csv(OUT + '/r8_bins.csv', index=False)
yr = [{'year': Y, **metrics(y[d.index.year == Y], p[d.index.year == Y])} for Y in range(2019, 2027)]
Yr = pd.DataFrame(yr); Yr.to_csv(OUT + '/r8_by_year.csv', index=False)
# 고농도 2건 제외 시 원척도 R² (영향도)
k = ~d.index.isin(pd.to_datetime(['2023-06-27', '2021-03-03']))
ex = metrics(y[k], p[k])
pd.set_option('display.width', 200); print(B.round(3).to_string()); print(Yr[['year', 'n', 'logR2', 'R2', 'MAE', 'MBE', 'n_ge25', 'ratio_ge25']].round(3).to_string())
print('2건 제외', {k2: round(v, 3) for k2, v in ex.items() if k2 in ('n', 'logR2', 'R2', 'MAE', 'RMSE')})
# 1일 예측 hindcast 구간별
hc = pd.read_csv(OUT + '/r1_hindcast_ops.csv', index_col=0, parse_dates=True); df = load(); o = obs_mask(df)
obs = df.SS[o & (df.index >= '2019-01-01')]
for h in (1, 3):
    for tag in ('obs', 'rain'):
        pp = hc[f'h{h}_{tag}'].reindex(obs.index - pd.Timedelta(days=h)).values; yy = obs.values; kk = np.isfinite(pp)
        print(h, tag, {n: round(float(np.median(pp[kk & mm] / yy[kk & mm])), 3) for n, mm in (('<10', yy < 10), ('≥10', yy >= 10), ('≥25', yy >= 25))})
