"""R9: 4·5일 예측 성능 (운영 Forecaster와 같은 모델·연도별 전진검증 2019–26). 7일과 비교. 7일 표의 관측만/실제 내린 비 조건과 같음."""
from h import *
df = load(); F, qc, bad, qpk = model.features(df); o = obs_mask(df)
static = json.loads(open(REPO + '/data/base/forecast_static.json', encoding='utf-8').read())
fr = fcm.Forecaster(df.P0, qc, qpk, df.Havg, df.Hmax, df.SS, static, model.temp_feats(df))
obs = df.SS[o & (df.index >= '2019-01-01')]; T = obs.index; y = obs.values; rows = []; P = {}
for h in (1, 3, 4, 5, 7):
    D = T - pd.Timedelta(days=h)
    fz = (df.P0.rolling(h, min_periods=h).sum().reindex(T).values, df.P0.rolling(h, min_periods=h).max().reindex(T).values)
    for tag, nm, fut in (('월별 기준값', '월별 기준값', None), ('관측만', 'XGB+Ridge 혼합 (관측만)', None), ('+실제 내린 비', 'XGB(예보강우 단조)+Ridge 혼합 + 예보강우', fz)):
        p = np.exp(fr.predict(h, nm, D, fut).values); P[f'h{h}_{tag}'] = p
        rows.append({'h': h, 'tag': tag, **metrics(y, p)})
    print(h, 'done', flush=True)
for h in (4, 5, 7):
    for fn, fnn in ((logr2, 'logR2'), (mae, 'MAE')):
        d, lo, hi = block_boot(y, P[f'h{h}_월별 기준값'], P[f'h{h}_관측만'], T, fn, B=1000)
        rows.append({'h': h, 'tag': f'Δ{fnn} 관측만−기준값', 'd': d, 'lo': lo, 'hi': hi})
R = pd.DataFrame(rows); R.to_csv(OUT + '/r9_h5.csv', index=False); pd.set_option('display.width', 200)
print(R[['h', 'tag', 'n', 'logR2', 'R2', 'MAE', 'MBE', 'ratio_ge25', 'd', 'lo', 'hi']].round(3).to_string())
