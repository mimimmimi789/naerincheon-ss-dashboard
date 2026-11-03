"""R9b: 5일 예측 80% 구간폭(전진검증 2019–26 로그 잔차 10·90%)과 성능표 행. 운영 static의 기존 h1·h3 산정 방식과 같음."""
from h import *
df = load(); F, qc, bad, qpk = model.features(df); o = obs_mask(df)
static = json.loads(open(REPO + '/data/base/forecast_static.json', encoding='utf-8').read())
fr = fcm.Forecaster(df.P0, qc, qpk, df.Havg, df.Hmax, df.SS, static, model.temp_feats(df))
obs = df.SS[o & (df.index >= '2019-01-01')]; T = obs.index; y = obs.values; h = 5; D = T - pd.Timedelta(days=h)
fz = (df.P0.rolling(h, min_periods=h).sum().reindex(T).values, df.P0.rolling(h, min_periods=h).max().reindex(T).values)
out = {}
for tag, nm, fut in (('obs', 'XGB+Ridge 혼합 (관측만)', None), ('rain', 'XGB(예보강우 단조)+Ridge 혼합 + 예보강우', fz)):
    lp = fr.predict(h, nm, D, fut).values; r = np.log(y) - lp; lo, hi = np.quantile(r, .1), np.quantile(r, .9)
    out[tag] = [round(float(lo), 3), round(float(hi), 3), round(float(((r >= lo) & (r <= hi)).mean()), 3)]
print(json.dumps(out)); json.dump(out, open(OUT + '/r9b_h5_bands.json', 'w'))
