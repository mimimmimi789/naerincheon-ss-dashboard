"""R7: 공개 지연(as-of) 민감도. 기준일 t에 학습에 쓸 수 있는 SS = 채수일 ≤ t − L (공개 이력이 없어 L은 가정: 0, 60, 120일).
# 알려진 문제: 분기 창 끝을 q + QuarterBegin(1)로 계산해 3·6·9·12월이 빠짐 (일별 193/295건만 평가). 비교는 r7b_same_days.py로 같은 날짜에서 수행. 고치려면 q + pd.DateOffset(months=3).
분기마다 재학습(분기 시작일 t0 기준). 일별 추정과 1일 예측(관측만)·3일 예측(+실제강우, 완벽예보)에 적용. 운영 연도별 방식과 비교."""
from h import *
df = load(); F, qc, bad, qpk = model.features(df); o = obs_mask(df)
static = json.loads(open(REPO + '/data/base/forecast_static.json', encoding='utf-8').read())
fr = fcm.Forecaster(df.P0, qc, qpk, df.Havg, df.Hmax, df.SS, static, model.temp_feats(df))
Q0 = pd.date_range('2019-01-01', '2026-10-01', freq='QS'); rows = []; t0 = time.time(); allp = {}
def run_daily(L):
    out = pd.Series(np.nan, index=df.index)
    for q in Q0:
        te = o & (df.index >= q) & (df.index < q + pd.offsets.QuarterBegin(1)); 
        if not te.any(): continue
        tr = o & (df.index <= q - pd.Timedelta(days=L))
        m, sm = model.fit_log(model.Blend, F[tr], df.SS[tr].values); out[te] = np.exp(m.predict(F[te]) + sm)
    return out[o & (df.index >= '2019-01-01')]
def run_fc(h, nm, L, fut):
    obs = df.SS[o & (df.index >= '2019-01-01')]; T = obs.index; D = T - pd.Timedelta(days=h)
    X, y = fr.rows(h, fut); out = pd.Series(np.nan, index=T)
    fz = (df.P0.rolling(h, min_periods=h).sum().reindex(T).values, df.P0.rolling(h, min_periods=h).max().reindex(T).values) if fut else None
    Xd, _ = fr.issue_X(D, h, fz)
    for q in Q0:
        sel = (D >= q) & (D < q + pd.offsets.QuarterBegin(1))
        if not sel.any(): continue
        tr = X.index <= q - pd.Timedelta(days=L)     # 라벨(채수일) 기준 공개 가정
        c = list(X.columns); out[sel] = fr.fit_pred(fcm.MODELS[nm][0], X.loc[tr, c], y[tr], Xd.loc[sel, c], None)
    return obs, np.exp(out)
for L in (0, 60, 120):
    p = run_daily(L); y = df.SS[p.index]; allp[f'daily_L{L}'] = p
    rows.append({'task': 'daily', 'scheme': f'분기 재학습 L={L}', **metrics(y, p)}); print('daily', L, round(time.time() - t0))
for h, fut, nm in ((1, False, 'XGB+Ridge 혼합 (관측만)'), (3, True, 'XGB(예보강우 단조)+Ridge 혼합 + 예보강우')):
    for L in (0, 60, 120):
        obs, p = run_fc(h, nm, L, fut); allp[f'h{h}_L{L}'] = p
        rows.append({'task': f'h{h}' + ('+완벽예보' if fut else ' 관측만'), 'scheme': f'분기 재학습 L={L}', **metrics(obs.values, p.values)}); print(h, L, round(time.time() - t0))
pd.DataFrame(allp).to_csv(OUT + '/r7_asof_preds.csv')
R = pd.DataFrame(rows); R.to_csv(OUT + '/r7_scores.csv', index=False); pd.set_option('display.width', 250)
print(R[['task', 'scheme', 'n', 'logR2', 'R2', 'MAE', 'MBE', 'ratio_ge25']].round(3).to_string())
