"""R1: 1·3·7일 예측 재현 + 과거 실제 발표 예보(기상청 23시 발표 보관본)로 재평가.
예보 보관본: capstone/ss_short_term_forecast_20261102/data/raw/ss_forecast_pairs.csv (2014–2024, lead 1·3·7, FP1=D+1, FP2=D+1..D+2)
              + forecast_variables_2025.csv (2025, 발표일 d). 2021-06 이전은 동네예보 R06(6시간 강수) 합."""
from h import *
RAW = '/mnt/project-files/capstone/ss_short_term_forecast_20261102/data/raw/'
df = load(); F, qc, bad, qpk = model.features(df)
static = json.loads(open(REPO + '/data/base/forecast_static.json', encoding='utf-8').read())
fr = fcm.Forecaster(df.P0, qc, qpk, df.Havg, df.Hmax, df.SS, static, model.temp_feats(df))
t0 = time.time(); hc = fr.hindcast(); print('hindcast', round(time.time() - t0))
hc.to_csv(OUT + '/r1_hindcast_ops.csv')
obs = df.SS[obs_mask(df)]; obs = obs[obs.index >= '2019-01-01']
rows = []
for h in fcm.H:
    for tag in ('obs', 'rain'):
        D = obs.index - pd.Timedelta(days=h); p = hc[f'h{h}_{tag}'].reindex(D).values
        k = np.isfinite(p); rows.append({'h': h, 'tag': tag, 'cond': 'repro_all', **metrics(obs.values[k], p[k])})
# 월별 기준값
for h in fcm.H:
    D = obs.index - pd.Timedelta(days=h); lp = fr.predict(h, '월별 기준값', D)
    rows.append({'h': h, 'tag': 'clim', 'cond': 'repro_all', **metrics(obs.values, np.exp(lp.values))})
# 실제 예보 보관본
pr = pd.read_csv(RAW + 'ss_forecast_pairs.csv', parse_dates=['date', 'd'])
pr['FP1x'] = pr.FP1.fillna(pr.R06_FP1); pr['FP2x'] = pr.FP2.fillna(pr.R06_FP2); pr['FPmax2x'] = pr.FPmax2.fillna(pr.R06_FPmax2)
f25 = pd.read_csv(RAW + 'forecast_variables_2025.csv', parse_dates=['d'])
ss = df.SS
add = [{'date': d + pd.Timedelta(days=1), 'd': d, 'lead_days': 1, 'FP1x': a, 'FP2x': b, 'FPmax2x': c}
       for d, a, b, c in zip(f25.d, f25.FP1, f25.FP2, f25.FPmax2) if obs_mask(df).get(d + pd.Timedelta(days=1), False)]
print('2025 보관본 중 다음날 실측 있는 발표일', len(add), '/', len(f25))
pr = pd.concat([pr, pd.DataFrame(add)], ignore_index=True)
pr = pr[(pr.date >= '2019-01-01') & pr.date.isin(obs.index)]
pairs = []
for h in fcm.H:
    q = pr[pr.lead_days == h].dropna(subset=['FP1x']).drop_duplicates('date')
    T = pd.DatetimeIndex(q.date); D = T - pd.Timedelta(days=h); y = ss[T].values
    nm_o, nm_r = fr.pick[h]
    act_sum = df.P0.rolling(h, min_periods=h).sum().reindex(T).values; act_max = df.P0.rolling(h, min_periods=h).max().reindex(T).values
    if h == 1: fs, fm, lab = q.FP1x.values, q.FP1x.values, '실제예보(완전, D+1)'
    else: fs, fm, lab = q.FP2x.values, q.FPmax2x.values, f'실제예보(불완전: {h}일 중 앞 2일만)'
    po = np.exp(fr.predict(h, nm_o, D).values); pp = np.exp(fr.predict(h, nm_r, D, (act_sum, act_max)).values)
    pf = np.exp(fr.predict(h, nm_r, D, (fs, fm)).values); pc = np.exp(fr.predict(h, '월별 기준값', D).values)
    # 앞 2일 실제 강우(불완전 예보와 같은 창)를 넣은 경우: 예보 오차와 창 부족 효과 분리
    a2s = df.P0.rolling(2, min_periods=2).sum().reindex(D + pd.Timedelta(days=2)).values
    a2m = df.P0.rolling(2, min_periods=2).max().reindex(D + pd.Timedelta(days=2)).values
    pa2 = np.exp(fr.predict(h, nm_r, D, (a2s, a2m)).values) if h > 1 else pp
    for nm, p in (('월별 기준값', pc), ('관측만', po), ('완벽예보(실제 내린 비)', pp), (lab, pf), ('실제 비 앞2일만', pa2)):
        if h == 1 and nm == '실제 비 앞2일만': continue
        k = np.isfinite(p); rows.append({'h': h, 'tag': nm, 'cond': 'same_days_archive', **metrics(y[k], p[k])})
    pairs.append(pd.DataFrame({'h': h, 'target': T, 'issue': D, 'ss': y, 'fc_sum': fs, 'fc_max': fm, 'act_sum': act_sum, 'act_max': act_max,
                               'p_clim': pc, 'p_obs': po, 'p_perfect': pp, 'p_realfc': pf, 'p_act2': pa2}))
    # 부트스트랩: 완벽 vs 실제예보 (로그R², MAE)
    for fn, fnn in ((logr2, 'logR2'), (mae, 'MAE')):
        b = block_boot(y, pp, pf, T, fn); rows.append({'h': h, 'tag': f'Δ{fnn}(실제-완벽)', 'cond': 'boot', 'n': len(y), 'd': b[0], 'lo': b[1], 'hi': b[2]})
        b = block_boot(y, po, pf, T, fn); rows.append({'h': h, 'tag': f'Δ{fnn}(실제예보-관측만)', 'cond': 'boot', 'n': len(y), 'd': b[0], 'lo': b[1], 'hi': b[2]})
P = pd.concat(pairs); P.to_csv(OUT + '/r1_realfc_pairs.csv', index=False)
# 예보 강우 자체의 정확도 (h=1, D+1 일강우)
q = P[P.h == 1]; e = q.fc_sum - q.act_sum
fa = {'n': len(q), 'bias_mm': e.mean(), 'MAE_mm': e.abs().mean(), 'corr': np.corrcoef(q.fc_sum, q.act_sum)[0, 1],
      'hit10': ((q.fc_sum >= 10) & (q.act_sum >= 10)).sum(), 'obs10': (q.act_sum >= 10).sum(), 'fc10': (q.fc_sum >= 10).sum(),
      'ratio_when_obs_ge20': (q.fc_sum[q.act_sum >= 20] / q.act_sum[q.act_sum >= 20]).median()}
print('예보강우 정확도(D+1)', {k: round(float(v), 3) for k, v in fa.items()})
R = pd.DataFrame(rows); R.to_csv(OUT + '/r1_scores.csv', index=False)
pd.set_option('display.width', 250); print(R[['h', 'tag', 'cond', 'n', 'logR2', 'R2', 'MAE', 'MBE', 'n_ge25', 'ratio_ge25', 'd', 'lo', 'hi']].round(3).to_string())
json.dump({k: float(v) for k, v in fa.items()}, open(OUT + '/r1_rain_fc_skill.json', 'w'))
