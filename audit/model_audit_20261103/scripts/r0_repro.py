"""R0: 저장소 기록 성능 재현 (일별 전진검증 + 1·3·7일 hindcast)."""
from h import *
t0 = time.time(); df = load(); print('자료', df.index.min().date(), df.index.max().date(), '실측', int(obs_mask(df).sum()), time.time()-t0)
F, qc, bad, qpk = model.features(df)
p = wf_daily(F, df, model.Blend); y = df.SS[p.index]
ref = pd.read_csv(REPO + '/data/base/ss_wf_blend_temp_hh_20261103.csv', parse_dates=['date']).set_index('date').ss_wf_mgL
print('일별 WF', {k: round(v, 3) for k, v in metrics(y, p).items()}, time.time()-t0)
print('저장본과 최대차', float(np.max(np.abs(ref.reindex(p.index) - p))))
p.rename('wf_ops').to_csv(OUT + '/r0_daily_wf.csv')
