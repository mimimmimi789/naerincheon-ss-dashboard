"""R6: 수문·자료 점검 (모델 학습 없음). 면적비·유량보정 vs 독립 실측유량(총량측정망 Q_meas), 수위–유량식 연도별 안정성,
강우 결측·무강우 처리, SS 분포·고농도 사례, 운영 예보 완전성."""
from h import *
df = load(); F, qc, bad, qpk = model.features(df); QH, QPK = model.stage_flow(df)
o = obs_mask(df); res = {}
# 1. 면적비·유량: 총량측정망 채수일 실측유량 Q_meas(독립) vs 원대교×1.0414
k = o & df.Q_meas.notna() & (df.Q_meas > 0) & df.Q.notna()
qw = df.Q[k] / 1.0414; qm = df.Q_meas[k]
r = qm / qw; res['Qmeas_vs_wondae'] = {'n': int(k.sum()), 'median_ratio_Qmeas_over_wondae': float(r.median()),
    'IQR': [float(r.quantile(.25)), float(r.quantile(.75))], 'logcorr': float(np.corrcoef(np.log(qm), np.log(qw))[0, 1]),
    'median_ratio_highflow(Q>50)': float(r[qw > 50].median()) if (qw > 50).any() else None, 'n_highflow': int((qw > 50).sum())}
# 보정된 실측일: 원 유량 / 보정 유량 / Q_meas 비교
kb = k & bad
cmp = pd.DataFrame({'SS': df.SS[kb], 'Q_raw': df.Q[kb], 'Q_corr': qc[kb], 'Q_meas': df.Q_meas[kb], 'Havg': df.Havg[kb]})
cmp['err_raw'] = np.log(cmp.Q_raw / cmp.Q_meas); cmp['err_corr'] = np.log(cmp.Q_corr / cmp.Q_meas)
cmp.round(3).to_csv(OUT + '/r6_flowfix_vs_qmeas.csv'); res['flowfix_on_obs_days'] = {'n': len(cmp),
    'median_abs_logerr_raw': float(cmp.err_raw.abs().median()) if len(cmp) else None, 'median_abs_logerr_corr': float(cmp.err_corr.abs().median()) if len(cmp) else None}
res['n_corrected_days_total'] = int(bad.sum()); res['n_corrected_by_year'] = {int(y): int(v) for y, v in bad.groupby(df.index.year).sum().items()}
# 2. 사례
for d in ['2023-06-27', '2021-03-03', '2023-06-26', '2021-03-02']:
    res[f'case_{d}'] = {c: (None if pd.isna(v) else round(float(v), 3)) for c, v in
        {'SS': df.SS.get(d), 'P0': df.P0[d], 'Pprev': df.P0.shift(1)[d], 'P7prior': F.P7prior[d], 'Q_raw': df.Q[d], 'Q_stage': QH[d], 'Q_corr': qc[d],
         'Q_meas': df.Q_meas.get(d), 'Qpk_stage': QPK[d], 'Havg': df.Havg[d], 'Hmax': df.Hmax[d], 'Hmin': df.Hmin[d], 'Hh10': df.Hh.get(d),
         'Hh_pk24': df.Hh_pk24.get(d), 'TA': F.TA[d], 'TAmin': F.TAmin[d], 'fixed': float(bad[d])}.items()}
wf = pd.read_csv(OUT + '/r0_daily_wf.csv', parse_dates=['date']).set_index('date').wf_ops
for d in ['2023-06-27', '2021-03-03']: res[f'case_{d}']['wf_pred'] = round(float(wf[d]), 2)
# 3. 수위–유량식 연도별 안정성: 같은 수위에서 연도별 환산유량, 잔차 표준편차, 0.7 밖 비율
d0 = df.dropna(subset=['Q', 'Havg']); d0 = d0[d0.Q > 0]; rows = []
for y in sorted(set(d0.index.year)):
    x = d0[d0.index.year == y]
    if len(x) < 100: continue
    c = np.polyfit(x.Havg, np.log(x.Q), 2); rr = np.log(x.Q) - np.polyval(c, x.Havg)
    x2 = x[abs(rr) < 0.7]; c2 = np.polyfit(x2.Havg, np.log(x2.Q), 2); rr2 = np.log(x2.Q) - np.polyval(c2, x2.Havg)
    rows.append({'year': y, 'n': len(x), 'Hmin': x.Havg.min(), 'Hmax_fit': x.Havg.max(), 'out07_frac': float((abs(rr) >= 0.7).mean()),
                 'resid_sd': float(rr2.std()), **{f'Q_at_H{h}': float(np.exp(np.polyval(c2, h))) for h in (2.0, 2.5, 3.0, 4.0)}})
RC = pd.DataFrame(rows); RC.round(3).to_csv(OUT + '/r6_rating_by_year.csv', index=False)
# 잔차 기준 민감도: 0.5/0.7/1.0에서 보정일 수
for thr in (0.5, 0.7, 1.0):
    res[f'n_fixed_thr{thr}'] = int((((np.log(df.Q) - np.log(QH)).abs() > thr) | (df.Q.isna() & QH.notna())).sum())
# 4. 강우: NaN 처리
P = df.P0; res['rain_nan_days'] = [str(x.date()) for x in P[P.isna()].index]
res['rain_nan_note'] = 'build(): rolling(min_periods=1).sum은 NaN을 건너뜀(0처럼), wet=(P>=1)은 NaN을 무강우로, API7 합은 NaN 전파'
f = F.loc[P.isna()]; res['features_on_nan_days'] = {'P0_nan': int(f.P0.isna().sum()), 'P3prior_nan': int(f.P3prior.isna().sum()), 'API7_nan_next7days': int(F.API7[P.isna().shift(1, fill_value=False)].isna().sum())}
# 기린 결빙 의심(2023-01~03-22) 처리 여부: 원자료 비영(非零) 파일과 base P0 비교
a = pd.read_csv('/mnt/project-files/capstone/ss_improve_20261102/aws_rn_2014_2023_nonzero.csv'); a['date'] = pd.to_datetime(a.date.astype(str)); a = a.set_index('date')
w = pd.Series(src_w := {321: 0.0487, 557: 0.5113, 559: 0.4400}); fr = a.loc['2023-01-01':'2023-03-22']
res['kirin_freeze_2023'] = {'days_neighbors_rain': int((fr[['rn_321', 'rn_559']].sum(axis=1) > 0).sum()), 'kirin_nonzero': int((fr.rn_557 > 0).sum()),
    'base_P0_sum': float(df.P0['2023-01-01':'2023-03-22'].sum()), 'thiessen_with_557_as0_sum': float((fr.rn_321 * .0487 + fr.rn_559 * .44).sum()),
    'renorm_without_557_sum': float(((fr.rn_321 * .0487 + fr.rn_559 * .44) / (.0487 + .44)).sum())}
res['aws_nonzero_file_nan'] = a.isna().sum().to_dict()
# 5. SS 분포
s = df.SS[o]; res['ss_dist'] = {'n': len(s), **{f'q{q}': float(s.quantile(q / 100)) for q in (10, 25, 50, 75, 90, 95, 99)}, 'max': float(s.max()),
    'n_ge10': int((s >= 10).sum()), 'n_ge25': int((s >= 25).sum()), 'n_ge100': int((s >= 100).sum()), 'p90_2014_23': float(s[s.index < '2024'].quantile(.9))}
res['ss_ge25_by_year'] = {int(y): int(v) for y, v in (s >= 25).groupby(s.index.year).sum().items()}
res['ss_n_by_year'] = {int(y): int(v) for y, v in s.groupby(s.index.year).size().items()}
top = s.sort_values(ascending=False).head(15); t = pd.DataFrame({'SS': top, 'P0': df.P0[top.index], 'P2': F.P2[top.index], 'P7prior': F.P7prior[top.index],
    'Q_corr': qc[top.index], 'Q_meas': df.Q_meas[top.index], 'fixed': bad[top.index].astype(int), 'wf': wf.reindex(top.index), 'month': top.index.month})
t.round(2).to_csv(OUT + '/r6_top15_ss.csv')
gap = s.index.to_series().diff().dt.days; res['sampling_gap_days'] = {'median': float(gap.median()), 'p90': float(gap.quantile(.9))}
# 6. 운영 예보 완전성(현재 kma_fcst.json) + 코드 동작
j = json.load(open(REPO + '/data/live/kma_fcst.json', encoding='utf-8')); iss = pd.Timestamp(j['issue'])
res['kma_fcst_now'] = {'issue': j['issue'], 'dates': list(j['rain_mm'].keys()), 'max_lead_day': int((pd.Timestamp(max(j['rain_mm'])) - iss).days),
    'note': 'kma_fcst()는 fcstDate별 PCP를 시간 수와 무관하게 합산(부분일도 하루로 계산). live()는 있는 날만 합산해 h일 모델에 넣고 days/need만 기록.'}
json.dump(res, open(OUT + '/r6_hydro.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float)[:6000]); print(RC.round(3).to_string()); print(t.round(1).to_string()); print(cmp.round(2).to_string())
