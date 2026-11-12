"""A+C: 두 단계 + 아침 07시의 1단계 예측(XT)을 한 번만 만들고, 2단계를 운영 / C 미세조정 / 고농도만 섞기 w10 으로 바꿔 비교. big/scripts에서 실행."""
import sys, os, pickle
src4 = open('/mnt/project-files/capstone/transfer_turb_20261110/flow_diag/d4_int_morn.py', encoding='utf-8').read()
exec(src4.split('# 2단계: 운영 일별 모델')[0])
RAW = '/mnt/project-files/capstone/ss_short_term_forecast_20261102/data/raw/'
pr = pd.read_csv(RAW + 'ss_forecast_pairs.csv', parse_dates=['date']); pr['FP1x'] = pr.FP1.fillna(pr.R06_FP1)
f25 = pd.read_csv(RAW + 'forecast_variables_2025.csv', parse_dates=['d'])
pr = pd.concat([pr, pd.DataFrame({'date': f25.d + pd.Timedelta(days=1), 'lead_days': 1, 'FP1x': f25.FP1})], ignore_index=True)
cmp = pd.read_csv('/mnt/project-files/capstone/transfer_turb_20261110/compare/results/fc1_kma.csv', index_col=0, parse_dates=True)
obs_c = pd.read_csv('/mnt/project-files/capstone/transfer_turb_20261110/compare/results/fc1_obs.csv', index_col=0, parse_dates=True)
q = pr[(pr.lead_days == 1) & pr.date.isin(cmp.index)].dropna(subset=['FP1x']).drop_duplicates('date').set_index('date')
Tq = pd.DatetimeIndex(cmp.index); y = cmp.SS.values; fk = q.FP1x.reindex(Tq).values * 0.63
To = pd.DatetimeIndex(obs_c.index); yo = obs_c.SS.values
XP = O + 'd6_XT.pkl'
if os.path.exists(XP): XT = pickle.load(open(XP, 'rb'))
else:
    EXTRA = ['morn']
    TT = Tq.union(To); pT = pd.Series(0.0, index=TT); pT[Tq] = fk
    XT = {'kma': predict_T(Tq, fk), 'obs': predict_T(To, np.zeros(len(To)))}
    pickle.dump(XT, open(XP, 'wb'))
print('1단계 끝', flush=True)
ODIR = os.getcwd()
def step2(GRP, CAND):
    g = {'__name__': 'x'}; g.update({k: v for k, v in globals().items() if k in ('pd', 'np')})
    sys.argv = ['t2', GRP]; os.chdir(ODIR)
    ns = dict(globals()); exec(open('t2.py', encoding='utf-8').read().split('\ndef flood_lock')[0], ns)
    MK = model.Daily if CAND == '운영' else {'C 미세조정300': lambda: ns['FineTune'](), 'D 합쳐학습w10': lambda: ns['Pool'](w=10)}[CAND]
    lo_oof = pd.read_pickle(f'oof_{GRP}_{CAND}.pkl'.replace(' ', '')); r_oof = np.log(Y[lo_oof.index]) - lo_oof
    wet_o = ((F.P2 >= 10) | (F.Hh_rng24 >= 0.3))[lo_oof.index].values; x_o = F.lQpk[lo_oof.index].fillna(2.441).values
    ref = np.log(ns['qc'][obs_mask(df)].values); FX = ns['FX']
    def addrel(X):
        Z = X.copy(); Z['lQrel'], Z['Qpct'] = ns['rel_q'](np.log(np.clip(X.Q.values.astype(float), 1e-3, None)), ref); return Z
    fits = {}
    def dp(X):
        out = pd.Series(np.nan, index=X.index); XZ = addrel(X) if CAND != '운영' else X
        for Yr in sorted(set(X.index.year)):
            s = X.index.year == Yr
            if Yr not in fits:
                tr = TR & (df.index.year < Yr); fits[Yr] = model.fit_log(MK, (F if CAND == '운영' else FX)[tr], Y[tr].values)
            m, sm = fits[Yr]; lo = m.predict(XZ[s]) + sm
            wet = ((X.P2[s] >= 10) | (X.Hh_rng24[s] >= 0.3)).values; x = X.lQpk[s].fillna(2.441).values; cal = lo_oof.index.year < Yr; adj = lo.copy()
            for st in (False, True):
                c = cal & (wet_o == st); kk = wet == st
                if not kk.any(): continue
                base = np.median(r_oof[c]); sh = HuberRegressor().fit(x_o[c].reshape(-1, 1), r_oof[c].values).predict(x[kk].reshape(-1, 1)) if st else base
                adj[kk] = lo[kk] + base + 0.75 * (sh - base)
            out[s] = adj
        return np.exp(out.values)
    return dp(XT['kma']), dp(XT['obs'])
rows = []
def sc(tag, n, yy, p):
    m = metrics(yy, p); rows.append(dict(평가=tag, 방식=n, n=m['n'], logR2=m['logR2'], R2=m['R2'], MAE=m['MAE'], 비10=m['ratio_ge10'],
        경보25=f"{int((p>=25).sum())}/{int(((p>=25)&(yy>=25)).sum())}/{int((yy>=25).sum())}", 경보10=f"{int((p>=10.2).sum())}/{int(((p>=10.2)&(yy>=10.2)).sum())}/{int((yy>=10.2).sum())}")); print(rows[-1], flush=True)
sc('KMA 257', '운영 1일', y, cmp['운영'].values); sc('관측만 295', '운영 1일', yo, obs_c['운영'].values)
P = {}
for GRP, CAND, lab in [('nearbig', '운영', 'A (2단계 운영)'), ('nearbig', 'C 미세조정300', 'A+C 미세조정'), ('nearbighi', 'D 합쳐학습w10', 'A+고농도만 섞기')]:
    pk, po = step2(GRP, CAND); P[lab] = (pk, po)
    pd.Series(pk, index=Tq).to_csv(O + f'd6_{lab}_kma.csv'); pd.Series(po, index=To).to_csv(O + f'd6_{lab}_obs.csv')
    sc('KMA 257', lab, y, pk); sc('관측만 295', lab, yo, po)
for lab in ['A+C 미세조정', 'A+고농도만 섞기']:
    for nm, yy, i, ix in [('KMA', y, 0, Tq), ('관측만', yo, 1, To)]:
        print(lab, nm, 'vs A ΔlogR2', np.round(block_boot(yy, P['A (2단계 운영)'][i], P[lab][i], ix, logr2), 3), 'ΔMAE', np.round(block_boot(yy, P['A (2단계 운영)'][i], P[lab][i], ix, mae), 3), flush=True)
pd.DataFrame(rows).to_csv(O + 'd6_scores.csv', index=False); print(pd.DataFrame(rows).round(3).to_string()); print('끝')
