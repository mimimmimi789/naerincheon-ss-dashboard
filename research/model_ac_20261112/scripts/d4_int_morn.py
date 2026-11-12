"""두 단계 1일 예측: (1) 오늘(D) 입력 40개 + 내일 비 → 내일(T) 유량·수위 변수 예측 (매일 자료로 학습)
(2) 예측한 T 입력을 운영 일별 추정 모델(+홍수일 보정 ¾)에 넣어 T의 SS.
평가: 기상청 보관 예보 257일(예보×0.63), 같은 날 실제 비(사후), 관측만(비 0) 295일. 연도별 전진검증.
transfer_20261105/scripts 평가틀(df.pkl, F.pkl, fw.flood 방식) 사용. big/scripts 에서 실행."""
from fw import *
import xgboost as xgb
from sklearn.linear_model import HuberRegressor
O = '/mnt/project-files/capstone/transfer_turb_20261110/flow_diag/'
F = F0.copy(); P = df.P0 if 'P0' in df else F.P0
RAIN = ['P0', 'Pprev', 'P3prior', 'P7prior', 'API7', 'ADD_prior', 'P2', 'Pmax3', 'P14prior', 'P30prior']
FIXED = RAIN + ['msin', 'mcos', 'TA', 'TA7', 'TAmin']
TGT = [c for c in F.columns if c not in FIXED]          # 내일 예측할 유량·수위·사상 변수
print('예측할 변수', len(TGT), TGT, flush=True)
Pd = F.P0
def rain_T(Tidx, pT):
    """목표일 강우 변수: 오늘까지 관측 + 내일 비 pT."""
    D = Tidx - pd.Timedelta(days=1); L = {0: pT}
    for k in range(1, 31): L[k] = Pd.reindex(Tidx - pd.Timedelta(days=k)).fillna(0).values
    f = pd.DataFrame(index=Tidx)
    f['P0'] = L[0]; f['Pprev'] = L[1]; f['P3prior'] = L[1] + L[2] + L[3]; f['P7prior'] = sum(L[k] for k in range(1, 8))
    f['API7'] = sum(L[k] * 0.9 ** (k - 1) for k in range(1, 8)); f['P2'] = L[0] + L[1]; f['Pmax3'] = np.maximum.reduce([L[0], L[1], L[2]])
    f['P14prior'] = sum(L[k] for k in range(1, 15)); f['P30prior'] = sum(L[k] for k in range(1, 31))
    addD = F.ADD_prior.reindex(D).fillna(0).values; pD = L[1]
    f['ADD_prior'] = np.where(pD < 1, addD + 1, 0)   # 내일 기준 무강우일수(오늘 비 기준)
    return f
EXTRA = []   # 실행 중 바꿈: 'int' 강도, 'morn' 아침 07시
def step1_X(Tidx, pT, imax=None, morn=None):
    X = F.reindex(Tidx - pd.Timedelta(days=1)).copy(); X.columns = [c + '_D' for c in X.columns]; X.index = Tidx
    X['pT'] = pT; X['lpT'] = np.log1p(pT); X['msin'] = np.sin(2 * np.pi * Tidx.month / 12); X['mcos'] = np.cos(2 * np.pi * Tidx.month / 12)
    if 'int' in EXTRA: X['Imax'] = IMAX.reindex(Tidx).values if imax is None else imax
    if 'morn' in EXTRA:
        M = MORN.reindex(Tidx)
        for c in M.columns: X[c] = M[c].values
    return X
import model as _M
HH = _M.read_hourly(['/home/user/naerincheon-ss-dashboard/data/auxdata/wondae_wl_hourly_2014_2026.csv'])
HH = HH[(HH > 0.5) & (HH < 10)]; HH = HH[~HH.index.duplicated(keep='last')].sort_index().asfreq('h').interpolate(limit=6, limit_area='inside')
RA = pd.read_csv('/mnt/project-files/capstone/ss_short_term_forecast_20261102/data/raw/inje_asos211_hourly_precip.csv', parse_dates=['datetime'])
RA = RA.set_index('datetime').precip_mm.fillna(0); RA = RA[~RA.index.duplicated()].asfreq('h').fillna(0)
days_all = F.index
IMAX = pd.Series(RA.resample('D', closed='right', label='left').max().reindex(days_all).values, index=days_all)   # 그날 01–24시 최대 시간강우
def _at(t): return HH.reindex(t).values
t0 = days_all; t7 = days_all + pd.Timedelta(hours=7)
H7 = _at(t7); H0 = _at(t0)
hmax = HH.rolling(8, min_periods=4).max().reindex(t7).values
r07 = RA.rolling(7, min_periods=1).sum().reindex(t7).values; rm07 = RA.rolling(7, min_periods=1).max().reindex(t7).values
MORN = pd.DataFrame({'H07': H7, 'dH07': H7 - H0, 'Hmx07': hmax - H7, 'R07': r07, 'Rmx07': rm07}, index=days_all)
allT = F.index[1:]; allT = allT[F.reindex(allT - pd.Timedelta(days=1)).notna().sum(1).values > 30]
_m = {}
def step1_models(Y):
    key = (Y, tuple(EXTRA))
    if key not in _m:
        XA = step1_X(allT, np.nan_to_num(Pd.reindex(allT).values))
        tr = allT[allT.year < Y]; ms = {}
        for c in TGT:
            yv = F[c].reindex(tr); ok = yv.notna().values
            ms[c] = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0).fit(XA.loc[tr][ok], yv[ok])
        _m[key] = ms; print('1단계 학습', Y, EXTRA, flush=True)
    return _m[key]
def predict_T(Tidx, pT, imax=None):
    """내일 입력 40개 (유량·수위는 예측, 강우는 오늘까지 관측+pT, 계절, 기온은 오늘 값)."""
    X1 = step1_X(Tidx, pT, imax); out = pd.DataFrame(index=Tidx, columns=F.columns, dtype=float)
    out[RAIN] = rain_T(Tidx, pT)[RAIN].values
    out['msin'] = X1.msin; out['mcos'] = X1.mcos
    for c in ['TA', 'TA7', 'TAmin']: out[c] = F[c].reindex(Tidx - pd.Timedelta(days=1)).values
    for Y in sorted(set(Tidx.year)):
        s = Tidx.year == Y; ms = step1_models(Y)
        for c in TGT: out.loc[s, c] = ms[c].predict(X1[s])
    return out[F.columns]
# 2단계: 운영 일별 모델 (그해 이전 실측) + 홍수일 보정 (그해 이전 OOF 잔차, 실제 입력으로 적합)
lo_oof = pd.read_pickle('oof_nearbig_운영.pkl'); r_oof = np.log(Y[lo_oof.index]) - lo_oof
wet_o = ((F.P2 >= 10) | (F.Hh_rng24 >= 0.3))[lo_oof.index].values; x_o = F.lQpk[lo_oof.index].fillna(2.441).values
_d = {}
def daily_pred(XT):
    out = pd.Series(np.nan, index=XT.index)
    for Yr in sorted(set(XT.index.year)):
        s = XT.index.year == Yr
        if Yr not in _d:
            tr = TR & (df.index.year < Yr); _d[Yr] = model.fit_log(model.Daily, F[tr], Y[tr].values)
        m, sm = _d[Yr]; lo = m.predict(XT[s]) + sm
        wet = ((XT.P2[s] >= 10) | (XT.Hh_rng24[s] >= 0.3)).values; x = XT.lQpk[s].fillna(2.441).values; cal = lo_oof.index.year < Yr; adj = lo.copy()
        for st in (False, True):
            c = cal & (wet_o == st); kk = wet == st
            if not kk.any(): continue
            base = np.median(r_oof[c]); sh = HuberRegressor().fit(x_o[c].reshape(-1, 1), r_oof[c].values).predict(x[kk].reshape(-1, 1)) if st else base
            adj[kk] = lo[kk] + base + 0.75 * (sh - base)
        out[s] = adj
    return out
# 평가 자료
RAW = '/mnt/project-files/capstone/ss_short_term_forecast_20261102/data/raw/'
pr = pd.read_csv(RAW + 'ss_forecast_pairs.csv', parse_dates=['date']); pr['FP1x'] = pr.FP1.fillna(pr.R06_FP1)
f25 = pd.read_csv(RAW + 'forecast_variables_2025.csv', parse_dates=['d'])
pr = pd.concat([pr, pd.DataFrame({'date': f25.d + pd.Timedelta(days=1), 'lead_days': 1, 'FP1x': f25.FP1})], ignore_index=True)
cmp = pd.read_csv('/mnt/project-files/capstone/transfer_turb_20261110/compare/results/fc1_kma.csv', index_col=0, parse_dates=True)
obs_c = pd.read_csv('/mnt/project-files/capstone/transfer_turb_20261110/compare/results/fc1_obs.csv', index_col=0, parse_dates=True)
q = pr[(pr.lead_days == 1) & pr.date.isin(cmp.index)].dropna(subset=['FP1x']).drop_duplicates('date').set_index('date')
Tq = pd.DatetimeIndex(cmp.index); y = cmp.SS.values; fk = q.FP1x.reindex(Tq).values * 0.63
# 예보 시간 강도: 전날 23시 발표 PCP, 목표일 01–24시, 3지점 티센 가중 평균의 최대
fp = pd.read_csv(RAW + 'fcst_points_merged.csv', usecols=['issue_time_kst', 'target_time_kst', 'element', 'station_id', 'raw_value'])
fp = fp[(fp.element == 'PCP') & fp.issue_time_kst.str.endswith('23:00')]
fp['v'] = pd.to_numeric(fp.raw_value, errors='coerce'); wts = {321: 0.0487, 557: 0.5113, 559: 0.4400}
fp['w'] = fp.station_id.map(wts); fp = fp.dropna(subset=['w'])
g = fp.groupby(['issue_time_kst', 'target_time_kst']).apply(lambda d: np.nansum(d.v * d.w) / d.w[d.v.notna()].sum() if d.v.notna().any() else np.nan).rename('v').reset_index()
g['tt'] = pd.to_datetime(g.target_time_kst); g['day'] = (g.tt - pd.Timedelta(hours=1)).dt.normalize()
g['iss'] = pd.to_datetime(g.issue_time_kst); g = g[g.day == g.iss.dt.normalize() + pd.Timedelta(days=1)]
cnt = g.groupby('day').v.count(); FIMAX = g.groupby('day').v.max()[cnt >= 20]
print('예보 시간강도 있는 날', len(FIMAX), flush=True)
rows = []
def sc(tag, n, yy, p):
    m = metrics(yy, p); rows.append(dict(평가=tag, 방식=n, n=m['n'], logR2=m['logR2'], R2=m['R2'], MAE=m['MAE'], 비10=m['ratio_ge10'],
        경보25=f"{int((p>=25).sum())}/{int(((p>=25)&(yy>=25)).sum())}/{int((yy>=25).sum())}", 경보10=f"{int((p>=10.2).sum())}/{int(((p>=10.2)&(yy>=10.2)).sum())}/{int((yy>=10.2).sum())}")); print(rows[-1], flush=True)
op = cmp['운영'].values; P = {}
def run(extra, Tidx, pT, imax=None):
    global EXTRA; EXTRA = extra; return np.exp(daily_pred(predict_T(Tidx, pT, imax)).values)
P['base'] = run([], Tq, fk); P['morn'] = run(['morn'], Tq, fk)
sel = Tq.isin(FIMAX.index); Ti = Tq[sel]
P['int'] = run(['int'], Ti, fk[sel], FIMAX.reindex(Ti).values * 0.63)
P['int_raw'] = run(['int'], Ti, fk[sel], FIMAX.reindex(Ti).values)
P['int_morn'] = run(['int', 'morn'], Ti, fk[sel], FIMAX.reindex(Ti).values * 0.63)
T1 = '1일 기상청 예보 257'
sc(T1, '운영', y, op); sc(T1, '두 단계', y, P['base']); sc(T1, '두 단계 + 아침 07시', y, P['morn']); sc(T1, '두 단계 + 아침 07시, 운영과 큰 값', y, np.maximum(op, P['morn']))
T2 = f'시간 강도 예보 있는 날 {sel.sum()} (2021-06~2024)'; ys = y[sel]
sc(T2, '운영', ys, op[sel]); sc(T2, '두 단계', ys, P['base'][sel]); sc(T2, '두 단계 + 강도(×0.63)', ys, P['int']); sc(T2, '두 단계 + 강도(그대로)', ys, P['int_raw'])
sc(T2, '두 단계 + 아침 07시', ys, P['morn'][sel]); sc(T2, '두 단계 + 강도 + 아침', ys, P['int_morn'])
To = pd.DatetimeIndex(obs_c.index); yo = obs_c.SS.values; opo = obs_c['운영'].values; z = np.zeros(len(To))
P['o_base'] = run([], To, z); P['o_morn'] = run(['morn'], To, z)
T3 = '1일 관측만 295'
sc(T3, '운영', yo, opo); sc(T3, '두 단계', yo, P['o_base']); sc(T3, '두 단계 + 아침 07시', yo, P['o_morn'])
pd.DataFrame(rows).to_csv(O + 'd4_scores.csv', index=False)
for k, v in P.items():
    idx = To if k.startswith('o_') else (Ti if k.startswith('int') else Tq); pd.Series(v, index=idx).to_csv(O + f'd4_{k}.csv')
print(pd.DataFrame(rows).round(3).to_string()); print('끝')
