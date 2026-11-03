"""R4: 일별 추정 변형 — 변수 묶음 축소(M0–M5), 유량보정 유무, 수위–유량식 as-of, 고농도 대안(원척도 손실·가중·2단계).
운영 혼합(Blend+중앙값 보정)과 같은 연도별 확장 전진검증(2019–26, n=295)."""
from h import *
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
df = load(); F, qc, bad, qpk = model.features(df)
tr_all = obs_mask(df); YRS = range(2019, 2027); idx = tr_all & (df.index.year >= 2019); yv = df.SS[idx].values; T = df.index[idx]
G = model.GROUPS
SEAS = ['msin', 'mcos']; RAIN = G['당일·전일 강우'] + G['선행강우 (3–30일)'] + G['무강우일수']; FLOW = G['유량']
STG = G['수위·첨두유량'] + model.HCOLS; TEMP = G['기온']
SETS = {'M0 계절': SEAS, 'M1 강우+계절': SEAS + RAIN, 'M2 +유량': SEAS + RAIN + FLOW, 'M3 +수위': SEAS + RAIN + FLOW + STG,
        'M4 +기온 (=운영 37)': SEAS + RAIN + FLOW + STG + TEMP,
        'M5 축소 13': ['P0', 'P2', 'API7', 'P30prior', 'ADD_prior', 'lQ', 'dlQ', 'Q_rel30', 'lQpk', 'Hh_pk24', 'TA7', 'msin', 'mcos']}
rows = []; preds = {}
def rec(nm, grp, p, extra=None):
    preds[nm] = p; rows.append({'group': grp, 'model': nm, 'n_feat': extra, **metrics(yv, p)})
t0 = time.time()
for nm, cols in SETS.items():
    rec(nm, 'ablation', wf_daily(F[cols], df, model.Blend).values, len(cols)); print(nm, round(time.time() - t0))
# 유량보정 없음: 원 유량 그대로(첨두유량 lQpk는 유지) / 수위 정보 전체 제거
dfx = df.copy(); Fn = model.build(dfx); Fn['lQpk'] = F.lQpk; Fn['Hrise'] = F.Hrise; Fn['dHmax'] = F.dHmax
for c in ['TA', 'TA7', 'TAmin'] + model.HCOLS: Fn[c] = F[c]
rec('유량보정 없음(원 유량)', 'flow', wf_daily(Fn[model.COLS], df, model.Blend).values, 37); print('nocorr', round(time.time() - t0))
print('보정된 날 수', int(bad.sum()), '실측일 중', int(bad[tr_all].sum()))
# 수위–유량식 as-of: 시험연도 Y 행은 Y-2..Y-1 자료로 적합한 식으로 보정(그 해 미래 유량 미사용). 학습행은 그대로.
def stage_asof(Y):
    d = df.dropna(subset=['Q', 'Havg']); d = d[(d.Q > 0) & (d.index.year >= Y - 2) & (d.index.year <= Y - 1)]
    c = np.polyfit(d.Havg, np.log(d.Q), 2); r = np.log(d.Q) - np.polyval(c, d.Havg); d = d[abs(r) < 0.7]
    c = np.polyfit(d.Havg, np.log(d.Q), 2); cap = d.Havg.max() + 0.5; m = df.index.year == Y
    QH = pd.Series(np.nan, index=df.index); QPK = QH.copy()
    QH[m] = np.exp(np.polyval(c, df.Havg[m].clip(upper=cap))); QPK[m] = np.exp(np.polyval(c, df.Hmax[m].clip(upper=cap)))
    return QH, QPK
QH0, QPK0 = model.stage_flow(df); out = pd.Series(np.nan, index=df.index); nchg = 0
for Y in YRS:
    QH, QPK = stage_asof(Y); m = df.index.year == Y
    QHm = QH0.copy(); QHm[m] = QH[m]; QPKm = QPK0.copy(); QPKm[m] = QPK[m]
    badm = ((np.log(df.Q) - np.log(QHm)).abs() > 0.7) | (df.Q.isna() & QHm.notna())
    dq = df.copy(); dq['Q'] = df.Q.where(~badm, QHm); Fa = model.build(dq); Fa['lQpk'] = np.log(QPKm.where(QPKm.notna(), dq.Q))
    for c in ['Hrise', 'dHmax', 'TA', 'TA7', 'TAmin'] + model.HCOLS: Fa[c] = F[c]
    Fa = Fa[model.COLS]; nchg += int((badm[m & tr_all] != bad[m & tr_all]).sum())
    tr = tr_all & (df.index.year < Y); te = tr_all & m
    mm, sm = model.fit_log(model.Blend, Fa[tr], df.SS[tr].values); out[te] = np.exp(mm.predict(Fa[te]) + sm)
rec('수위–유량식 as-of(전 2년)', 'leak', out[idx].values, 37); print('asof', round(time.time() - t0), '보정여부 바뀐 실측일', nchg)
# 고농도 대안 (XGB10 부분만 바꾸고 Ridge와 log 평균, 중앙값 보정 유지)
class WBag(model.BagXGB):
    """가중 학습: w = 1 + (log y 의 학습분포 상위 정도). 고농도 표본 가중."""
    def __init__(self, kind='w'): super().__init__(); self.kind = kind
    def fit(self, X, y):
        self.cols = list(X.columns)
        if self.kind == 'w':
            w = 1 + 2 * (y >= np.log(10)) + 3 * (y >= np.log(25)); self.ms = [model.XG(i).fit(X, y, sample_weight=w) for i in range(self.n)]
        else:   # 원척도 Tweedie(로그 링크): 원척도 평균을 맞춤 → log로 변환해 반환
            self.ms = [model.XG(i, objective='reg:tweedie', tweedie_variance_power=1.5).fit(X, np.exp(y)) for i in range(self.n)]
        return self
    def predict(self, X):
        p = np.mean([m.predict(X[self.cols]) for m in self.ms], axis=0)
        return p if self.kind == 'w' else np.log(np.maximum(p, 1e-3))
class BlendV(model.Blend):
    def __init__(self, kind): self.kind = kind
    def fit(self, X, y): self.a = WBag(self.kind).fit(X, y); self.b = model.SafeRidge().fit(X, y); return self
rec('고농도 가중(≥10 ×3, ≥25 ×6)+Ridge', 'highconc', wf_daily(F, df, lambda: BlendV('w')).values, 37); print('w', round(time.time() - t0))
rec('원척도 Tweedie XGB+Ridge(로그 평균)', 'highconc', wf_daily(F, df, lambda: BlendV('tw')).values, 37); print('tw', round(time.time() - t0))
# 2단계: P(SS≥10) 로지스틱 × [≥10 조건부 Ridge] + (1-P) × [운영 혼합] (log 공간 기대값). 중앙값 보정은 운영 부분만.
out = pd.Series(np.nan, index=df.index)
for Y in YRS:
    tr = tr_all & (df.index.year < Y); te = tr_all & (df.index.year == Y); yl = np.log(df.SS[tr].values)
    base, sm = model.fit_log(model.Blend, F[tr], df.SS[tr].values)
    clf = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(F[tr], yl >= np.log(10))
    hi = yl >= np.log(10); rh = model.SafeRidge().fit(F[tr][hi], yl[hi])
    pz = clf.predict_proba(F[te])[:, 1]; out[te] = np.exp(pz * rh.predict(F[te]) + (1 - pz) * (base.predict(F[te]) + sm))
    print(Y, '고농도 학습표본', int(hi.sum()))
rec('2단계(분류 P≥10 × 고농도 Ridge)', 'highconc', out[idx].values, 37)
pd.DataFrame(preds, index=T).assign(ss=yv).to_csv(OUT + '/r4_daily_preds.csv')
base = pd.read_csv(OUT + '/r0_daily_wf.csv', parse_dates=['date']).set_index('date').wf_ops.reindex(T).values
rows.append({'group': 'ref', 'model': '운영 (R0)', **metrics(yv, base)})
bt = []
for nm in preds:
    for fn, fnn in ((logr2, 'logR2'), (mae, 'MAE')):
        d, lo, hi = block_boot(yv, base, preds[nm], T, fn, B=1000); bt.append({'model': nm, 'metric': fnn, 'diff_vs_ops': d, 'lo95': lo, 'hi95': hi})
pd.DataFrame(bt).to_csv(OUT + '/r4_boot.csv', index=False)
R = pd.DataFrame(rows); R.to_csv(OUT + '/r4_scores.csv', index=False); pd.set_option('display.width', 250)
print(R[['group', 'model', 'n_feat', 'n', 'logR2', 'R2', 'MAE', 'RMSE', 'MBE', 'MAE_ge25', 'ratio_ge25', 'recall10', 'prec10', 'recall25', 'MAE_lt10', 'logR2_lt10']].round(3).to_string())
print(pd.DataFrame(bt).round(3).to_string()); print('총', round(time.time() - t0))
