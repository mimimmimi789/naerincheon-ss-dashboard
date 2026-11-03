"""R3: 일별 추정 — 앙상블 구성, 혼합비, 보정방식, 고농도 대안, 변수 축소(M0–M5), 유량보정 유무.
모두 같은 연도별 확장 전진검증(2019–26, 그 해 이전 실측만 학습, n=295)으로 비교. 혼합비 선택은 중첩(이전 연도 OOF만 사용)."""
from h import *
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
df = load(); F, qc, bad, qpk = model.features(df)
tr_all = obs_mask(df); YRS = range(2019, 2027)
y_all = np.log(df.SS)

def oof_inner(mk, X, y):
    """운영 fit_log와 같은 3겹 shuffle 내부 OOF 예측(보정값 계산용)."""
    o = np.zeros(len(y))
    for a, b in KFold(3, shuffle=True, random_state=0).split(X):
        o[b] = mk().fit(X.iloc[a], y[a]).predict(X.iloc[b])
    return o

def wf_comp(mk, X, name, sw=None):
    """반환: 시험일 log 예측(보정 전), 그 해 학습자료 내부 OOF(보정용)."""
    pred = pd.Series(np.nan, index=df.index); offs = {}
    for Y in YRS:
        tr = tr_all & (df.index.year < Y); te = tr_all & (df.index.year == Y)
        Xt, yt = X[tr], y_all[tr].values
        m = mk().fit(Xt, yt); pred[te] = m.predict(X[te]); offs[Y] = (oof_inner(mk, Xt, yt), yt)
    return pred, offs

class Seed1:
    def __init__(self): pass
    def fit(self, X, y): self.m = model.XG(0).fit(X, y); return self
    def predict(self, X): return self.m.predict(X)

def mkRF(): return make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(500, min_samples_leaf=3, max_features=0.5, n_jobs=4, random_state=0))
def mkET(): return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(500, min_samples_leaf=3, max_features=0.5, n_jobs=4, random_state=0))
def mkSVR(): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), SVR(C=3, gamma='scale', epsilon=0.1))

t0 = time.time(); X = F
comps = {}
for nm, mk in (('XGB10', model.BagXGB), ('XGB1', Seed1), ('Ridge', model.SafeRidge), ('RF', mkRF), ('ET', mkET), ('SVR', mkSVR)):
    comps[nm] = wf_comp(mk, X, nm); print(nm, round(time.time() - t0))
idx = tr_all & (df.index.year >= 2019); yv = df.SS[idx].values; T = df.index[idx]

def combo(ws, corr='median'):
    """ws: {이름: 가중}. 보정: median(운영) / duan / none — 그 해 학습자료 내부 OOF로 계산(시험자료 미사용)."""
    lp = sum(w * comps[n][0] for n, w in ws.items())
    out = pd.Series(np.nan, index=df.index)
    for Y in YRS:
        te = tr_all & (df.index.year == Y)
        o = sum(w * comps[n][1][Y][0] for n, w in ws.items()); yt = comps['XGB10'][1][Y][1]; r = yt - o
        off = {'median': np.median(r), 'duan': np.log(np.mean(np.exp(r))), 'none': 0.0}[corr]
        out[te] = lp[te] + off
    return np.exp(out[idx].values)

rows = []; preds = {}
def rec(nm, p, grp):
    preds[nm] = p; rows.append({'group': grp, 'model': nm, **metrics(yv, p)})
for n in comps: rec(n, combo({n: 1}), '단일')
rec('운영: XGB10+Ridge 0.5', combo({'XGB10': .5, 'Ridge': .5}), '앙상블')
rec('XGB1+Ridge 0.5', combo({'XGB1': .5, 'Ridge': .5}), '앙상블')
for w in (0.25, 0.75): rec(f'XGB10+Ridge w={w}', combo({'XGB10': w, 'Ridge': 1 - w}), '혼합비')
rec('XGB10+Ridge+SVR 1/3', combo({'XGB10': 1/3, 'Ridge': 1/3, 'SVR': 1/3}), '앙상블')
rec('RF+Ridge 0.5', combo({'RF': .5, 'Ridge': .5}), '앙상블')
rec('ET+Ridge 0.5', combo({'ET': .5, 'Ridge': .5}), '앙상블')
rec('운영 + Duan 평균보정', combo({'XGB10': .5, 'Ridge': .5}, 'duan'), '보정')
rec('운영 + 보정 없음', combo({'XGB10': .5, 'Ridge': .5}, 'none'), '보정')
# 중첩 혼합비: 연도 Y의 비율 = Y 이전 시험연도(2019..Y-1) OOF 예측에서 로그 MSE 최소 w (2019는 0.5)
lpX, lpR = comps['XGB10'][0], comps['Ridge'][0]; nest = pd.Series(np.nan, index=df.index); wsel = {}
for Y in YRS:
    prev = tr_all & (df.index.year >= 2019) & (df.index.year < Y)
    if prev.sum() < 30: w = .5
    else:
        ws = np.linspace(0, 1, 21); w = ws[np.argmin([np.mean((y_all[prev] - (a * lpX[prev] + (1 - a) * lpR[prev])) ** 2) for a in ws])]
    wsel[Y] = float(w); te = tr_all & (df.index.year == Y)
    o = w * comps['XGB10'][1][Y][0] + (1 - w) * comps['Ridge'][1][Y][0]; off = np.median(comps['XGB10'][1][Y][1] - o)
    nest[te] = w * lpX[te] + (1 - w) * lpR[te] + off
rec('XGB10+Ridge 중첩선택 w', np.exp(nest[idx].values), '혼합비'); print('중첩 w', wsel)
pd.DataFrame(preds, index=T).assign(ss=yv).to_csv(OUT + '/r3_daily_preds.csv')
# 상보성: 잔차 상관, 고농도에서 누가 나은가
rX = np.log(preds['XGB10']) - np.log(yv); rR = np.log(preds['Ridge']) - np.log(yv)
comp = {'corr_resid_XGB_Ridge': float(np.corrcoef(rX, rR)[0, 1]),
        'ge25_absres_XGB': float(np.mean(np.abs(rX[yv >= 25]))), 'ge25_absres_Ridge': float(np.mean(np.abs(rR[yv >= 25]))),
        'lt10_absres_XGB': float(np.mean(np.abs(rX[yv < 10]))), 'lt10_absres_Ridge': float(np.mean(np.abs(rR[yv < 10])))}
print(comp); json.dump(comp, open(OUT + '/r3_complementarity.json', 'w'))
# 부트스트랩(월 블록): 운영 vs 대안
base = preds['운영: XGB10+Ridge 0.5']; bt = []
for nm in ('XGB10', 'XGB1', 'Ridge', 'XGB1+Ridge 0.5', 'XGB10+Ridge+SVR 1/3', 'XGB10+Ridge 중첩선택 w', '운영 + Duan 평균보정', 'RF+Ridge 0.5'):
    for fn, fnn in ((logr2, 'logR2'), (mae, 'MAE')):
        d, lo, hi = block_boot(yv, base, preds[nm], T, fn, B=1000); bt.append({'model': nm, 'metric': fnn, 'diff_vs_ops': d, 'lo95': lo, 'hi95': hi})
pd.DataFrame(bt).to_csv(OUT + '/r3_boot.csv', index=False)
R = pd.DataFrame(rows); R.to_csv(OUT + '/r3_scores.csv', index=False)
pd.set_option('display.width', 250)
print(R[['group', 'model', 'n', 'logR2', 'R2', 'MAE', 'RMSE', 'MedAE', 'MBE', 'MAE_ge25', 'ratio_ge25', 'recall25', 'prec25', 'MAE_lt10']].round(3).to_string())
print(pd.DataFrame(bt).round(3).to_string()); print('총', round(time.time() - t0))
