"""공통 도구: 운영 코드(pipeline/)를 그대로 불러와 자료를 조립하고, 평가 지표를 계산한다. 운영 파일은 쓰지 않는다."""
import sys, os, time, json
import numpy as np, pandas as pd
REPO = os.environ.get('REPO', '/home/claude/naerincheon-ss-dashboard')
sys.path.insert(0, REPO + '/pipeline')
import model, forecast as fcm, run_daily as rd  # noqa
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

_df = None
def load():
    global _df
    if _df is None:
        _df = rd.assemble()
    return _df.copy()

def obs_mask(df): return df.SS.notna() & (df.SS > 0)

def metrics(y, p, thr=(10, 25)):
    y = np.asarray(y, float); p = np.asarray(p, float); k = np.isfinite(p) & np.isfinite(y); y, p = y[k], p[k]
    ly, lp = np.log(y), np.log(p)
    r = lambda a, b: 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)
    d = {'n': len(y), 'logR2': r(ly, lp), 'R2': r(y, p), 'MAE': np.mean(np.abs(y - p)), 'RMSE': np.sqrt(np.mean((y - p) ** 2)),
         'logRMSE': np.sqrt(np.mean((ly - lp) ** 2)), 'MedAE': np.median(np.abs(y - p)), 'MBE': np.mean(p - y),
         'logMBE': np.mean(lp - ly)}
    for t in thr:
        hi = y >= t; ph = p >= t
        d[f'n_ge{t}'] = int(hi.sum()); d[f'MAE_ge{t}'] = np.mean(np.abs(y[hi] - p[hi])) if hi.any() else np.nan
        d[f'ratio_ge{t}'] = np.median(p[hi] / y[hi]) if hi.any() else np.nan
        d[f'recall{t}'] = (ph & hi).sum() / hi.sum() if hi.any() else np.nan
        d[f'prec{t}'] = (ph & hi).sum() / ph.sum() if ph.any() else np.nan
    lo = y < 10; d['MAE_lt10'] = np.mean(np.abs(y[lo] - p[lo])); d['logR2_lt10'] = r(ly[lo], lp[lo])
    return d

def block_boot(y, pa, pb, idx, fn, B=2000, seed=0, block='M'):
    """월 블록 부트스트랩: fn(y,p) 차이 (b - a)의 95% 구간."""
    idx = pd.DatetimeIndex(idx); g = idx.to_period(block).astype(str); ug = np.unique(g)
    pos = {u: np.where(g == u)[0] for u in ug}; rng = np.random.default_rng(seed); out = []
    y, pa, pb = map(np.asarray, (y, pa, pb))
    for _ in range(B):
        s = np.concatenate([pos[u] for u in rng.choice(ug, len(ug))])
        out.append(fn(y[s], pb[s]) - fn(y[s], pa[s]))
    out = np.array(out); return float(fn(y, pb) - fn(y, pa)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

logr2 = lambda y, p: 1 - np.sum((np.log(y) - np.log(p)) ** 2) / np.sum((np.log(y) - np.log(y).mean()) ** 2)
mae = lambda y, p: np.mean(np.abs(y - p))

def wf_daily(F, df, mk, years=range(2019, 2027), corr='median', w=None):
    """연도별 확장 전진검증(그 해 이전 실측으로 학습)."""
    tr_all = obs_mask(df); out = pd.Series(np.nan, index=df.index)
    for Y in years:
        tr = tr_all & (df.index.year < Y); te = tr_all & (df.index.year == Y)
        if not te.any(): continue
        m, sm = model.fit_log(mk, F[tr], df.SS[tr].values) if corr == 'median' else (mk().fit(F[tr], np.log(df.SS[tr].values)), 0.0)
        out[te] = np.exp(m.predict(F[te]) + sm)
    return out[tr_all & (df.index.year >= min(years))]
