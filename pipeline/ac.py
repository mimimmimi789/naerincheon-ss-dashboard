"""모델 C(전이 미세조정, 일별 추정)와 A+C(두 단계 + 아침 07시, 1일 예측). 2026-11-12 사용자 결정으로 운영 반영.

C: 가까운 6곳(인북A·소양A2·홍천A~A4) 실측 SS + 원주·경안천 탁도 보간 SS = 원천 7,312행(data/base/transfer_src_nearbig.csv,
   공통 18변수, 목표 yrel = log SS − 지점 중앙값)으로 XGB 300그루 × 5시드를 사전학습 → 내린천2 42변수(운영 40 + 상대유량 lQrel·Qpct)로
   300그루를 이어 붙여 미세조정 → 운영처럼 SafeRidge(내린천2 40변수)와 log 반반, 중앙값 보정, 홍수일 보정 ¾(C 전용 상수).
   전진검증 2019–26(n=295) 로그R² 0.780 / R² 0.770 / MAE 2.53 (운영 0.770 / 0.663 / 2.79), 잠금 2024–26(n=106) 0.795 / 0.860 / 1.55.
A: 1단계 = 기준일 D의 입력 40개 + 오늘(T=D+1) 비 + 계절 + 오늘 아침 07시 정보(원대교 수위 H07·dH07·Hmx07, 인제 ASOS 01–07시 강우
   R07·Rmx07) → 오늘의 유량·수위·사상 변수 25개를 XGB로 예측(모든 날 자료로 학습). 2단계 = 예측한 오늘 입력을 C에 넣는다.
   기상청 보관 예보 257일(예보×0.63) 로그R² 0.794 / R² 0.823 / MAE 2.36 (운영 0.762 / 0.355 / 2.86), 25 mg/L 경보 6/10.
연구 코드: research/model_ac_20261112 (t2.py FineTune, d4_int_morn.py, d6_ac_morn.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import model

ROOT = Path(__file__).resolve().parent.parent
SRC_CSV = ROOT / 'data' / 'base' / 'transfer_src_nearbig.csv'
RCOLS = ['P0', 'Pprev', 'P3prior', 'P7prior', 'API7', 'ADD_prior', 'P2', 'Pmax3', 'P14prior', 'P30prior', 'msin', 'mcos']
SHARED = RCOLS + ['Pev', 'TA', 'TA7', 'TAmin', 'lQrel', 'Qpct']
UCOLS = list(model.COLS) + ['lQrel', 'Qpct']
MONO2 = model.MONO + ('lQrel', 'Qpct')
NSEED = 5

# C 전진검증(2016–26) OOF 잔차로 다시 적합한 홍수일 보정 상수(research t2 nearbig C미세조정300). 형식은 model.FLOOD와 같음
FLOOD_C = dict(a=-0.4534, b=0.1162, wet_med=0.0098, dry_med=-0.1267, fill=2.441, k=0.75)
FLOOD_C_2023 = dict(a=-0.4593, b=0.1310, wet_med=0.0118, dry_med=-0.1274, fill=2.441, k=0.75)   # 잠금 평가용(2023년까지 잔차)
# C 80% 범위: 홍수일 보정 후 전진검증 로그 잔차 10–90% (2019–26, 평상시 228·비 온 날 67)
BAND_C = {'dry': (-0.5924, 0.6228), 'wet': (-0.7465, 0.8005)}
# A+C 1일 예측 80% 범위: 기상청 보관 예보 257일 / 관측만(비 0) 295일 로그 잔차 10–90%
BAND_AC = {'kma': (-0.652, 0.605), 'obs': (-0.628, 0.656)}

_src = None


def source():
    global _src
    if _src is None: _src = pd.read_csv(SRC_CSV, parse_dates=['date']).set_index('date')
    return _src


def rel_q(lq, ref):
    """지점 유량 분포(ref: 실측일 log Q) 기준 상대값(중앙값 차)·백분위."""
    ref = np.asarray(ref, float); ref = ref[np.isfinite(ref)]; srt = np.sort(ref)
    return lq - np.median(ref), np.searchsorted(srt, lq) / len(srt)


class RelQ:
    """내린천2 상대유량. 기준 = 실측일 보정유량 분포(학습할 때 정해 예측에도 같은 기준)."""
    def __init__(self, qc, obs_idx): self.ref = np.log(qc.reindex(obs_idx).values.astype(float))

    def add(self, F):
        X = F.copy(); lq = np.log(np.clip(F.Q.values.astype(float), 1e-3, None))
        X['lQrel'], X['Qpct'] = rel_q(lq, self.ref)
        return X


def pre_booster(cut, seed):
    s = source(); s = s[s.index < cut]
    Z = pd.DataFrame(np.nan, index=range(len(s)), columns=UCOLS)
    for c in SHARED: Z[c] = s[c].values
    mono = tuple(int(c in MONO2) for c in UCOLS)
    p = dict(max_depth=3, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, seed=seed,
             monotone_constraints=str(mono), nthread=4)
    return xgb.train(p, xgb.DMatrix(Z, label=s.yrel.values), 300), p


class FineTune:
    """C: 원천 사전학습 XGB에 내린천2 자료로 300그루를 이어 붙이고, SafeRidge(내린천2 40변수)와 log 반반.
    X는 UCOLS(운영 40 + lQrel·Qpct). 원천은 학습 마지막 날이 속한 해 이전(다음 해 1월 1일 전) 행만 쓴다."""
    def __init__(self, nft=300): self.nft = nft

    def fit(self, X, y):
        cut = pd.Timestamp(year=X.index.max().year + 1, month=1, day=1)
        self.med = float(np.median(y)); d = xgb.DMatrix(X[UCOLS], label=y - self.med); self.bs = []
        for sd in range(NSEED):
            b0, p = pre_booster(cut, sd)
            self.bs.append(xgb.train(p, d, self.nft, xgb_model=b0))
        self.r = model.SafeRidge().fit(X[list(model.COLS)], y); return self

    def predict(self, X):
        a = np.mean([b.predict(xgb.DMatrix(X[UCOLS])) for b in self.bs], axis=0) + self.med
        return (a + self.r.predict(X[list(model.COLS)])) / 2

    def contribs(self, X):
        """log 기여(열 = UCOLS + 기준값). XGB 평균 SHAP과 Ridge 선형 기여(lQrel·Qpct 칸은 0)의 반반."""
        cx = np.mean([b.predict(xgb.DMatrix(X[UCOLS]), pred_contribs=True) for b in self.bs], axis=0)
        cx[:, -1] += self.med
        cr = self.r.contribs(X[list(model.COLS)]); k = len(model.COLS)
        cr2 = np.zeros_like(cx); cr2[:, :k] = cr[:, :k]; cr2[:, -1] = cr[:, -1]
        return (cx + cr2) / 2


def band_c(F):
    w = model.wet_day(F)
    return np.where(w, BAND_C['wet'][0], BAND_C['dry'][0]), np.where(w, BAND_C['wet'][1], BAND_C['dry'][1])


def fit_c(F, qc, SS, upto=None):
    """C를 upto까지(없으면 전체) 실측으로 학습. 반환 (모델, 중앙값 보정, RelQ)."""
    tr = SS.notna() & (SS > 0)
    if upto is not None: tr &= SS.index <= upto
    rq = RelQ(qc, SS.index[SS.notna() & (SS > 0)])
    X = rq.add(F)
    m, sm = model.fit_log(FineTune, X[tr], SS[tr].values)
    return m, sm, rq


# ---------------- A: 두 단계 + 아침 07시 ----------------
RAIN = ['P0', 'Pprev', 'P3prior', 'P7prior', 'API7', 'ADD_prior', 'P2', 'Pmax3', 'P14prior', 'P30prior']
FIXED = RAIN + ['msin', 'mcos', 'TA', 'TA7', 'TAmin']
TGT = [c for c in model.COLS if c not in FIXED]          # 오늘 예측할 유량·수위·사상 변수 25개
MORN = ['H07', 'dH07', 'Hmx07', 'R07', 'Rmx07']


def morning(hh, rain_h, days):
    """날마다 아침 07시 정보. hh: 원대교 시간 수위(시각 색인), rain_h: 인제 ASOS 시간강우(시각 색인, 그 시각까지 1시간).
    H07 07시 수위, dH07 = H07 − 00시 수위, Hmx07 = 00–07시 최고 − H07, R07·Rmx07 = 01–07시 강우 합·최대."""
    h = hh[(hh > 0.5) & (hh < 10)]
    h = h[~h.index.duplicated(keep='last')].sort_index().asfreq('h').interpolate(limit=6, limit_area='inside')
    t0 = days; t7 = days + pd.Timedelta(hours=7)
    H7 = h.reindex(t7).values; H0 = h.reindex(t0).values
    hmax = h.rolling(8, min_periods=4).max().reindex(t7).values
    r = rain_h[~rain_h.index.duplicated(keep='last')].sort_index().asfreq('h')   # 빈 시각 = 자료 없음(NaN)
    r07 = r.rolling(7, min_periods=7).sum().reindex(t7).values; rm07 = r.rolling(7, min_periods=7).max().reindex(t7).values
    return pd.DataFrame({'H07': H7, 'dH07': H7 - H0, 'Hmx07': hmax - H7, 'R07': r07, 'Rmx07': rm07}, index=days)


def read_rain_hourly(hist, live):
    """인제 ASOS 시간강우: 과거 파일(비 온 시각만, 기간 안 빈 시각 = 0) + 매일 수집분(모든 시각). 시각 색인 Series."""
    parts = []
    if Path(hist).exists():
        d = pd.read_csv(hist, dtype={'tm': str})
        s = pd.Series(d.rn.values, index=pd.to_datetime(d.tm, format='%Y%m%d%H'))
        full = pd.date_range('2014-01-01 00:00', '2026-09-21 09:00', freq='h')
        parts.append(s.reindex(full).fillna(0.0))
    if Path(live).exists():
        d = pd.read_csv(live, dtype={'tm': str})
        parts.append(pd.Series(pd.to_numeric(d.rn, errors='coerce').values, index=pd.to_datetime(d.tm, format='%Y%m%d%H')))
    if not parts: return pd.Series(dtype=float)
    s = pd.concat(parts); return s[~s.index.duplicated(keep='last')].sort_index()


class StepOne:
    """1단계: D 입력 + 오늘 비 pT + 계절 + 아침 정보 → 오늘의 TGT 25개."""
    def __init__(self, F, P, M):
        self.F = F; self.P = P; self.M = M

    def X(self, T, pT):
        X = self.F.reindex(T - pd.Timedelta(days=1)).copy(); X.columns = [c + '_D' for c in X.columns]; X.index = T
        X['pT'] = pT; X['lpT'] = np.log1p(pT)
        X['msin'] = np.sin(2 * np.pi * T.month / 12); X['mcos'] = np.cos(2 * np.pi * T.month / 12)
        M = self.M.reindex(T)
        for c in MORN: X[c] = M[c].values
        return X

    def fit(self, upto=None):
        T = self.F.index[1:]; T = T[self.F.reindex(T - pd.Timedelta(days=1)).notna().sum(axis=1).values > 30]
        if upto is not None: T = T[T <= upto]
        XA = self.X(T, np.nan_to_num(self.P.reindex(T).values)); self.ms = {}
        for c in TGT:
            yv = self.F[c].reindex(T); ok = yv.notna().values
            self.ms[c] = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                          n_jobs=4, random_state=0).fit(XA[ok], yv[ok])
        return self

    def rain_T(self, T, pT):
        """오늘 강우 변수: 어제까지 관측 + 오늘 비 pT."""
        Pd = self.P; L = {0: np.asarray(pT, float)}
        for k in range(1, 31): L[k] = Pd.reindex(T - pd.Timedelta(days=k)).fillna(0).values
        f = pd.DataFrame(index=T)
        f['P0'] = L[0]; f['Pprev'] = L[1]; f['P3prior'] = L[1] + L[2] + L[3]; f['P7prior'] = sum(L[k] for k in range(1, 8))
        f['API7'] = sum(L[k] * 0.9 ** (k - 1) for k in range(1, 8)); f['P2'] = L[0] + L[1]; f['Pmax3'] = np.maximum.reduce([L[0], L[1], L[2]])
        f['P14prior'] = sum(L[k] for k in range(1, 15)); f['P30prior'] = sum(L[k] for k in range(1, 31))
        addD = self.F.ADD_prior.reindex(T - pd.Timedelta(days=1)).fillna(0).values
        f['ADD_prior'] = np.where(L[1] < 1, addD + 1, 0)
        return f

    def predict(self, T, pT):
        """오늘 입력 40개(유량·수위는 예측, 강우는 어제까지 관측 + pT, 기온은 어제 값)."""
        pT = np.broadcast_to(np.asarray(pT, float), (len(T),)).copy()
        X1 = self.X(T, pT); out = pd.DataFrame(index=T, columns=model.COLS, dtype=float)
        out[RAIN] = self.rain_T(T, pT)[RAIN].values
        out['msin'] = X1.msin.values; out['mcos'] = X1.mcos.values
        for c in ['TA', 'TA7', 'TAmin']: out[c] = self.F[c].reindex(T - pd.Timedelta(days=1)).values
        for c in TGT: out[c] = self.ms[c].predict(X1)
        return out[list(model.COLS)]


def ac_predict(step1, cm, T, pT):
    """A+C log 예측: 1단계로 오늘 입력을 만들고 C(+홍수일 보정)에 넣는다. cm = (모델, 보정, RelQ)."""
    m, sm, rq = cm
    XT = step1.predict(T, pT)
    return m.predict(rq.add(XT)) + sm + model.flood_shift(XT, FLOOD_C)


GROUPS_C = {k: (v + ['lQrel', 'Qpct'] if k == '유량' else v) for k, v in model.GROUPS.items()}


def estimate_c(df):
    """model.estimate와 같은 출력을 C로. 반환: (DataFrame(est, lo, hi, q_corr, q_fix, partial, qpk), (SHAP 묶음, 기준값 log), (C 모델, F, RelQ))."""
    F, qc, bad, qpk = model.features(df)
    cm = fit_c(F, qc, df.SS)
    m, sm, rq = cm; X = rq.add(F)
    fs = model.flood_shift(F, FLOOD_C)
    lp = m.predict(X) + sm + fs; blo, bhi = band_c(F)
    est = pd.DataFrame({'est': np.exp(lp), 'lo': np.exp(lp + blo), 'hi': np.exp(lp + bhi), 'q_corr': qc,
                        'q_fix': bad.astype(int), 'partial': F.isna().any(axis=1).astype(int), 'qpk': qpk}, index=df.index)
    sh, base = model.shap_groups(m, X[UCOLS], GROUPS_C, sm)
    sh['홍수일 보정'] = fs
    return est, (sh, base), (cm, F)


def live_h1(live, df, F, cm, M, kma_scale=0.63, fcst_rain=None, fcst_full=None):
    """운영 예측(live)의 1일 칸을 A+C로 바꾼다. 오늘 아침 07시 정보(수위·시간강우)가 다 있을 때만 바꾸고,
    없으면 기존(운영) 1일 예측을 그대로 두고 이유를 적는다. 시나리오 강우 g mm는 '실제로 내린 비'로 본다(배율 없음),
    기상청 예보는 과거 보관본 실제/예보 비율(0.63)로 줄여 넣는다(연구 평가와 같음)."""
    e = live['h'][1]; D = pd.Timestamp(live['issue']); T = pd.DatetimeIndex([D + pd.Timedelta(days=1)])
    mT = M.reindex(T)
    if mT.isna().any(axis=None):
        e['method'] = '운영'; e['method_note'] = '오늘 07시 수위·강우 미도착: ' + ', '.join(c for c in MORN if mT[c].isna().any())
        return None
    step1 = StepOne(F, df.P0, M).fit(upto=D)
    lp = lambda p: float(ac_predict(step1, cm, T, p)[0])
    lo, hi = BAND_AC['obs']; v = lp(0.0)
    e['obs'] = [round(float(np.exp(v + k)), 2) for k in (0, lo, hi)]
    lo, hi = BAND_AC['kma']
    e['rain'] = [[round(float(np.exp(lp(float(g)) + k)), 2) for k in (0, lo, hi)] for g in live['grid_mm']]
    e['rain_high'] = [None] * len(live['grid_mm'])
    t = str(T[0].date())
    if fcst_rain and t in fcst_rain and (fcst_full or {}).get(t, True):
        mm = float(fcst_rain[t]); v = lp(mm * kma_scale)
        e['kma'] = {'mm': round(mm, 1), 'days': 1, 'need': 1, 'v': [round(float(np.exp(v + k)), 2) for k in (0, lo, hi)]}
        e.pop('kma_partial', None)
    e['method'] = 'A+C'; e['morning'] = {c: (None if pd.isna(mT[c].iloc[0]) else round(float(mT[c].iloc[0]), 2)) for c in MORN}
    return step1
