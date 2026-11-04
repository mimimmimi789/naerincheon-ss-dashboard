"""SS 일별 추정 모델 (2026-11-02 개선안): [XGB 10시드 평균 + 안전 Ridge] log 평균, 중앙값 보정
+ 확장변수 + 수위 기반 유량보정 + 첨두수위 + 기온·시간 수위(2026-11-03). (2026-10-30 고정안은 XGB 1개 + Duan 평균 보정)

입력 df: 날짜 색인, 열 P0(유역 Thiessen 일강우 mm), Q(원대교 일유량 × 면적비 m³/s), Havg·Hmin·Hmax(원대교 수위 m), SS(실측 mg/L),
TAMAX·TAMIN(인제 ASOS 211 일최고·최저기온 ℃), HCOLS(원대교 시간 수위 변수, hourly_feats)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 전진검증 잔차 10–90% (2019–26, n=295; 2026-11-04 사상변수·단조 반영. 이전 -0.720, 0.651; 시간수위 추가 전 -0.677, 0.611; 기온 추가 전 -0.763, 0.670; 이전 XGB+Duan은 -1.057, 0.553)
LOG_LO, LOG_HI = -0.6722, 0.6768   # 2026-11-04 홍수일 보정(¾) 반영 후 (보정 전 -0.6948, 0.6352). 참고용, 화면은 BAND 사용
# 2026-11-04 80% 범위를 비·수위 상승일(flood_shift와 같은 판정)과 평상시로 나눔. 같은 전진검증 잔차(2019–26, n=295: 평상시 228, 비 온 날 67).
# 하나로 쓰면 평상시 포함 84%·비 온 날 66%였는데, 나누면 각각 80%·79%. 범위 폭은 평상시 3.4배, 비 온 날 5.8배(이전 공통 3.9배).
BAND = {'dry': (-0.5533, 0.6633), 'wet': (-0.7893, 0.9605)}


def XG(random_state=0, **kw):
    return xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=3, n_jobs=4, random_state=random_state, **kw)


class BagXGB:
    """XGB를 시드만 바꿔 n개 학습해 평균(분산 감소). mono: 증가 제약을 줄 열 이름."""
    def __init__(self, n=10, mono=()):
        self.n = n; self.mono = tuple(mono)

    def fit(self, X, y):
        kw = {'monotone_constraints': tuple(int(c in self.mono) for c in X.columns)} if self.mono else {}
        self.cols = list(X.columns); self.ms = [XG(i, **kw).fit(X, y) for i in range(self.n)]; return self

    def predict(self, X): return np.mean([m.predict(X[self.cols]) for m in self.ms], axis=0)

    def contribs(self, X):
        return np.mean([m.get_booster().predict(xgb.DMatrix(X[self.cols]), pred_contribs=True) for m in self.ms], axis=0)


class SafeRidge:
    """log 선형(Ridge). 입력은 학습 범위로 자르고, 출력은 학습 최대 log SS + ln2(최대 실측의 2배)로 제한.
    나무 모델이 못 하는 고유량 외삽을 맡되, 실측이 없는 큰 홍수일에 값이 폭주하지 않게 한다."""
    def fit(self, X, y):
        self.cols = list(X.columns); self.lo = X.min(); self.hi = X.max(); self.cap = float(np.max(y) + np.log(2))
        self.m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                               RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(X, y); return self

    def _x(self, X): return X[self.cols].clip(self.lo, self.hi, axis=1)

    def predict(self, X): return np.minimum(self.m.predict(self._x(X)), self.cap)

    def contribs(self, X):
        imp, sc, rg = self.m.steps[0][1], self.m.steps[1][1], self.m.steps[2][1]
        z = sc.transform(imp.transform(self._x(X)))
        return np.column_stack([z * rg.coef_, np.full(len(z), rg.intercept_)])


class Blend:
    """BagXGB와 SafeRidge의 log 예측 평균. mono: XGB에 증가 제약을 줄 열 이름."""
    def __init__(self, mono=()): self.mono = mono

    def fit(self, X, y):
        self.a = BagXGB(mono=self.mono).fit(X, y); self.b = SafeRidge().fit(X, y); return self

    def predict(self, X): return (self.a.predict(X) + self.b.predict(X)) / 2

    def contribs(self, X): return (self.a.contribs(X) + self.b.contribs(X)) / 2


def build(df):
    P = df.P0; Q = df.Q; f = pd.DataFrame(index=df.index)
    f['P0'] = P; f['Pprev'] = P.shift(1)
    f['P3prior'] = P.shift(1).rolling(3, min_periods=1).sum()
    f['P7prior'] = P.shift(1).rolling(7, min_periods=1).sum()
    f['API7'] = sum(P.shift(k) * 0.9 ** (k - 1) for k in range(1, 8))
    wet = (P >= 1).astype(float); g = (wet.shift(1) == 1).cumsum()
    f['ADD_prior'] = wet.shift(1).eq(0).groupby(g).cumsum()
    f['Q'] = Q; f['Qprev'] = Q.shift(1); f['Qdelta1'] = Q - Q.shift(1)
    f['Qratio3'] = Q / Q.shift(1).rolling(3, min_periods=1).mean()
    m = df.index.month; f['msin'] = np.sin(2 * np.pi * m / 12); f['mcos'] = np.cos(2 * np.pi * m / 12)
    lq = np.log(Q)
    f['lQ'] = lq
    f['P2'] = P + P.shift(1)
    f['Pmax3'] = P.rolling(3, min_periods=1).max()
    f['P14prior'] = P.shift(1).rolling(14, min_periods=1).sum()
    f['P30prior'] = P.shift(1).rolling(30, min_periods=1).sum()
    f['Qmax3'] = Q.rolling(3, min_periods=1).max()
    f['Q_rel30'] = Q / Q.shift(1).rolling(30, min_periods=5).max()
    f['Q_rel90'] = Q / Q.shift(1).rolling(90, min_periods=10).max()
    f['Q_over_base'] = Q / Q.rolling(30, min_periods=5).min()
    f['dlQ'] = lq - lq.shift(1); f['dlQ2'] = lq - lq.shift(2)
    big = (Q > 100).astype(float)
    f['days_since_big'] = big.eq(0).groupby(big.cumsum()).cumsum().clip(upper=365)
    return f


HCOLS = ['Hh', 'Hh_d3', 'Hh_d6', 'Hh_pk6', 'Hh_pk24', 'Hh_rng24', 'Hh_avg', 'Hh_hpk', 'Hh_pk72']
HH_AT = 10   # 채수 대표 시각 14시(실측 중앙 13:53) − 원대교→내린천2 흐름 지연 4시간


COLS = ['P0', 'Pprev', 'P3prior', 'P7prior', 'API7', 'ADD_prior', 'Q', 'Qprev', 'Qdelta1', 'Qratio3', 'msin', 'mcos',
        'lQ', 'dlQ', 'dlQ2', 'Qmax3', 'Q_rel30', 'Q_rel90', 'Q_over_base', 'days_since_big', 'P2', 'Pmax3',
        'P14prior', 'P30prior', 'lQpk', 'Hrise', 'dHmax', 'TA', 'TA7', 'TAmin', 'Pev'] + HCOLS

# 2026-11-04 일별 모델 XGB 단조 제약: 유량·수위·강우가 커지면 SS 추정이 줄지 않게 한다.
# 사상 변수(Hh_hpk·Hh_pk72·Pev)와 함께 전진검증 2019–26 로그R² 0.761→0.768, MAE 2.95→2.86 (capstone/perf_explore_20261104)
MONO = ('Q', 'lQ', 'Qmax3', 'lQpk', 'Hrise', 'dHmax', 'P0', 'P2', 'Pmax3')


def Daily(): return Blend(mono=MONO)


# 2026-11-04 홍수일 보정(사용자 요청: 고농도 우선, 강도 ¾). 비·수위 상승일(P2≥10 mm 또는 24시간 수위 범위≥0.3 m)에는
# 첨두유량(lQpk)에 따라 log 보정량을 Huber 선형으로, 그 밖의 날은 잔차 중앙값으로 옮긴다. 계수는 2016–26 전진검증 잔차로 적합.
# 검증(그해 이전 잔차로만 적합, 2019–26): 로그R² 0.768→0.770, R² 0.578→0.663, MAE 2.86→2.79, 평상시(<10) MAE 0.82→0.79,
# 물 불어난 날 ≥10 mg/L 예측/실측 중앙 0.73→0.92, 25 이상 경보 7/7→7/10 (capstone/perf_explore_20261104 p10.py)
FLOOD = dict(a=-0.4684, b=0.1296, wet_med=0.0495, dry_med=-0.1238, fill=2.441, k=0.75)
FLOOD_2025 = dict(a=-0.4694, b=0.1386, wet_med=0.0749, dry_med=-0.1123, fill=2.441, k=0.75)   # 잠금 평가용(2025년까지 잔차)
FLOOD_2024 = dict(a=-0.5241, b=0.1510, wet_med=0.0780, dry_med=-0.1192, fill=2.441, k=0.75)   # 잠금 평가 2(2024년까지 잔차, 2016–24 전진검증)


def wet_day(F):
    """비·수위 상승일: 2일 강우 10 mm 이상 또는 24시간 수위 범위 0.3 m 이상."""
    return ((F.P2 >= 10) | (F.Hh_rng24 >= 0.3)).values


def band(F):
    """log 80% 범위(아래, 위). 비·수위 상승일은 넓게."""
    w = wet_day(F)
    return np.where(w, BAND['wet'][0], BAND['dry'][0]), np.where(w, BAND['wet'][1], BAND['dry'][1])


def flood_shift(F, c=FLOOD):
    wet = wet_day(F)
    lin = c['a'] + c['b'] * F.lQpk.fillna(c['fill']).values
    return np.where(wet, c['wet_med'] + c['k'] * (lin - c['wet_med']), c['dry_med'])




def hourly_feats(h, days, havg=None):
    """원대교 시간 수위(색인 = 시각, 24시는 다음날 0시)로 날마다 HH_AT시 기준 변수(2026-11-03 추가).
    Hh 수위, Hh_d3·Hh_d6 직전 3·6시간 변화, Hh_pk6·Hh_pk24 직전 6·24시간 최고−현재, Hh_rng24 24시간 범위, Hh_avg 그날 일평균 대비.
    2026-11-04 홍수 사상 변수: Hh_hpk 직전 72시간 최고수위 이후 경과시간(상승부≈0), Hh_pk72 직전 72시간 최고−현재(하강량)."""
    h = h[(h > 0.5) & (h < 10)]
    h = h[~h.index.duplicated(keep='last')].sort_index().asfreq('h').interpolate(limit=6, limit_area='inside')
    t = days + pd.Timedelta(hours=HH_AT)
    at = lambda k: h.reindex(t - pd.Timedelta(hours=k)).values
    m6 = h.rolling(7, min_periods=4).max(); m24 = h.rolling(25, min_periods=13).max(); n24 = h.rolling(25, min_periods=13).min()
    H0 = at(0)
    out = pd.DataFrame({'Hh': H0, 'Hh_d3': H0 - at(3), 'Hh_d6': H0 - at(6), 'Hh_pk6': m6.reindex(t).values - H0,
                        'Hh_pk24': m24.reindex(t).values - H0, 'Hh_rng24': m24.reindex(t).values - n24.reindex(t).values}, index=days)
    out['Hh_avg'] = out.Hh - (havg.reindex(days).values if havg is not None else np.nan)
    m72 = h.rolling(73, min_periods=36).max()
    hpk = h.rolling(73, min_periods=36).apply(lambda a: len(a) - 1 - np.nanargmax(a) if np.isfinite(a).any() else np.nan, raw=True)
    out['Hh_hpk'] = hpk.reindex(t).values; out['Hh_pk72'] = m72.reindex(t).values - H0
    return out


def read_hourly(paths):
    """ymdh(YYYYMMDDHH, 01–24시), wl 열 CSV들을 합쳐 시각 색인 Series로. 뒤 파일 우선."""
    parts = []
    for p in paths:
        try:
            d = pd.read_csv(p, dtype={'ymdh': str}, encoding='utf-8-sig')
        except FileNotFoundError:
            continue
        parts.append(pd.Series(pd.to_numeric(d.wl, errors='coerce').values,
                               index=pd.to_datetime(d.ymdh.str[:8]) + pd.to_timedelta(d.ymdh.str[8:10].astype(int), unit='h')))
    if not parts: return pd.Series(dtype=float)
    s = pd.concat(parts); return s[~s.index.duplicated(keep='last')].sort_index()


def temp_feats(df):
    """인제 ASOS 기온(2026-11-03 추가): 당일 평균((최고+최저)/2), 7일 평균, 당일 최저. 자료가 없으면 NaN."""
    tx = df['TAMAX'] if 'TAMAX' in df else pd.Series(np.nan, index=df.index)
    tn = df['TAMIN'] if 'TAMIN' in df else pd.Series(np.nan, index=df.index)
    tx = tx.interpolate(limit=3, limit_area='inside'); tn = tn.interpolate(limit=3, limit_area='inside')
    ta = (tx + tn) / 2
    return pd.DataFrame({'TA': ta, 'TA7': ta.rolling(7, min_periods=4).mean(), 'TAmin': tn}, index=df.index)


def stage_flow(df):
    """연도별 log Q ~ 평균수위 2차식(잔차 0.7 넘는 날 빼고 재적합)으로 수위환산 유량과 첨두유량."""
    d = df.dropna(subset=['Q', 'Havg']); d = d[d.Q > 0]
    QH = pd.Series(np.nan, index=df.index); QPK = QH.copy()
    years = sorted(set(df.index.year))
    for y in years:
        x = d[d.index.year == y]
        if len(x) < 100:                     # 자료가 적은 해(새 해 초)는 전년과 합쳐 적합
            x = d[(d.index.year >= y - 1) & (d.index.year <= y)]
        if len(x) < 100: continue
        c = np.polyfit(x.Havg, np.log(x.Q), 2); res = np.log(x.Q) - np.polyval(c, x.Havg)
        x = x[abs(res) < 0.7]; c = np.polyfit(x.Havg, np.log(x.Q), 2)
        m = df.index.year == y; cap = x.Havg.max() + 0.5
        QH[m] = np.exp(np.polyval(c, df.Havg[m].clip(upper=cap))); QPK[m] = np.exp(np.polyval(c, df.Hmax[m].clip(upper=cap)))
    return QH, QPK


def features(df):
    """유량보정 후 모델 입력. 반환: (입력표, 보정유량, 보정여부, 첨두유량)"""
    QH, QPK = stage_flow(df)
    bad = ((np.log(df.Q) - np.log(QH)).abs() > 0.7) | (df.Q.isna() & QH.notna())
    dfx = df.copy(); dfx['Q'] = df.Q.where(~bad, QH)
    F = build(dfx)
    F['lQpk'] = np.log(QPK.where(QPK.notna(), dfx.Q))
    F['Hrise'] = df.Hmax - df.Hmin
    F['dHmax'] = df.Hmax - df.Hmin.rolling(60, min_periods=10).quantile(0.1)
    F = F.join(temp_feats(df))
    P = df.P0.fillna(0); wet = P >= 1
    F['Pev'] = P.where(wet, 0).groupby((~wet).cumsum()).cumsum()   # 2026-11-04 연속강우(1 mm 이상 날이 이어지는 동안) 누적
    for c in HCOLS:
        F[c] = df[c] if c in df else np.nan
    return F[COLS], dfx.Q, bad, QPK


def fit_log(mk, X, y):
    """log 목표 적합 + 중앙값 보정(3겹 내부 잔차의 중앙값).
    Duan(평균) 보정은 고농도 꼬리 때문에 평상시 농도를 약 1.3배 과대추정해 로그 R²·MAE를 떨어뜨렸다."""
    m = mk().fit(X, np.log(y)); res = []
    for a, b in KFold(3, shuffle=True, random_state=0).split(X):
        mm = mk().fit(X.iloc[a], np.log(y[a])); res += list(np.log(y[b]) - mm.predict(X.iloc[b]))
    return m, float(np.median(res))


# SHAP 기여도를 화면에 보일 변수 묶음
GROUPS = {'유량': ['Q', 'Qprev', 'Qdelta1', 'Qratio3', 'lQ', 'dlQ', 'dlQ2', 'Qmax3', 'Q_rel30', 'Q_rel90', 'Q_over_base', 'days_since_big'],
          '당일·전일 강우': ['P0', 'Pprev', 'P2', 'Pmax3'],
          '선행강우 (3–30일)': ['P3prior', 'P7prior', 'P14prior', 'P30prior', 'API7'],
          '무강우일수': ['ADD_prior'],
          '수위·첨두유량': ['lQpk', 'Hrise', 'dHmax'],
          '계절': ['msin', 'mcos'], '기온': ['TA', 'TA7', 'TAmin'], '시간 수위 (오전)': HCOLS[:7],
          '홍수 사상': ['Hh_hpk', 'Hh_pk72', 'Pev']}


def shap_groups(m, X, groups, offset=0.0):
    """TreeSHAP(log 척도; 혼합모델은 XGB 평균 SHAP과 Ridge 선형 기여의 평균)을 변수 묶음별로 더한다. 반환: (묶음별 DataFrame, 기준값 log)."""
    c = m.contribs(X) if hasattr(m, 'contribs') else m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    cols = list(X.columns)
    out = pd.DataFrame({g: c[:, [cols.index(v) for v in vs if v in cols]].sum(axis=1) for g, vs in groups.items()
                        if any(v in cols for v in vs)}, index=X.index)
    return out, float(c[0, -1] + offset)


def estimate(df):
    """전체 실측으로 학습해 매일 SS 사후추정.
    반환: (DataFrame(est, lo, hi, q_corr, q_fix, partial, qpk), (SHAP 묶음별 기여 DataFrame, 기준값 log))"""
    F, qc, bad, qpk = features(df)
    tr = df.SS.notna() & (df.SS > 0)
    m, sm = fit_log(Daily, F[tr], df.SS[tr].values)
    lp = m.predict(F) + sm + flood_shift(F); blo, bhi = band(F)
    est = pd.DataFrame({'est': np.exp(lp), 'lo': np.exp(lp + blo), 'hi': np.exp(lp + bhi), 'q_corr': qc,
                        'q_fix': bad.astype(int), 'partial': F.isna().any(axis=1).astype(int), 'qpk': qpk}, index=df.index)
    sh, base = shap_groups(m, F, GROUPS, sm)
    sh['홍수일 보정'] = flood_shift(F)   # 보정량(log)을 따로 한 묶음으로 보여 합계가 추정값과 맞게
    return est, (sh, base)
