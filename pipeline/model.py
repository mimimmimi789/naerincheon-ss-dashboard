"""SS 일별 추정 모델 (2026-11-02 개선안): [XGB 10시드 평균 + 안전 Ridge] log 평균, 중앙값 보정
+ 확장변수 + 수위 기반 유량보정 + 첨두수위. (2026-10-30 고정안은 XGB 1개 + Duan 평균 보정)

입력 df: 날짜 색인, 열 P0(유역 Thiessen 일강우 mm), Q(원대교 일유량 × 면적비 m³/s), Havg·Hmin·Hmax(원대교 수위 m), SS(실측 mg/L)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 전진검증 잔차 10–90% (2019–26, n=295, 포함률 79.7%; 이전 XGB+Duan은 -1.057, 0.553)
LOG_LO, LOG_HI = -0.7625834110974855, 0.6699241069878145


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
    """BagXGB와 SafeRidge의 log 예측 평균."""
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


COLS = ['P0', 'Pprev', 'P3prior', 'P7prior', 'API7', 'ADD_prior', 'Q', 'Qprev', 'Qdelta1', 'Qratio3', 'msin', 'mcos',
        'lQ', 'dlQ', 'dlQ2', 'Qmax3', 'Q_rel30', 'Q_rel90', 'Q_over_base', 'days_since_big', 'P2', 'Pmax3',
        'P14prior', 'P30prior', 'lQpk', 'Hrise', 'dHmax']


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
          '계절': ['msin', 'mcos']}


def shap_groups(m, X, groups, offset=0.0):
    """TreeSHAP(log 척도; 혼합모델은 XGB 평균 SHAP과 Ridge 선형 기여의 평균)을 변수 묶음별로 더한다. 반환: (묶음별 DataFrame, 기준값 log)."""
    c = m.contribs(X) if hasattr(m, 'contribs') else m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    cols = list(X.columns)
    out = pd.DataFrame({g: c[:, [cols.index(v) for v in vs]].sum(axis=1) for g, vs in groups.items()}, index=X.index)
    return out, float(c[0, -1] + offset)


def estimate(df):
    """전체 실측으로 학습해 매일 SS 사후추정.
    반환: (DataFrame(est, lo, hi, q_corr, q_fix, partial, qpk), (SHAP 묶음별 기여 DataFrame, 기준값 log))"""
    F, qc, bad, qpk = features(df)
    tr = df.SS.notna() & (df.SS > 0)
    m, sm = fit_log(Blend, F[tr], df.SS[tr].values)
    lp = m.predict(F) + sm
    est = pd.DataFrame({'est': np.exp(lp), 'lo': np.exp(lp + LOG_LO), 'hi': np.exp(lp + LOG_HI), 'q_corr': qc,
                        'q_fix': bad.astype(int), 'partial': F.isna().any(axis=1).astype(int), 'qpk': qpk}, index=df.index)
    return est, shap_groups(m, F, GROUPS, sm)
