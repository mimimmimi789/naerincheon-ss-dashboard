"""SS 일별 추정 모델 (2026-10-30 고정): XGB(log SS, Duan 보정) + 확장변수 + 수위 기반 유량보정 + 첨두수위.

입력 df: 날짜 색인, 열 P0(유역 Thiessen 일강우 mm), Q(원대교 일유량 × 면적비 m³/s), Havg·Hmin·Hmax(원대교 수위 m), SS(실측 mg/L)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

# 전진검증 잔차 10–90% (2019–26, n=295, 포함률 79.7%)
LOG_LO, LOG_HI = -1.0571774399385403, 0.5531360317582736


def XG():
    return xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=3, n_jobs=4, random_state=0)


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
    """log 목표 적합 + Duan 보정계수(3겹 내부 잔차)."""
    m = mk().fit(X, np.log(y)); res = []
    for a, b in KFold(3, shuffle=True, random_state=0).split(X):
        mm = mk().fit(X.iloc[a], np.log(y[a])); res += list(np.log(y[b]) - mm.predict(X.iloc[b]))
    return m, np.log(np.mean(np.exp(res)))


def estimate(df):
    """전체 실측으로 학습해 매일 SS 사후추정. 반환 DataFrame(est, lo, hi, q_corr, q_fix, partial, qpk)."""
    F, qc, bad, qpk = features(df)
    tr = df.SS.notna() & (df.SS > 0)
    m, sm = fit_log(XG, F[tr], df.SS[tr].values)
    lp = m.predict(F) + sm
    return pd.DataFrame({'est': np.exp(lp), 'lo': np.exp(lp + LOG_LO), 'hi': np.exp(lp + LOG_HI), 'q_corr': qc,
                         'q_fix': bad.astype(int), 'partial': F.isna().any(axis=1).astype(int), 'qpk': qpk}, index=df.index)
