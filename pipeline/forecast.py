"""SS 1·3·7일 예측: 기준일 D 24시까지의 관측(+ D+1..D+h 예보강우)으로 D+h일 실측 SS를 예측한다.

모델 선택·전진검증 성적은 forecast_20261030 분석에서 고정했다(data/base/forecast_static.json).
여기서는 매일 (1) 2019년 이후 기준일별 표본외 예측(그 해 이전 자료로만 학습)과
(2) 마지막 기준일 운영 예측(전체 자료 학습, 기상청 예보강우와 0–120 mm 시나리오)을 다시 만든다.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

H = [1, 3, 7]
SM = ['lQ', 'Qrel30', 'AP7', 'AP14', 'msin', 'mcos']
FUT = ['Pfut', 'Pfut_max']
GRID = [0, 5, 10, 20, 30, 50, 80, 120]


def XG(): return xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=3, n_jobs=4, random_state=0)
def RG(): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


# h: (관측만 모델, 예보강우 모델) — (만드는 함수 또는 'clim', 입력열 'all' 또는 목록, 예보강우 사용)
MODELS = {'월별 기준값': ('clim', None, False), 'XGB 전체입력 (관측만)': (XG, 'all', False),
          'Ridge 전체입력 + 예보강우': (RG, 'all', True), 'Ridge 축소입력 + 예보강우': (RG, SM, True)}


def feats(P, Q, Qpk, Havg, Hmax):
    f = pd.DataFrame(index=P.index)
    f['P0'] = P; f['P1'] = P.shift(1)
    f['AP3'] = P.rolling(3, min_periods=3).sum(); f['AP7'] = P.rolling(7, min_periods=7).sum()
    f['AP14'] = P.rolling(14, min_periods=14).sum(); f['Pmax3'] = P.rolling(3, min_periods=3).max()
    f['API7'] = sum(P.shift(k) * 0.85 ** k for k in range(7))
    wet = (P >= 1).astype(float)
    f['ADD'] = wet.groupby(wet.cumsum()).cumcount().where(wet == 0, 0)
    lq = np.log(Q.clip(lower=0.05))
    f['lQ'] = lq; f['lQ1'] = lq.shift(1); f['dlQ'] = lq - lq.shift(1); f['dlQ3'] = lq - lq.shift(3)
    f['lQmax3'] = lq.rolling(3, min_periods=2).max()
    f['Qrel30'] = lq - lq.rolling(30, min_periods=10).median()
    f['lQpk'] = np.log(Qpk.clip(lower=0.05))
    f['Hrise'] = Hmax - Havg
    return f


class Forecaster:
    def __init__(self, P, Q, Qpk, Havg, Hmax, SS, static):
        self.P = P; self.f = feats(P, Q, Qpk, Havg, Hmax)
        o = SS.dropna(); self.obs = o[o > 0]
        self.static = static; self.pick = {int(k): v for k, v in static['pick'].items()}

    def rows(self, h, fut):
        X = self.f.shift(h).loc[self.obs.index].copy(); t = self.obs.index
        X['msin'] = np.sin(2 * np.pi * t.month / 12); X['mcos'] = np.cos(2 * np.pi * t.month / 12)
        if fut:
            X['Pfut'] = np.log1p(self.P.rolling(h, min_periods=h).sum().reindex(t).values)
            X['Pfut_max'] = np.log1p(self.P.rolling(h, min_periods=h).max().reindex(t).values)
        return X, np.log(self.obs.values)

    def issue_X(self, days, h, fut_rain=None):
        tgt = days + pd.Timedelta(days=h); Xd = self.f.loc[days].copy()
        Xd['msin'] = np.sin(2 * np.pi * tgt.month / 12); Xd['mcos'] = np.cos(2 * np.pi * tgt.month / 12)
        if fut_rain is not None: Xd['Pfut'], Xd['Pfut_max'] = np.log1p(fut_rain[0]), np.log1p(fut_rain[1])
        return Xd, tgt

    @staticmethod
    def fit_pred(mk, Xtr, ytr, Xte):
        m = mk().fit(Xtr, ytr); res = []
        for a, b in KFold(3, shuffle=True, random_state=0).split(Xtr):
            mm = mk().fit(Xtr.iloc[a], ytr[a]); res += list(ytr[b] - mm.predict(Xtr.iloc[b]))
        return m.predict(Xte) + np.log(np.mean(np.exp(res)))

    def predict(self, h, nm, days, fut_rain=None, train_upto=None):
        """log 예측. train_upto=None이면 기준일 연도 이전 실측으로만 학습(표본외)."""
        mk, cols, fut = MODELS[nm]; X, y = self.rows(h, fut)
        Xd, tgt = self.issue_X(days, h, fut_rain)
        out = pd.Series(np.nan, index=days)
        groups = [(days.year == Y, X.index.year < Y) for Y in sorted(set(days.year))] if train_upto is None \
            else [(np.ones(len(days), bool), X.index <= train_upto)]
        for sel, tr in groups:
            if not sel.any() or tr.sum() < 30: continue
            if mk == 'clim':
                med = pd.Series(y[tr]).groupby(X.index.month[tr]).median()
                out[sel] = tgt[sel].month.map(med).astype(float)
            else:
                c = list(X.columns) if cols == 'all' else cols + FUT if fut else cols
                out[sel] = self.fit_pred(mk, X.loc[tr, c], y[tr], Xd.loc[sel, c])
        return out

    def hindcast(self, start='2019-01-01'):
        """기준일별 표본외 예측. 예보강우 모델은 실제 내린 비를 예보 자리에 넣는다(사후 점검)."""
        days = self.f.index[self.f.index >= start]; last = self.f.index[-1]
        out = pd.DataFrame(index=days)
        for h in H:
            for tag, nm in zip(['obs', 'rain'], self.pick[h]):
                lo, hi, _ = self.static['bands'][f'h{h}_{tag}']; fr = None
                if tag == 'rain':
                    tgt = days + pd.Timedelta(days=h)
                    fr = (self.P.rolling(h, min_periods=h).sum().reindex(tgt).values,
                          self.P.rolling(h, min_periods=h).max().reindex(tgt).values)
                lp = self.predict(h, nm, days, fr)
                if tag == 'rain': lp[days + pd.Timedelta(days=h) > last] = np.nan
                for k, a in (('', 0), ('_lo', lo), ('_hi', hi)):
                    out[f'h{h}_{tag}{k}'] = np.exp(lp + a).round(2)
        return out

    def live(self, fcst_rain=None):
        """마지막 기준일 운영 예측. fcst_rain: {'YYYY-MM-DD': mm} 기상청 예보(Thiessen 가중 일강우)."""
        last = self.f.index[-1:]; D = last[0]
        res = {'issue': str(D.date()), 'grid_mm': GRID, 'h': {}}
        for h in H:
            nm_o, nm_r = self.pick[h]
            lo, hi, _ = self.static['bands'][f'h{h}_obs']
            lp = self.predict(h, nm_o, last, train_upto=D).iloc[0]
            e = {'target': str((D + pd.Timedelta(days=h)).date()), 'obs': [round(float(np.exp(lp + k)), 2) for k in (0, lo, hi)], 'rain': []}
            lo, hi, _ = self.static['bands'][f'h{h}_rain']
            for g in GRID:
                lp = self.predict(h, nm_r, last, (np.array([g]), np.array([g])), train_upto=D).iloc[0]
                e['rain'].append([round(float(np.exp(lp + k)), 2) for k in (0, lo, hi)])
            if fcst_rain:
                want = [str((D + pd.Timedelta(days=k)).date()) for k in range(1, h + 1)]
                have = [fcst_rain[x] for x in want if x in fcst_rain]
                if have:
                    tot, mx = float(sum(have)), float(max(have))
                    lp = self.predict(h, nm_r, last, (np.array([tot]), np.array([mx])), train_upto=D).iloc[0]
                    e['kma'] = {'mm': round(tot, 1), 'days': len(have), 'need': h,
                                'v': [round(float(np.exp(lp + k)), 2) for k in (0, lo, hi)]}
            res['h'][h] = e
        return res
