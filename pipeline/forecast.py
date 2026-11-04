"""SS 1·3·5일 예측: 기준일 D 24시까지의 관측(+ D+1..D+h 예보강우)으로 D+h일 실측 SS를 예측한다.

모델 선택·전진검증 성적은 data/base/forecast_static.json에 고정한다(2026-11-02 개선안: XGB 10시드+안전 Ridge 혼합,
중앙값 보정. 이전 forecast_20261030 모델도 MODELS에 남겨 두어 static의 pick만 바꾸면 되돌릴 수 있다).
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

H = [1, 3, 5]   # 2026-11-03 7일→5일: 7일은 월별 기준값과 차이 유의하지 않고 기상청 단기예보(D+5)로 채울 수 없음
SM = ['lQ', 'Qrel30', 'AP7', 'AP14', 'msin', 'mcos']
FUT = ['Pfut', 'Pfut_max']
GRID = [0, 5, 10, 20, 30, 50, 80, 120]


def XG(): return xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=3, n_jobs=4, random_state=0)
def RG(): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


from model import Blend
import model as _model

# 2026-11-03 고농도 개선: 강우→유량→SS 연결 모델을 1·3일 '예보강우' 예측에 운영 모델과 log 반반 혼합한다.
# 검증(기상청 예보 보관본, capstone/highss_20261103 c3_fix.py): 1일 로그R² 0.749→0.759, MAE 3.02→2.85,
# 3일(앞 2일 예보) 0.599→0.625, MAE 3.19→3.02, 유출형 고농도 예측/실측 0.24→0.35. 5일은 실제 예보 검증 불가라 제외.
CHAIN_H = (1, 3)
# 2026-11-04: 관측만 1일 예측에도 연결 모델(앞으로 비 0 mm)을 log 반반 혼합. 전진검증 2019–26 로그R² 0.703→0.718, MAE 3.51→3.45.
# 3일은 효과 없음(0.448→0.446) (capstone/perf_explore_20261104 p4.py)
CHAIN_OBS_H = (1,)
FLV = ['lQ', 'lQpk', 'Hrise', 'dlQ']
RS_RAIN = ['P0', 'Pprev', 'P3prior', 'P7prior', 'API7', 'ADD_prior', 'P2', 'Pmax3', 'P14prior', 'P30prior']


def XGF(): return xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                   min_child_weight=5, n_jobs=4, random_state=0)


class FlowChain:
    """(1) 유량모델: 기준일 D 관측(Forecaster 입력) + D+1..D+h 일강우 → 목표일 T의 log 유량·log 첨두유량·수위변동·유량변화.
    매일 유량 자료가 있으므로 모든 날(~4,000일)로 학습한다. (2) 축소 일별 SS 모델: T의 강우변수·계절·기온 + 유량변수 4개
    (운영 일별 모델과 같은 혼합, 실측일 학습). 예측 때 T의 유량은 (1)의 예측값, 기온은 D 관측, 아직 오지 않은 날의 강우는
    예보(또는 시나리오)를 넣는다."""
    def __init__(self, fr, P, Q, Qpk, Hmax, Hmin, SS, T):
        self.fr = fr; self.P = P; idx = P.index
        b = _model.build(pd.DataFrame({'P0': P, 'Q': Q}, index=idx))
        self.R = b[RS_RAIN]; self.ADD = b.ADD_prior; self.T = T[['TA', 'TA7', 'TAmin']]
        lq = np.log(Q.clip(lower=0.05))
        self.Y = pd.DataFrame({'lQ': lq, 'lQpk': np.log(Qpk.where(Qpk.notna(), Q).clip(lower=0.05)), 'Hrise': Hmax - Hmin,
                               'dlQ': lq - lq.shift(1)}, index=idx)
        m = idx.month
        self.RSX = self.R.assign(msin=np.sin(2 * np.pi * m / 12), mcos=np.cos(2 * np.pi * m / 12)).join(self.T).join(self.Y)
        o = SS.dropna(); self.obs = o[o > 0]; self._fm = {}; self._rs = {}

    def fm_X(self, h, tgt, fut=None):
        """행 = 목표일 tgt. fut: (len, h) 배열 D+1..D+h 일강우(없으면 실제 내린 비)."""
        X = self.fr.f.reindex(tgt - pd.Timedelta(days=h)).copy(); X.index = tgt
        X['msin'] = np.sin(2 * np.pi * tgt.month / 12); X['mcos'] = np.cos(2 * np.pi * tgt.month / 12)
        for k in range(h):
            X[f'Pf{k + 1}'] = self.P.reindex(tgt - pd.Timedelta(days=h - 1 - k)).values if fut is None else fut[:, k]
        X['Pfsum'] = X[[f'Pf{k + 1}' for k in range(h)]].sum(axis=1, min_count=h)
        return X

    def _fit(self, h, key, upto):
        """upto: 학습에 쓰는 목표일 상한(이 날 미만). key로 재사용."""
        if (h, key) not in self._fm:
            tgt = self.P.index[self.P.index < upto]; X = self.fm_X(h, tgt)
            self._fm[(h, key)] = {c: XGF().fit(X[self.Y[c].reindex(tgt).notna().values], self.Y[c].reindex(tgt).dropna())
                                  for c in FLV}
        if key not in self._rs:
            t = self.obs.index[self.obs.index < upto]
            self._rs[key] = _model.fit_log(Blend, self.RSX.loc[t], self.obs[t].values)
        return self._fm[(h, key)], self._rs[key]

    def rain_feats(self, h, days, fut):
        """목표일 강우변수: 시차 k<h(D 이후)는 fut, k≥h는 관측."""
        tgt = days + pd.Timedelta(days=h); n = len(days)
        L = {k: (fut[:, h - 1 - k] if k < h else self.P.reindex(tgt - pd.Timedelta(days=k)).fillna(0).values) for k in range(31)}
        f = pd.DataFrame(index=tgt)
        f['P0'] = L[0]; f['Pprev'] = L[1]; f['P3prior'] = L[1] + L[2] + L[3]; f['P7prior'] = sum(L[k] for k in range(1, 8))
        f['API7'] = sum(L[k] * 0.9 ** (k - 1) for k in range(1, 8)); f['P2'] = L[0] + L[1]
        f['Pmax3'] = np.maximum.reduce([L[0], L[1], L[2]])
        f['P14prior'] = sum(L[k] for k in range(1, 15)); f['P30prior'] = sum(L[k] for k in range(1, 31))
        add = self.ADD.reindex(days).fillna(0).values; pD = self.P.reindex(days).fillna(0).values
        dry = np.ones(n, bool); run = np.zeros(n)
        for k in range(1, h):
            run = np.where(dry & (L[k] < 1), run + 1, run); dry &= (L[k] < 1)
        f['ADD_prior'] = np.where(dry, run + np.where(pD < 1, 1 + add, 0), run)
        return f

    def predict(self, h, days, fut=None, train_upto=None):
        """log 예측(목표일 = days+h). fut=None이면 실제 내린 비(사후). train_upto=None이면 기준일 연도 이전 자료로만 학습."""
        tgt = days + pd.Timedelta(days=h)
        if fut is None: fut = np.column_stack([self.P.reindex(days + pd.Timedelta(days=k + 1)).values for k in range(h)])
        fut = np.nan_to_num(np.asarray(fut, float))
        Xf = self.fm_X(h, tgt, fut); R = self.rain_feats(h, days, fut)
        out = pd.Series(np.nan, index=days)
        groups = [(days.year == Y, Y, pd.Timestamp(f'{Y}-01-01')) for Y in sorted(set(days.year))] if train_upto is None \
            else [(np.ones(len(days), bool), str(train_upto), train_upto + pd.Timedelta(days=1))]
        for sel, key, upto in groups:
            if not sel.any() or (self.obs.index < upto).sum() < 30: continue
            fm, (rs, sm) = self._fit(h, key, upto)
            X = R[sel].copy(); t = tgt[sel]
            X['msin'] = np.sin(2 * np.pi * t.month / 12); X['mcos'] = np.cos(2 * np.pi * t.month / 12)
            X[['TA', 'TA7', 'TAmin']] = self.T.reindex(days[sel]).values
            for c in FLV: X[c] = fm[c].predict(Xf[sel])
            out[sel] = rs.predict(X[self.RSX.columns]) + sm
        return out


# h: (관측만 모델, 예보강우 모델) — (만드는 함수 또는 'clim', 입력열 'all' 또는 목록, 예보강우 사용)
MODELS = {'월별 기준값': ('clim', None, False), 'XGB 전체입력 (관측만)': (XG, 'all', False),
          'Ridge 전체입력 + 예보강우': (RG, 'all', True), 'Ridge 축소입력 + 예보강우': (RG, SM, True),
          'XGB+Ridge 혼합 (관측만)': (Blend, 'all', False),
          'XGB(예보강우 단조)+Ridge 혼합 + 예보강우': (lambda: Blend(mono=FUT), 'all', True)}
# 이전 모델(Duan 평균 보정)과 개선 모델(중앙값 보정) 구분
DUAN = {'XGB 전체입력 (관측만)', 'Ridge 전체입력 + 예보강우', 'Ridge 축소입력 + 예보강우'}


def feats(P, Q, Qpk, Havg, Hmax, T=None):
    """T: model.temp_feats 결과(TA, TA7, TAmin). 기준일 D의 관측 기온이라 예측에 그대로 쓸 수 있다."""
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
    if T is not None:
        f = f.join(T[['TA', 'TA7', 'TAmin']])
    return f


class Forecaster:
    def __init__(self, P, Q, Qpk, Havg, Hmax, SS, static, T=None):
        self.P = P; self.f = feats(P, Q, Qpk, Havg, Hmax, T)
        o = SS.dropna(); self.obs = o[o > 0]
        self.static = static; self.pick = {int(k): v for k, v in static['pick'].items()}
        self._cache = {}; self.chain = None

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

    def fit_pred(self, mk, Xtr, ytr, Xte, key=None, duan=False):
        """학습(같은 key면 재사용) + 보정(3겹 내부 잔차: 이전 모델 Duan 평균, 개선 모델 중앙값)."""
        if key is None or key not in self._cache:
            m = mk().fit(Xtr, ytr); res = []
            for a, b in KFold(3, shuffle=True, random_state=0).split(Xtr):
                mm = mk().fit(Xtr.iloc[a], ytr[a]); res += list(ytr[b] - mm.predict(Xtr.iloc[b]))
            off = np.log(np.mean(np.exp(res))) if duan else float(np.median(res))
            if key is None: return m.predict(Xte) + off
            self._cache[key] = (m, off)
        m, off = self._cache[key]
        return m.predict(Xte) + off

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
                key = (h, nm, str(train_upto) if train_upto is not None else int(days[sel][0].year))
                out[sel] = self.fit_pred(mk, X.loc[tr, c], y[tr], Xd.loc[sel, c], key, nm in DUAN)
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
                if tag == 'rain' and self.chain is not None and h in CHAIN_H:
                    lp = (lp + self.chain.predict(h, days)) / 2
                if tag == 'obs' and self.chain is not None and h in CHAIN_OBS_H:
                    lp = (lp + self.chain.predict(h, days, np.zeros((len(days), h)))) / 2
                if tag == 'rain': lp[days + pd.Timedelta(days=h) > last] = np.nan
                for k, a in (('', 0), ('_lo', lo), ('_hi', hi)):
                    out[f'h{h}_{tag}{k}'] = np.exp(lp + a).round(2)
        return out

    G1 = {'유량': ['lQ', 'lQ1', 'dlQ', 'dlQ3', 'lQmax3', 'Qrel30'], '당일·전일 강우': ['P0', 'P1', 'Pmax3'],
          '선행강우 (3–30일)': ['AP3', 'AP7', 'AP14', 'API7'], '무강우일수': ['ADD'], '수위·첨두유량': ['lQpk', 'Hrise'],
          '계절': ['msin', 'mcos'], '기온': ['TA', 'TA7', 'TAmin']}

    def shap_h1(self):
        """1일 예측(관측만)을 전체 실측으로 학습해 모든 기준일의 SHAP 묶음 기여(log). live()와 같은 학습을 재사용."""
        from model import shap_groups
        nm = self.pick[1][0]; mk = MODELS[nm][0]
        X, y = self.rows(1, False); D = self.f.index[-1]
        key = (1, nm, str(D))
        if key not in self._cache: self.fit_pred(mk, X[X.index <= D], y[X.index <= D], X.iloc[:1], key, nm in DUAN)
        m, off = self._cache[key]
        Xd, _ = self.issue_X(self.f.index, 1)
        return shap_groups(m, Xd[X.columns], self.G1, off)

    def live(self, fcst_rain=None, fcst_full=None):
        """마지막 기준일 운영 예측. fcst_rain: {'YYYY-MM-DD': mm} 기상청 예보(Thiessen 가중 일강우).
        fcst_full: {'YYYY-MM-DD': bool} 그날 24시간 예보가 다 있는지(없으면 모두 완전한 날로 본다).
        기상청 예보 반영값은 h일 모두 완전한 예보가 있을 때만 낸다."""
        last = self.f.index[-1:]; D = last[0]
        res = {'issue': str(D.date()), 'grid_mm': GRID, 'h': {}}
        for h in H:
            nm_o, nm_r = self.pick[h]
            lo, hi, _ = self.static['bands'][f'h{h}_obs']
            lp = self.predict(h, nm_o, last, train_upto=D).iloc[0]
            if self.chain is not None and h in CHAIN_OBS_H:
                lp = (lp + self.chain.predict(h, last, np.zeros((1, h)), train_upto=D).iloc[0]) / 2
            e = {'target': str((D + pd.Timedelta(days=h)).date()), 'obs': [round(float(np.exp(lp + k)), 2) for k in (0, lo, hi)], 'rain': []}
            lo, hi, _ = self.static['bands'][f'h{h}_rain']
            mix = self.chain is not None and h in CHAIN_H
            for g in GRID:
                lp = self.predict(h, nm_r, last, (np.array([g]), np.array([g])), train_upto=D).iloc[0]
                if mix:   # 연결 모델 쪽 시나리오는 h일 동안 고르게 내린다고 본다
                    lp = (lp + self.chain.predict(h, last, np.full((1, h), g / h), train_upto=D).iloc[0]) / 2
                e['rain'].append([round(float(np.exp(lp + k)), 2) for k in (0, lo, hi)])
            if fcst_rain:
                want = [str((D + pd.Timedelta(days=k)).date()) for k in range(1, h + 1)]
                full = fcst_full or {}
                have = [fcst_rain[x] for x in want if x in fcst_rain and full.get(x, True)]
                if len(have) == h:
                    tot, mx = float(sum(have)), float(max(have))
                    lp = self.predict(h, nm_r, last, (np.array([tot]), np.array([mx])), train_upto=D).iloc[0]
                    if mix:   # 연결 모델에는 과거 보관본 기준 실제/예보 비율로 줄인 예보강우를 넣는다(기상청 예보가 비를 많이 예보)
                        sc = float(self.static.get('kma_rain_scale', 1.0))
                        lp = (lp + self.chain.predict(h, last, np.array([have]) * sc, train_upto=D).iloc[0]) / 2
                    e['kma'] = {'mm': round(tot, 1), 'days': len(have), 'need': h,
                                'v': [round(float(np.exp(lp + k)), 2) for k in (0, lo, hi)]}
                elif have:
                    # 예보가 h일을 다 채우지 못하면 반영값을 내지 않는다. 학습은 h일 실제 강우 합으로 했으므로
                    # 모자란 날을 0 mm로 넣으면 과소예측된다(2026-11-03 감사: 7일에 2일 예보만 넣으면 로그 R² 0.288 < 관측만 0.383).
                    e['kma_partial'] = {'mm': round(float(sum(have)), 1), 'days': len(have), 'need': h}
            res['h'][h] = e
        return res
