"""취수원 관리자용 보조 정보 (2026-11-03): 자료 최신성, 예보 확보율, 1일 기준 초과확률, 입력 학습범위 이탈, 유사 강우사상 확장.

모델 출력값은 바꾸지 않는다. 여기서 만드는 값은 화면 표시와 판단 근거용이다.
- 초과확률: 운영 점예측 + 과거 표본외 예측(hindcast)의 log 잔차 분포. 2020–26 시간순 검증에서 1일 앞만 판별력이 있어
  (SS≥10.2: PR-AUC 0.81, SS≥25: 0.57, 발생률 0.10·0.04) 1일만 화면에 쓴다. 3·7일은 PR-AUC가 발생률 수준이라 쓰지 않는다.
- 미채수일은 평가에서 뺀다. 확률은 '정확히 그날' 초과확률이며 '기간 안에 한 번이라도'와 다르다.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESH = [10.2, 25.0]            # 10.2 = 2014–23 실측 상위 10%(통계 기준), 25 = 하천 생활환경기준 Ia SS(참고). 취수 운영기준 아님
FETCH_LOG = 'fetch_ok.json'      # 자료원별 마지막 수집 성공 시각, 새 SS 실측이 처음 보인 시각

# 자료별 허용 지연(일): 넘으면 '늦음'. SS는 약 8일 간격 채수 + 검증 후 공개라 따로 둔다.
LIMIT = {'rain': 2, 'flow': 2, 'stage': 2, 'temp': 2, 'fcst': 1.5, 'ss': 14}


def update_fetch_log(live_dir, status, ss_last, now):
    """수집이 '정상'인 자료원은 성공 시각을 기록. 새 SS 실측일이 처음 보이면 그 시각을 '공개 확인'으로 남긴다."""
    p = Path(live_dir) / FETCH_LOG
    try:
        j = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    except ValueError:
        j = {}
    ok = j.setdefault('ok', {})
    t = now.strftime('%Y-%m-%d %H:%M')
    for k, v in (status or {}).items():
        if str(v).startswith('정상'):
            ok[k] = t
    seen = j.setdefault('ss_seen', {})
    if ss_last and ss_last not in seen and status and str(status.get('물모아 총량측정망 SS', '')).startswith('정상'):
        seen[ss_last] = t
    p.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding='utf-8')
    return j


def _last(s):
    s = s.dropna() if s is not None else None
    return None if s is None or not len(s) else s.index.max()


def freshness(df, live_dir, status, fcst, now, updated, fetch):
    """자료별 마지막 관측(또는 발표) 시점과 상태. 반환: 행 목록 [이름, 마지막, 경과일, 상태(ok|late|fail|none), 비고]."""
    live_dir = Path(live_dir); today = pd.Timestamp(now.date())
    rd = lambda f: pd.read_csv(live_dir / f, encoding='utf-8-sig') if (live_dir / f).exists() else None
    rows = []

    def add(name, src, last, lim, note='', shown=None):
        st = (status or {}).get(src, '')
        if last is None:
            state, age = 'none', None
        else:
            age = (today - pd.Timestamp(last).normalize()).days
            state = 'late' if age > lim else 'ok'
        if str(st).startswith('실패'):
            state = 'fail'; note = (note + ' · ' if note else '') + '오늘 수집 실패, 이전 자료 사용'
        okt = (fetch.get('ok') or {}).get(src)
        if okt: note = (note + ' · ' if note else '') + '마지막 수집 성공 ' + okt
        rows.append([name, shown or (None if last is None else str(pd.Timestamp(last).date())), age, state, note])

    obs = df.SS[df.SS > 0].dropna()
    ss_last = None if not len(obs) else str(obs.index.max().date())
    seen = (fetch.get('ss_seen') or {}).get(ss_last)
    add('SS 실측 (총량측정망)', '물모아 총량측정망 SS', ss_last, LIMIT['ss'],
        ('공개 확인 ' + seen if seen else '공개 확인 시각 기록 없음') + ' · 약 8일 간격 채수, 검증 후 공개까지 수주–2개월')
    aw = rd('kma_aws_rain.csv')
    if aw is not None:
        aw = aw.set_index(pd.to_datetime(aw.date)).drop(columns='date')
        la = _last(aw.dropna(how='all'))
        miss = [c.replace('rn_', '') for c in aw.columns if la is not None and pd.isna(aw.at[la, c])]
        add('강우 (AWS 원통·기린·내면)', '기상청 AWS 일강수', la, LIMIT['rain'], ('마지막 날 결측 관측소 ' + ','.join(miss)) if miss else '')
    else:
        add('강우 (AWS 원통·기린·내면)', '기상청 AWS 일강수', _last(df.P0), LIMIT['rain'])
    fl = rd('wamis_flow.csv')
    add('유량 (WAMIS 원대교)', 'WAMIS 원대교 유량', _last(fl.set_index(pd.to_datetime(fl.date)).q_wondae) if fl is not None else _last(df.Q), LIMIT['flow'])
    hh = rd('wamis_stage_hourly.csv')
    if hh is not None and len(hh):
        y = hh.dropna().ymdh.astype(str).max(); t = pd.Timestamp(y[:8]) + pd.Timedelta(hours=int(y[8:10]))
        add('수위 (WAMIS 원대교, 시간)', 'WAMIS 원대교 수위', t, LIMIT['stage'], shown=t.strftime('%Y-%m-%d %H시'))
    else:
        add('수위 (WAMIS 원대교, 시간)', 'WAMIS 원대교 수위', _last(df.Havg), LIMIT['stage'])
    tf = rd('kma_asos_temp.csv')
    add('기온 (인제 ASOS)', '기상청 인제 기온', _last(tf.set_index(pd.to_datetime(tf.date)).ta_max) if tf is not None else None, LIMIT['temp'])
    if fcst:
        b = pd.Timestamp(fcst['base']); age = (pd.Timestamp(now.replace(tzinfo=None)) - b).total_seconds() / 86400
        st = 'late' if age > LIMIT['fcst'] else 'ok'
        if str((status or {}).get('기상청 단기예보', '')).startswith('실패'): st = 'fail'
        rows.append(['기상청 단기예보 발표', fcst['base'], round(age, 1), st, '예보 강수 ' + str(len(fcst.get('rain_mm') or {})) + '일치'])
    else:
        rows.append(['기상청 단기예보 발표', None, None, 'none', '예보 파일 없음'])
    rows.append(['예측 생성', updated, 0, 'ok', '자료 마지막 날 ' + str(df.index.max().date()) + ' 기준'])
    return rows


def coverage(live):
    """예보강우 확보 일수: h별 {'have','need','full'}. live()의 kma(완전) 또는 kma_partial(부족)을 그대로 읽는다."""
    out = {}
    for h, e in live['h'].items():
        k = e.get('kma') or e.get('kma_partial')
        out[int(h)] = {'have': int(k['days']) if k else 0, 'need': int(h), 'full': bool(k and k['days'] >= int(h) and 'kma' in e)}
    return out


def exceed(hc, ss, live, cols=('h1_obs', 'h1_rain'), thresh=THRESH, lag=60):
    """1일 뒤 SS가 기준 이상일 확률. hc: Forecaster.hindcast() 결과(기준일 색인, mg/L), ss: 실측.
    잔차 = log(실측) − log(표본외 예측), 기준일보다 lag일 이상 앞선 목표일만(공개 지연)."""
    o = ss[ss > 0].dropna(); D = pd.Timestamp(live['issue']); res = {}
    e = live['h'][1] if 1 in live['h'] else live['h'].get('1')
    for c in cols:
        if c not in hc: continue
        p = hc[c].dropna(); tg = p.index + pd.Timedelta(days=1)
        y = o.reindex(tg); m = y.notna().values & (tg <= D - pd.Timedelta(days=lag))
        r = np.log(y.values[m]) - np.log(p.values[m])
        if len(r) < 50: continue
        kind = 'obs' if c.endswith('obs') else 'kma'
        if kind == 'obs': v = e['obs'][0]
        elif e.get('kma'): v = e['kma']['v'][0]
        else: continue
        res[kind] = {'v': v, 'n_res': int(len(r)), 'p': {str(C): round(float(np.mean(np.log(v) + r >= np.log(C))), 3) for C in thresh}}
    return res


def ood(fr, h=1):
    """마지막 기준일 예측 입력이 학습 실측일 입력의 범위를 벗어났는지. 반환: {'out': [(묶음, 변수, 값, 최소, 최대)], 'p99': [...]}."""
    X, _ = fr.rows(h, False); x = fr.f.iloc[-1]
    grp = {v: g for g, vs in fr.G1.items() for v in vs}
    out, tail = [], []
    for c in fr.f.columns:
        col = X[c].dropna()
        if not len(col) or pd.isna(x[c]): continue
        lo, hi = col.min(), col.max(); q1, q99 = col.quantile(.01), col.quantile(.99)
        rec = [grp.get(c, c), c, round(float(x[c]), 3), round(float(lo), 3), round(float(hi), 3)]
        if x[c] < lo or x[c] > hi: out.append(rec)
        elif x[c] < q1 or x[c] > q99: tail.append(rec)
    miss = [c for c in fr.f.columns if pd.isna(x[c])]
    return {'out': out, 'tail': tail, 'missing': miss, 'n_train': int(len(X))}


def events(P, Q, Havg, Hmax, Hmin, SS):
    """유사 강우사상 후보: 실측일 T마다 [날짜, 3일 강우(T-2..T), 선행 7일(T-9..T-3), 사상 전 유량(T-3), 실측 SS, 월,
    그날 유량, 그날 수위 변동폭(최고−최저, m), 사상 전후(T-3..T+3) 최대 실측 SS, 선행 30일(T-32..T-3)]."""
    o = SS[SS > 0].dropna()
    R3 = P.rolling(3, min_periods=3).sum(); A7 = P.shift(3).rolling(7, min_periods=7).sum(); Q0 = Q.shift(3)
    A30 = P.shift(3).rolling(30, min_periods=25).sum()
    f = lambda v, n=1: None if pd.isna(v) else round(float(v), n)
    ev = []
    for T, s in o.items():
        if pd.isna(R3.get(T)) or pd.isna(A7.get(T)) or pd.isna(Q0.get(T)): continue
        win = o[(o.index >= T - pd.Timedelta(days=3)) & (o.index <= T + pd.Timedelta(days=3))]
        ev.append([str(T.date()), f(R3[T]), f(A7[T]), f(Q0[T]), float(s), int(T.month), f(Q.get(T)),
                   f(Hmax.get(T) - Hmin.get(T), 2) if pd.notna(Hmax.get(T)) and pd.notna(Hmin.get(T)) else None,
                   float(win.max()), f(A30.get(T), 0)])
    return ev
