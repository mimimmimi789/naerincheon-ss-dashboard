"""매일 실행: 자료 수집 → 입력 조립 → SS 추정 → 1·3·5일 예측 → docs/index.html 생성 (→ 선택: git push).

사용법:  python pipeline/run_daily.py [--no-fetch] [--push]
"""
import datetime as dt
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast as fcm  # noqa: E402
import manager as mg  # noqa: E402
import model  # noqa: E402
import sources as src  # noqa: E402

ROOT = src.ROOT
BASE = ROOT / 'data' / 'base'
AUX = ROOT / 'data' / 'auxdata'          # Windows에서 'aux'는 예약 이름이라 auxdata
DOCS = ROOT / 'docs'
LOCK = pd.Timestamp('2026-08-26')        # 기본 입력 파일 마지막 날(수집 시작 기준)
# 잠금 평가 기준(2026-11-04 사용자 요청): 2023년까지 실측으로만 학습한 모델로 2024년 이후 실측을 맞춘다(학습:시험 약 8:2, 3년, 2024-07-23 90 mg/L 포함).
# 이전 기준 2025-12-31(2026년만 26건)은 한 해에 기대 너무 짧았다. 2024–26 실측도 모델 구성을 고를 때 전진검증에 포함됐으므로 완전한 미지 자료는 아니다(학습에는 미사용).
LOCK_EVAL = pd.Timestamp('2023-12-31')
KST = dt.timezone(dt.timedelta(hours=9))
FCLOG = DOCS / 'forecast_log.csv'          # 매일 운영 예측 기록(기준일·선행일별 1줄). 새 실측이 오면 실제 예보 성적을 매긴다
MODEL_TAG = 'blend+temp+chain13+obs1chain+flood075+himode1 2026-11-04'       # 예측 모델이 바뀌면 이 이름도 바꿔 기록에서 구분


def csv(p, **kw):
    return pd.read_csv(p, encoding='utf-8-sig', parse_dates=['date'], **kw).set_index('date') if Path(p).exists() else None


def collect(today, key, status):
    yday = today - dt.timedelta(days=1)

    def run(name, fn, *a):
        try:
            r = fn(*a)
            status[name] = '정상' + (f' ({len(r)}행)' if hasattr(r, '__len__') and not isinstance(r, dict) else '')
            return r
        except Exception as e:  # 한 자료원이 실패해도 나머지는 계속
            hide = lambda s: s.replace(key, '***') if key else s     # 오류 메시지의 URL에 키가 들어 있으므로 가림
            status[name] = hide(f'실패: {type(e).__name__} {str(e)[:150]}')
            print(hide(traceback.format_exc()), file=sys.stderr)

    run('WAMIS 원대교 유량', src.wamis_flow, sorted({2026, yday.year}))
    st = csv(src.LIVE / 'wamis_stage_daily.csv')
    s0 = dt.date(2026, 1, 1) if st is None else (st.index.max() - pd.Timedelta(days=10)).date()
    run('WAMIS 원대교 수위', src.wamis_stage, s0, yday)
    if key:
        aw = csv(src.LIVE / 'kma_aws_rain.csv')
        a0 = LOCK.date() + dt.timedelta(days=1) if aw is None else (aw.index.max() - pd.Timedelta(days=7)).date()
        run('기상청 AWS 일강수', src.kma_aws_day, list(pd.date_range(a0, yday).date), key)
        tf = csv(src.LIVE / 'kma_asos_temp.csv')
        t0 = dt.date(2026, 10, 25) if tf is None else (tf.index.max() - pd.Timedelta(days=3)).date()
        run('기상청 인제 기온', src.kma_asos_temp, list(pd.date_range(t0, yday).date), key)
        run('기상청 단기예보', src.kma_fcst, yday, key)
        run('기상청 특보', src.kma_warnings, key, today)
    else:
        status['기상청'] = '키 없음 (.env 의 KMA_API_KEY)'
    run('물모아 총량측정망 SS', src.mulmoa_ss, yday.year)


def assemble():
    """기본 입력(2014-01-22–2026-08-26) 뒤에 수집 자료를 붙인다. 유량·강우가 모두 있는 마지막 날까지."""
    b = csv(BASE / 'base_inputs_20140122_20260826.csv')
    fl, st, aw, ss = (csv(src.LIVE / f) for f in ('wamis_flow.csv', 'wamis_stage_daily.csv', 'kma_aws_rain.csv', 'mulmoa_ss.csv'))
    end = b.index.max()
    if fl is not None and aw is not None:
        p = aw.reindex(columns=[f'rn_{s}' for s in src.AWS])
        w = pd.Series(src.AWS.values(), index=p.columns)
        P0 = (p * w).sum(axis=1, min_count=1) / p.notna().mul(w).sum(axis=1).replace(0, np.nan)
        Q = (fl.q_wondae * src.DAR).round(1)
        end = max(end, min(P0.dropna().index.max(), Q.dropna().index.max()))
    idx = pd.date_range(b.index.min(), end, freq='D')
    df = b.reindex(idx)
    new = idx > b.index.max()
    if new.any():
        df.loc[new, 'P0'] = P0.reindex(idx[new]).round(1).values
        df.loc[new, 'Q'] = Q.reindex(idx[new]).values
    if st is not None:
        s = st.reindex(idx)
        for c in ('Havg', 'Hmax', 'Hmin'):
            df[c] = df[c].where(df[c].notna(), s[c])
    # 기온(인제 ASOS 211 일최고·최저): 과거 파일 + 매일 수집분(겹치면 수집분 우선)
    ta = csv(AUX / 'inje_asos211_daily_temp_20140101_20261101.csv')
    tl = csv(src.LIVE / 'kma_asos_temp.csv')
    df['TAMAX'] = ta.TA_MAX.reindex(idx) if ta is not None else np.nan
    df['TAMIN'] = ta.TA_MIN.reindex(idx) if ta is not None else np.nan
    if tl is not None:
        t = tl.reindex(idx)
        df['TAMAX'] = t.ta_max.where(t.ta_max.notna(), df.TAMAX); df['TAMIN'] = t.ta_min.where(t.ta_min.notna(), df.TAMIN)
    # 원대교 시간 수위 → 날마다 오전 10시 기준 변수(과거 파일 + 매일 수집분)
    hh = model.read_hourly([AUX / 'wondae_wl_hourly_2014_2026.csv', src.LIVE / 'wamis_stage_hourly.csv'])
    if len(hh):
        df = df.join(model.hourly_feats(hh, idx, df.Havg))
    if ss is not None:
        s = ss[ss.index > b.index.max()].reindex(idx)
        df['SS'] = df.SS.where(df.SS.notna(), s.ss); df['Q_meas'] = df.Q_meas.where(df.Q_meas.notna(), s.q_meas)
    df.index.name = 'date'
    return df


def locked_eval(df, upto=LOCK_EVAL, flood=None):
    """upto까지 실측으로만 학습한 고정 모델로 그 뒤 실측을 맞춘 성적(월별 기준값과 비교).
    cov = 실측이 80% 범위 안에 든 비율. years = 연도별 성적."""
    flood = flood or model.FLOOD_2023
    since = str((upto + pd.Timedelta(days=1)).date())
    newo = df.index[(df.index > upto) & df.SS.notna() & (df.SS > 0)]
    if not len(newo): return {'n': 0, 'since': since, 'train': upto.year}
    F, _, _, _ = model.features(df)
    tr = (df.index <= upto) & df.SS.notna() & (df.SS > 0)
    m, sm = model.fit_log(model.Daily, F[tr], df.SS[tr].values)
    Fn = F.loc[newo]; lp = m.predict(Fn) + sm + model.flood_shift(Fn, flood); blo, bhi = model.band(Fn)
    p = np.exp(lp); y = df.SS[newo].values; lo, hi = np.exp(lp + blo), np.exp(lp + bhi)
    med = np.log(df.SS[tr]).groupby(df.index[tr].month).median()
    c = np.exp(newo.month.map(med).values.astype(float))
    r2 = lambda a, b: None if len(a) < 2 else round(float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)), 3)

    def stats(k):
        return {'n': int(k.sum()), 'mae': round(float(np.mean(np.abs(y[k] - p[k]))), 2), 'r2': r2(y[k], p[k]),
                'logr2': r2(np.log(y[k]), np.log(p[k])), 'mae_clim': round(float(np.mean(np.abs(y[k] - c[k]))), 2),
                'logr2_clim': r2(np.log(y[k]), np.log(c[k])), 'cov': round(float(np.mean((y[k] >= lo[k]) & (y[k] <= hi[k]))), 2),
                'ymax': float(y[k].max())}
    out = {'since': since, 'train': upto.year, **stats(np.ones(len(y), bool)),
           'rows': [[str(d.date()), float(a), round(float(b), 2), round(float(l), 2), round(float(h), 2)]
                    for d, a, b, l, h in zip(newo, y, p, lo, hi)]}
    yrs = sorted(set(newo.year))
    if len(yrs) > 1: out['years'] = {int(Y): stats(newo.year == Y) for Y in yrs}
    return out


def log_forecast(live, updated):
    """운영 예측을 기록 파일에 쌓는다. 같은 기준일을 다시 돌리면 마지막 실행으로 덮어쓴다."""
    rows = []
    for h, e in live['h'].items():
        k = e.get('kma') or {}
        rows.append({'issue': live['issue'], 'h': int(h), 'target': e['target'], 'obs': e['obs'][0], 'obs_lo': e['obs'][1],
                     'obs_hi': e['obs'][2], 'kma_mm': k.get('mm'), 'kma_days': k.get('days'), 'kma': (k.get('v') or [None])[0],
                     'model': MODEL_TAG, 'run': updated})
    new = pd.DataFrame(rows)
    old = pd.read_csv(FCLOG, encoding='utf-8-sig') if FCLOG.exists() else new.iloc[:0]
    out = pd.concat([old[old.issue != live['issue']], new]).sort_values(['issue', 'h'])
    out.to_csv(FCLOG, index=False, encoding='utf-8-sig')
    return out


def score_log(log, df):
    """기록된 운영 예측 중 목표일에 실측이 생긴 것만 채점. 기상청 예보값은 h일을 모두 받은 경우만 쓴다.
    월별 기준값 = 기준일까지 실측의 목표월 log SS 중앙값."""
    o = df.SS[df.SS > 0]; res = {'n_log': int(log.issue.nunique()), 'first': str(log.issue.min()), 'h': {}}
    for h in fcm.H:
        g = log[log.h == h].copy(); g['y'] = pd.to_datetime(g.target).map(o)
        g = g[g.y.notna()]
        if not len(g): res['h'][h] = {'n': 0}; continue
        clim = []
        for _, r in g.iterrows():
            a = o[o.index <= pd.Timestamp(r.issue)]; m = pd.Timestamp(r.target).month
            clim.append(float(np.exp(np.log(a[a.index.month == m]).median())))
        g['clim'] = clim; full = g.kma.notna() & (g.kma_days >= h)
        mae = lambda c, s=slice(None): round(float((g.loc[s, c] - g.loc[s, 'y']).abs().mean()), 2)
        res['h'][h] = {'n': int(len(g)), 'mae_obs': mae('obs'), 'mae_clim': mae('clim'), 'n_kma': int(full.sum()),
                       'mae_kma': mae('kma', full) if full.any() else None,
                       'rows': [[r.issue, r.target, float(r.y), r.obs, None if pd.isna(r.kma) else r.kma] for r in g.itertuples()]}
    return res


def build(df, est, hc, live, static, status, warnings, locked, updated, shp, wx=None, fscore=None, mgr=None):
    r = est.join(df[['P0', 'SS', 'Q_meas', 'TAMAX', 'TAMIN']])
    wf = csv(BASE / 'ss_wf_e3mono_flood075_20261104.csv').ss_wf_mgL.reindex(r.index)   # 운영 모델 전진검증(2019–26, 2026-11-04)
    f = lambda v, n=2: [None if pd.isna(x) else round(float(x), n) for x in v]
    o = r[r.SS.notna()]
    p90 = float(o[o.index < '2024-01-01'].SS.quantile(.9))
    D = {'d0': str(r.index.min().date()), 'p': f(r.P0, 1), 'est': f(r.est), 'lo': f(r.lo), 'hi': f(r.hi), 'q': f(r.q_corr, 1),
         'wf': f(wf), 'qfix': [i for i, x in enumerate(r.q_fix) if x == 1], 'part': [i for i, x in enumerate(r.partial) if x == 1],
         'obs': [[str(a.date()), float(b), None if pd.isna(c) else round(float(c), 1)] for a, b, c in zip(o.index, o.SS, o.Q_meas)],
         'meta': {'p90': round(p90, 1), 'n_est': int(r.SS.isna().sum()), 'updated': updated, 'sources': status,
                  'warnings': warnings, 'locked': locked, 'wx': wx, 'fscore': fscore}}
    F = {'off': (hc.index[0] - r.index.min()).days}
    for h in fcm.H:
        for t in ('obs', 'rain'):
            F[f'{h}{t}'] = [f(hc[f'h{h}_{t}']), f(hc[f'h{h}_{t}_lo']), f(hc[f'h{h}_{t}_hi'])]
    F['scen'] = live; F['bands'] = static['bands']; F['tbl'] = static['tbl']; F['real'] = static.get('tbl_real', [])
    D['fc'] = F
    # SHAP: 묶음별 log 기여, 일별 배열
    D['shap'] = {k: {'base': round(b, 3), 'off': (S.index[0] - r.index.min()).days,
                     'g': {g: [None if pd.isna(x) else round(float(x), 3) for x in S[g]] for g in S.columns}}
                 for k, (S, b) in shp.items()}
    # 과거 유사 강우사상: 실측일 T마다 3일 강우(T-2..T), 그 전 7일 강우(T-9..T-3), 사상 전 유량(T-3)
    P, Q = r.P0, r.q_corr
    R3 = P.rolling(3, min_periods=3).sum(); A7 = P.shift(3).rolling(7, min_periods=7).sum(); Q0 = Q.shift(3)
    ev = []
    for T, s in o.SS.items():
        if pd.notna(R3[T]) and pd.notna(A7[T]) and pd.notna(Q0[T]):
            ev.append([str(T.date()), round(float(R3[T]), 1), round(float(A7[T]), 1), round(float(Q0[T]), 1), float(s)])
    D['ev'] = ev
    if mgr:   # 취수원 관리자용 보조 정보(manager.py): 유사사상은 확장 열로 바꿔 넣음
        mgr = dict(mgr); D['ev'] = mgr.pop('ev', ev); D['meta']['mgr'] = mgr
    t = (Path(__file__).parent / 'template.html').read_text(encoding='utf-8')
    pl = (Path(__file__).parent / 'plotly-basic.min.js').read_text(encoding='utf-8').replace('</script', '<\\/script')
    html = (t.replace('/*DATA*/', json.dumps(D, separators=(',', ':'), ensure_ascii=False))
             .replace('/*GEO*/', (Path(__file__).parent / 'geo.json').read_text(encoding='utf-8')).replace('/*PLOTLY*/', pl))
    DOCS.mkdir(exist_ok=True)
    head = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<meta name="theme-color" content="#0e3b4a"><meta name="apple-mobile-web-app-capable" content="yes">'
            '<meta name="apple-mobile-web-app-title" content="내린천 SS"></head><body>\n')
    (DOCS / 'index.html').write_text(head + html + '\n</body></html>', encoding='utf-8')
    (DOCS.parent / 'build').mkdir(exist_ok=True)
    (DOCS.parent / 'build' / 'artifact.html').write_text(html, encoding='utf-8')   # Claude 아티팩트용(문서 틀은 아티팩트가 씌움)
    out = r.rename(columns={'P0': 'p0_mm', 'SS': 'ss_obs_mgL', 'Q_meas': 'q_meas_m3s', 'est': 'ss_est_mgL', 'lo': 'ss_lo80',
                            'hi': 'ss_hi80', 'TAMAX': 'ta_max_c', 'TAMIN': 'ta_min_c', 'q_corr': 'q_corr_m3s', 'q_fix': 'q_corrected', 'partial': 'input_partial', 'qpk': 'q_peak_m3s'})
    out.round(3).to_csv(DOCS / 'ss_daily.csv', encoding='utf-8-sig')
    hc.to_csv(DOCS / 'forecast_daily.csv', encoding='utf-8-sig')
    (DOCS / 'latest.json').write_text(json.dumps({'updated': updated, 'last_day': str(r.index.max().date()), 'forecast': live,
                                                  'sources': status, 'locked': locked}, ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    args = set(sys.argv[1:])
    now = dt.datetime.now(KST); today = now.date()
    status = {}
    key = src.kma_key()
    if '--no-fetch' not in args:
        collect(today, key, status)
    df = assemble()
    est, shp_est = model.estimate(df)
    static = json.loads((BASE / 'forecast_static.json').read_text(encoding='utf-8'))
    fr = fcm.Forecaster(df.P0, est.q_corr, est.qpk, df.Havg, df.Hmax, df.SS, static, model.temp_feats(df))
    fr.chain = fcm.FlowChain(fr, df.P0, est.q_corr, est.qpk, df.Hmax, df.Hmin, df.SS, model.temp_feats(df))   # 2026-11-03 1·3일 연결 혼합
    hc = fr.hindcast()
    fj = src.LIVE / 'kma_fcst.json'
    rain = wx = full = None
    if fj.exists():
        j = json.loads(fj.read_text(encoding='utf-8'))
        if j['issue'] == str(df.index.max().date()): rain = j['rain_mm']; full = j.get('full_day')
        if j.get('weather') and (now.date() - dt.date.fromisoformat(j['issue'])).days <= 2:   # 오래된 예보는 날씨 칸에 쓰지 않음
            wx = {'base': j['base'], 'point': j.get('weather_point', ''), 'days': j['weather'], 'example': bool(j.get('example'))}
    live = fr.live(rain, full)
    wj = src.LIVE / 'kma_warnings.json'
    warnings = json.loads(wj.read_text(encoding='utf-8')) if wj.exists() else None
    locked = [locked_eval(df)]
    updated = now.strftime('%Y-%m-%d %H:%M KST')
    fscore = score_log(log_forecast(live, updated), df)
    o = df.SS[df.SS > 0].dropna()
    fetch = mg.update_fetch_log(src.LIVE, status, str(o.index.max().date()) if len(o) else None, now)
    mgr = {'fresh': mg.freshness(df, src.LIVE, status, j if fj.exists() else None, now, updated, fetch),
           'cover': mg.coverage(live), 'exceed': mg.exceed(hc, df.SS, live), 'ood': mg.ood(fr),
           'ev': mg.events(df.P0, est.q_corr, df.Havg, df.Hmax, df.Hmin, df.SS)}
    build(df, est, hc, live, static, status, warnings, locked, updated, {'est': shp_est, 'h1': fr.shap_h1()}, wx, fscore, mgr)
    print('마지막 자료일', df.index.max().date(), '| 실측', int(df.SS.notna().sum()), '| 잠금평가', [L.get('n') for L in locked], '|', status)
    if '--push' in args:
        g = lambda *c: subprocess.run(['git', '-C', str(ROOT), *c], check=False)
        g('add', 'docs', 'data/live')
        g('commit', '-m', f'자동 갱신 {updated} (자료 {df.index.max().date()}까지)')
        g('push')


if __name__ == '__main__':
    main()
