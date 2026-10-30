"""매일 실행: 자료 수집 → 입력 조립 → SS 추정 → 1·3·7일 예측 → docs/index.html 생성 (→ 선택: git push).

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
import model  # noqa: E402
import sources as src  # noqa: E402

ROOT = src.ROOT
BASE = ROOT / 'data' / 'base'
DOCS = ROOT / 'docs'
LOCK = pd.Timestamp('2026-08-26')        # 잠금 평가 기준: 이 날까지 실측으로만 학습
KST = dt.timezone(dt.timedelta(hours=9))


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
    if ss is not None:
        s = ss[ss.index > b.index.max()].reindex(idx)
        df['SS'] = df.SS.where(df.SS.notna(), s.ss); df['Q_meas'] = df.Q_meas.where(df.Q_meas.notna(), s.q_meas)
    df.index.name = 'date'
    return df


def locked_eval(df):
    """8/26까지 실측으로만 학습한 고정 모델로 그 뒤 새 실측을 맞춘 성적."""
    newo = df.index[(df.index > LOCK) & df.SS.notna() & (df.SS > 0)]
    if not len(newo): return {'n': 0}
    F, _, _, _ = model.features(df)
    tr = (df.index <= LOCK) & df.SS.notna() & (df.SS > 0)
    m, sm = model.fit_log(model.XG, F[tr], df.SS[tr].values)
    p = np.exp(m.predict(F.loc[newo]) + sm); y = df.SS[newo].values
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 1 else None
    return {'n': int(len(y)), 'mae': round(float(np.mean(np.abs(y - p))), 2), 'r2': None if r2 is None else round(float(r2), 3),
            'rows': [[str(d.date()), float(a), round(float(b), 2)] for d, a, b in zip(newo, y, p)]}


def build(df, est, hc, live, static, status, warnings, locked, updated):
    r = est.join(df[['P0', 'SS', 'Q_meas']])
    wf = csv(BASE / 'daily_xgblog_final_20261030.csv').ss_wf_mgL.reindex(r.index)
    f = lambda v, n=2: [None if pd.isna(x) else round(float(x), n) for x in v]
    o = r[r.SS.notna()]
    p90 = float(o[o.index < '2024-01-01'].SS.quantile(.9))
    D = {'d0': str(r.index.min().date()), 'p': f(r.P0, 1), 'est': f(r.est), 'lo': f(r.lo), 'hi': f(r.hi), 'q': f(r.q_corr, 1),
         'wf': f(wf), 'qfix': [i for i, x in enumerate(r.q_fix) if x == 1], 'part': [i for i, x in enumerate(r.partial) if x == 1],
         'obs': [[str(a.date()), float(b), None if pd.isna(c) else round(float(c), 1)] for a, b, c in zip(o.index, o.SS, o.Q_meas)],
         'meta': {'p90': round(p90, 1), 'n_est': int(r.SS.isna().sum()), 'updated': updated, 'sources': status,
                  'warnings': warnings, 'locked': locked}}
    F = {'off': (hc.index[0] - r.index.min()).days}
    for h in fcm.H:
        for t in ('obs', 'rain'):
            F[f'{h}{t}'] = [f(hc[f'h{h}_{t}']), f(hc[f'h{h}_{t}_lo']), f(hc[f'h{h}_{t}_hi'])]
    F['scen'] = live; F['bands'] = static['bands']; F['tbl'] = static['tbl']
    D['fc'] = F
    t = (Path(__file__).parent / 'template.html').read_text(encoding='utf-8')
    pl = (Path(__file__).parent / 'plotly-basic.min.js').read_text(encoding='utf-8').replace('</script', '<\\/script')
    html = (t.replace('/*DATA*/', json.dumps(D, separators=(',', ':'), ensure_ascii=False))
             .replace('/*GEO*/', (Path(__file__).parent / 'geo.json').read_text(encoding='utf-8')).replace('/*PLOTLY*/', pl))
    DOCS.mkdir(exist_ok=True)
    (DOCS / 'index.html').write_text(html, encoding='utf-8')
    out = r.rename(columns={'P0': 'p0_mm', 'SS': 'ss_obs_mgL', 'Q_meas': 'q_meas_m3s', 'est': 'ss_est_mgL', 'lo': 'ss_lo80',
                            'hi': 'ss_hi80', 'q_corr': 'q_corr_m3s', 'q_fix': 'q_corrected', 'partial': 'input_partial', 'qpk': 'q_peak_m3s'})
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
    est = model.estimate(df)
    static = json.loads((BASE / 'forecast_static.json').read_text(encoding='utf-8'))
    fr = fcm.Forecaster(df.P0, est.q_corr, est.qpk, df.Havg, df.Hmax, df.SS, static)
    hc = fr.hindcast()
    fj = src.LIVE / 'kma_fcst.json'
    rain = None
    if fj.exists():
        j = json.loads(fj.read_text(encoding='utf-8'))
        if j['issue'] == str(df.index.max().date()): rain = j['rain_mm']
    live = fr.live(rain)
    wj = src.LIVE / 'kma_warnings.json'
    warnings = json.loads(wj.read_text(encoding='utf-8')) if wj.exists() else None
    locked = locked_eval(df)
    updated = now.strftime('%Y-%m-%d %H:%M KST')
    build(df, est, hc, live, static, status, warnings, locked, updated)
    print('마지막 자료일', df.index.max().date(), '| 실측', int(df.SS.notna().sum()), '| 잠금평가', locked.get('n'), '|', status)
    if '--push' in args:
        g = lambda *c: subprocess.run(['git', '-C', str(ROOT), *c], check=False)
        g('add', 'docs', 'data/live')
        g('commit', '-m', f'자동 갱신 {updated} (자료 {df.index.max().date()}까지)')
        g('push')


if __name__ == '__main__':
    main()
