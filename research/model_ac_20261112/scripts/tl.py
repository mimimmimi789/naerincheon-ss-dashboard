"""전이학습 공통: 원천(주변 총량측정망 지점) 표 만들기 + 공통 변수.
원천 지점은 실측일 유량이 있는 자연하천·댐하류 지점. 강우는 가까운 ASOS/AWS 평균, 기온은 가까운 ASOS.
공통 변수(원천·내린천2 모두 계산 가능): 상대유량(lQrel = log Q − 지점 중앙값), 지점 내 유량 백분위, 강우 이력, 계절, 기온.
목표는 log SS − 지점 중앙값(상대 농도) → 지점마다 다른 바탕 농도를 빼고 '반응 모양'만 옮긴다."""
from fw import *
N = REPO + '/data/neighbor/'
a = pd.read_csv(N + 'asos_211_212_daily.csv', parse_dates=['date'])
RN = a.pivot(index='date', columns='stn', values='rn'); TX = a.pivot(index='date', columns='stn', values='ta_max'); TN = a.pivot(index='date', columns='stn', values='ta_min')
SL = pd.read_csv(N + 'neighbor_ss_slim.csv', parse_dates=['date'])
IDX = pd.date_range('2014-01-01', '2026-11-04')
# 지점: (강우 관측소, 기온 관측소, 묶음)  near = 내린천과 가까운 자연하천, far = 북한강 본류·댐 영향·먼 곳
SITES = {
    '1011A20': ([594, 518, 211], 211, 'near'),   # 인북A (인북천)
    '1012A15': ([211, 594, 518], 211, 'near'),   # 소양A2
    '1014A40': ([212, 535, 522], 212, 'near'),   # 홍천A2
    '1014A70': ([212, 535, 522], 212, 'near'),   # 홍천A
    '1014A80': ([212, 535, 522], 212, 'near'),   # 홍천A3
    '1014A90': ([212, 535, 522], 212, 'near'),   # 홍천A4
    '1012A40': ([211], 211, 'far'),              # 소양B (소양댐 하류)
    '1013A60': ([212], 212, 'far'),              # 북한C
    '1013A90': ([212], 212, 'far'),              # 가평A
    '1015A10': ([212], 212, 'far'),              # 북한D
    '1015A30': ([212], 212, 'far'),              # 조종A
    '1015A50': ([212], 212, 'far'),              # (청평댐) 1015A50
}
RCOLS = ['P0', 'Pprev', 'P3prior', 'P7prior', 'API7', 'ADD_prior', 'P2', 'Pmax3', 'P14prior', 'P30prior', 'msin', 'mcos']
SHARED = RCOLS + ['Pev', 'TA', 'TA7', 'TAmin', 'lQrel', 'Qpct']


def rain_feats(P, tx, tn):
    b = model.build(pd.DataFrame({'P0': P, 'Q': np.nan}, index=P.index))[RCOLS]
    wet = P.fillna(0) >= 1; b['Pev'] = P.fillna(0).where(wet, 0).groupby((~wet).cumsum()).cumsum()
    return b.join(model.temp_feats(pd.DataFrame({'TAMAX': tx, 'TAMIN': tn}, index=P.index)))


def rel_q(lq, ref):
    """지점 유량 분포(ref: 그 지점 전체 실측일 log Q) 기준 상대값·백분위."""
    med = np.nanmedian(ref); srt = np.sort(ref[np.isfinite(ref)])
    return lq - med, np.searchsorted(srt, lq) / len(srt)


def source_table():
    if os.path.exists('src.pkl'): return pd.read_pickle('src.pkl')
    rows = []
    for code, (st, tst, grp) in SITES.items():
        P = RN[st].reindex(IDX).mean(axis=1)
        b = rain_feats(P, TX[tst].reindex(IDX), TN[tst].reindex(IDX))
        s = SL[SL.code == code].dropna(subset=['SS']).drop_duplicates('date').set_index('date')
        s = s[(s.SS > 0)]; q = pd.to_numeric(s['유량'], errors='coerce'); s = s[q > 0]; q = q[q > 0]
        X = b.reindex(s.index).copy(); lq = np.log(q.values)
        X['lQrel'], X['Qpct'] = rel_q(lq, lq)
        ly = np.log(s.SS.values); X['yrel'] = ly - np.median(ly); X['ly'] = ly
        X['site'] = code; X['grp'] = grp; rows.append(X)
    D = pd.concat(rows); D.to_pickle('src.pkl'); return D


def target_shared(df, F, qc):
    """내린천2 날짜 전체의 공통 변수(강우는 운영 Thiessen, 유량은 운영 보정유량)."""
    T = F[[c for c in SHARED if c in F.columns]].copy()
    ref = np.log(qc[obs_mask(df)].values)
    T['lQrel'], T['Qpct'] = rel_q(np.log(qc.values), ref)
    return T[SHARED]
