"""자료 수집: WAMIS 원대교 유량·수위, 기상청 AWS 일강수·단기예보·특보, 물모아 총량측정망 SS.

모두 data/live/ 아래 CSV/JSON에 날짜 기준으로 합쳐 저장한다(같은 날짜는 새 값으로 덮어씀).
기상청 키는 환경변수 KMA_API_KEY 또는 저장소 상위 폴더들의 .env / .env.txt 에서 읽는다.
"""
try:
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass
import datetime as dt
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / 'data' / 'live'
LIVE.mkdir(parents=True, exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

WAMIS = 'http://www.wamis.go.kr:8080/wamis/openapi/wkw'
WONDAE = '1012650'                       # 인제군(원대교) 수위·유량 관측소
DAR = 1.0414                             # 내린천2 / 원대교 유역면적비
KMA = 'https://apihub.kma.go.kr'
AWS = {321: 0.0487, 557: 0.5113, 559: 0.4400}           # 원통·기린·내면 Thiessen 가중치
GRID = {321: (80, 139), 557: (83, 136), 559: (84, 132)}  # 단기예보 격자 (각 AWS 위치)
MULMOA = 'https://www.mulmoa.go.kr'
STATION = '1012A20'                      # 총량측정망 소양A (내린천2)


def kma_key():
    if os.environ.get('KMA_API_KEY'):
        return os.environ['KMA_API_KEY'].strip()
    for d in [ROOT, *ROOT.parents][:4]:
        for name in ('.env', '.env.txt'):
            p = d / name
            if not p.exists(): continue
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    if k.strip() == 'KMA_API_KEY': return v.strip().strip('"')
                else:
                    return line
    return ''


def get(url, **kw):
    for i in range(3):
        try:
            r = requests.get(url, timeout=60, headers=UA, **kw); r.raise_for_status(); return r
        except requests.RequestException:
            if i == 2: raise
            time.sleep(5 * (i + 1))


def merge(path, new, key='date'):
    """기존 CSV와 합쳐 날짜 순으로 저장. 같은 키는 새 값 우선."""
    if path.exists():
        old = pd.read_csv(path, encoding='utf-8-sig', dtype={key: str})
        new = pd.concat([old, new]).drop_duplicates(key, keep='last')
    new = new.sort_values(key)
    new.to_csv(path, index=False, encoding='utf-8-sig')
    return new


# ---------- WAMIS ----------
def wamis_flow(years):
    rows = []
    for y in years:
        j = get(f'{WAMIS}/flw_dtdata?obscd={WONDAE}&year={y}&output=json').json()
        for r in j.get('list', []):
            v = pd.to_numeric(r.get('fw'), errors='coerce')
            rows.append({'date': f"{r['ymd'][:4]}-{r['ymd'][4:6]}-{r['ymd'][6:]}", 'q_wondae': v})
    return merge(LIVE / 'wamis_flow.csv', pd.DataFrame(rows))


def wamis_stage(start, end):
    """시간 수위를 한 달씩 받아 일 평균·최저·최고로 요약."""
    rows = []
    s = start
    while s <= end:
        e = min(end, (s.replace(day=1) + pd.offsets.MonthEnd(0)).date())
        j = get(f'{WAMIS}/wl_hrdata?obscd={WONDAE}&startdt={s:%Y%m%d}&enddt={e:%Y%m%d}&output=json').json()
        for r in j.get('list', []):
            rows.append({'ymdh': r['ymdh'], 'wl': pd.to_numeric(r.get('wl'), errors='coerce')})
        s = e + dt.timedelta(days=1); time.sleep(1)
    if not rows: return None
    h = pd.DataFrame(rows).dropna()
    h = h[(h.wl > 0.5) & (h.wl < 10)]
    # ymdh의 24시는 그날 자정(다음날 00시) 값이라 그날 자료로 둔다
    h['date'] = h.ymdh.str[:4] + '-' + h.ymdh.str[4:6] + '-' + h.ymdh.str[6:8]
    g = h.groupby('date').wl
    d = pd.DataFrame({'Havg': g.mean().round(3), 'Hmin': g.min(), 'Hmax': g.max(), 'n_hours': g.size()}).reset_index()
    return merge(LIVE / 'wamis_stage_daily.csv', d)


# ---------- 기상청 ----------
def kma_aws_day(days, key):
    """AWS 일강수(rn_day). 하루 한 번 전체 지점 요청 후 세 지점만 남긴다."""
    rows = []
    for d in days:
        t = get(f'{KMA}/api/typ01/url/sfc_aws_day.php?tm2={d:%Y%m%d}&obs=rn_day&stn=0&disp=0&help=0&authKey={key}')
        t = t.content.decode('euc-kr', 'replace')
        for line in t.splitlines():
            p = line.split()
            if len(p) >= 6 and p[0].isdigit() and int(p[1]) in AWS:
                v = float(p[5]); rows.append({'date': f'{p[0][:4]}-{p[0][4:6]}-{p[0][6:8]}', 'stn': int(p[1]), 'rn': v if v >= 0 else None})
        time.sleep(0.5)
    if not rows: return None
    w = pd.DataFrame(rows).pivot_table(index='date', columns='stn', values='rn').reset_index()
    w.columns = ['date'] + [f'rn_{c}' for c in w.columns[1:]]
    return merge(LIVE / 'kma_aws_rain.csv', w)


def pcp_mm(s):
    """단기예보 PCP 문자열 → mm. '강수없음'=0, '1mm 미만'=0.5, '30.0~50.0mm'=40, '50.0mm 이상'=50."""
    s = str(s)
    if '없음' in s: return 0.0
    if '미만' in s: return 0.5
    n = [float(x) for x in re.findall(r'[\d.]+', s)]
    return sum(n) / len(n) if n else 0.0


def env_key(name):
    """KMA 외 키(DATA_GO_KR_KEY 등)는 환경변수 또는 .env 의 NAME=값 줄에서."""
    if os.environ.get(name): return os.environ[name].strip()
    for d in [ROOT, *ROOT.parents][:4]:
        for fn in ('.env', '.env.txt'):
            p = d / fn
            if p.exists():
                for line in p.read_text(encoding='utf-8-sig').splitlines():
                    if line.strip().startswith(name + '='): return line.split('=', 1)[1].strip().strip('"')
    return ''


def kma_fcst(issue, key):
    """기준일 D 23시 발표 단기예보로 D+1.. 일강수 예보(Thiessen 가중).
    기상청 API허브 키를 먼저 쓰고, 공공데이터포털 키(DATA_GO_KR_KEY)가 있으면 그쪽으로도 받을 수 있다."""
    dk = env_key('DATA_GO_KR_KEY'); out = {}
    for stn, (nx, ny) in GRID.items():
        q = f'pageNo=1&numOfRows=1000&dataType=JSON&base_date={issue:%Y%m%d}&base_time=2300&nx={nx}&ny={ny}'
        url = (f'https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst?serviceKey={dk}&{q}' if dk else
               f'{KMA}/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst?{q}&authKey={key}')
        j = get(url).json()
        items = j['response']['body']['items']['item']
        day = {}
        for it in items:
            if it['category'] == 'PCP':
                day[it['fcstDate']] = day.get(it['fcstDate'], 0) + pcp_mm(it['fcstValue'])
        out[stn] = day
    dates = sorted(set().union(*[d.keys() for d in out.values()]))
    rain = {f'{x[:4]}-{x[4:6]}-{x[6:]}': round(sum(AWS[s] * out[s].get(x, 0) for s in AWS), 1) for x in dates}
    res = {'issue': str(issue), 'base': f'{issue:%Y-%m-%d} 23:00', 'rain_mm': rain}
    (LIVE / 'kma_fcst.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    return res


def kma_warnings(key, today):
    """특보 현황·올해 이력 중 인제 관련 줄만 원문으로 보관."""
    res = {'fetched': str(today)}
    for nm, url in {'now': f'{KMA}/api/typ01/url/wrn_now_data.php?fe=f&disp=0&help=1&authKey={key}',
                    'hist': f'{KMA}/api/typ01/url/wrn_met_data.php?wrn=A&reg=0&tmfc1={today:%Y}01010000'
                            f'&tmfc2={today:%Y%m%d}2300&disp=0&help=1&authKey={key}'}.items():
        t = get(url).content.decode('euc-kr', 'replace')
        head = [l for l in t.splitlines() if l.startswith('#')][-3:]
        res[nm] = {'header': head, 'rows': [l for l in t.splitlines() if not l.startswith('#') and '인제' in l]}
    (LIVE / 'kma_warnings.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    return res


# ---------- 물모아 ----------
def mulmoa_ss(year):
    """물모아 분야별 상세검색 > 총량측정망 일자료 (2026-10 개편 후 경로)."""
    s = requests.Session(); s.headers.update(UA)
    page = MULMOA + '/web/totalSearch'
    tok = re.search(r'name="_csrf" content="([^"]+)"', s.get(page, timeout=90).text).group(1)
    h = {'Referer': page, 'X-CSRF-TOKEN': tok, 'X-Requested-With': 'XMLHttpRequest'}

    def post(path, data):
        time.sleep(3.5); r = s.post(MULMOA + path, data=data, headers=h, timeout=90); r.raise_for_status(); return r

    groups = post('/web/totalSearch/getNewDataCodeListJson', {'top_data_code': '1'}).json()['list']
    menu = next((g for g in groups if g['CD_NM'] == '총량측정망'), None)
    if menu is None:
        for g in groups:
            if int(g.get('CNT') or 0) > 0:
                sub = post('/web/totalSearch/dataCodeListJson', {'upNo': g['CD_NO']}).json()['list']
                menu = next((d for d in sub if d['CD_NM'] == '총량측정망'), None)
                if menu: break
    post('/web/totalSearch/tabsInfoListJson', {'gubun': menu['CD_NO']})
    post('/web/totalSearch/searchForm', {'gubun': menu['CD_NO'], 'data_code': menu['CD_NO'], 'tbmn_sn': menu['TAB_SN'],
                                         'maxYmd': menu['MAX_YMD'], 'initSYY': menu['MAX_YYYY'], '_csrf': tok})
    p = {'data_code': menu['CD_NO'], 'tbmn_sn': menu['TAB_SN'], 'pLgubun': '02', 'STATION_CODES': json.dumps([STATION]),
         'pDateGun': '1', 'pSYY': str(year), 'pEYY': str(year), 'pSMM': '01', 'pEMM': '12',
         'pSDate': f'{year}01', 'pEDate': f'{year}12', 'excel': 'N'}
    j = post('/web/totalSearch/water/totAmount/mesureDataCall', p).json()
    names = {c['COL_NM']: c['SCRN_EXPRS_NM'] for c in j['columns']}
    ss = next(k for k, v in names.items() if v.startswith('부유물질'))
    q = next(k for k, v in names.items() if v.startswith('유량'))
    rows = [{'date': r['INSP_YMD'], 'ss': pd.to_numeric(r[ss], errors='coerce'), 'q_meas': pd.to_numeric(r[q], errors='coerce')}
            for r in j['list'] if r.get('WQMN_CD') == STATION]
    if not rows: return None
    return merge(LIVE / 'mulmoa_ss.csv', pd.DataFrame(rows))
