"""한 번만 도는 자료 조사 (2026-11-10, 전이학습 후보 지점 찾기).

클라우드에서 WAMIS·물모아가 막혀 있어 PC 매일 실행(run_daily --push) 끝에 따로 프로세스로 돌린다.
결과는 data/survey/ 에 저장되어 함께 push 된다. 작업마다 완료 표시 파일이 있으면 다시 돌지 않는다.
어떤 오류가 나도 매일 갱신에는 영향이 없다(run_daily에서 시간 제한·오류 무시).
"""
import json, re, sys, time, traceback
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'survey'
OUT.mkdir(parents=True, exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
WAMIS = 'http://www.wamis.go.kr:8080/wamis/openapi/wkw'
MULMOA = 'https://www.mulmoa.go.kr'
KEYS = ('홍천', '인제', '가평', '화천', '춘천', '양구', '평창', '횡성')   # 관측소명 지역어


def save(name, obj):
    p = OUT / name
    if isinstance(obj, (dict, list)):
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    else:
        p.write_text(str(obj), encoding='utf-8')


def task(name):
    def deco(f):
        def run():
            flag = OUT / f'{name}.done'
            if flag.exists(): return
            try:
                f(); flag.write_text(time.strftime('%Y-%m-%d %H:%M'), encoding='utf-8')
                print('survey', name, 'ok')
            except Exception:
                save(f'{name}.error.txt', traceback.format_exc()); print('survey', name, 'error')
        return run
    return deco


def wget(ep, **p):
    r = requests.get(f'{WAMIS}/{ep}', params={**p, 'output': 'json'}, headers=UA, timeout=60)
    r.raise_for_status(); time.sleep(0.5); return r.json()


@task('wamis_wl')
def wamis_wl():
    """한강 수위관측소 전체 목록 + 지역어 관측소의 좌표·시간수위 표본 연도."""
    full = wget('wl_dubwlobs', basin='1')
    save('wamis_wl_list_raw.json', full)
    sel = [o for o in full.get('list', []) if any(k in str(o.get('obsnm', '')) for k in KEYS)]
    rows = []
    for o in sel:
        row = dict(o)
        try: row['info'] = (wget('wl_obsinfo', obscd=o['obscd']).get('list') or [{}])[0]
        except Exception as e: row['info_error'] = str(e)
        yrs = []
        for y in (2014, 2016, 2019, 2022, 2024, 2026):
            try:
                if wget('wl_hrdata', obscd=o['obscd'], startdt=f'{y}0701', enddt=f'{y}0702').get('list'): yrs.append(y)
            except Exception: pass
        row['hourly_years'] = yrs
        rows.append(row)
    save('wamis_wl_candidates.json', rows)


@task('mulmoa_menus')
def mulmoa_menus():
    """물모아 분야별 상세검색 메뉴 구조와 자동측정망·총량·수질측정망 검색 화면(지점 목록 포함)을 그대로 저장."""
    s = requests.Session(); s.headers.update(UA)
    page = MULMOA + '/web/totalSearch'
    tok = re.search(r'name="_csrf" content="([^"]+)"', s.get(page, timeout=90).text).group(1)
    h = {'Referer': page, 'X-CSRF-TOKEN': tok, 'X-Requested-With': 'XMLHttpRequest'}

    def post(path, data):
        time.sleep(3.5); r = s.post(MULMOA + path, data=data, headers=h, timeout=90); r.raise_for_status(); return r

    tree = {}
    for top in ('1', '2', '3'):
        try: groups = post('/web/totalSearch/getNewDataCodeListJson', {'top_data_code': top}).json().get('list', [])
        except Exception as e: tree[top] = {'error': str(e)}; continue
        tree[top] = {'groups': groups, 'sub': {}}
        for g in groups:
            try: tree[top]['sub'][g['CD_NO']] = post('/web/totalSearch/dataCodeListJson', {'upNo': g['CD_NO']}).json().get('list', [])
            except Exception as e: tree[top]['sub'][g['CD_NO']] = {'error': str(e)}
    save('mulmoa_menu_tree.json', tree)
    menus = []
    for t in tree.values():
        for m in list(t.get('groups', [])) + [d for v in t.get('sub', {}).values() if isinstance(v, list) for d in v]:
            if any(k in str(m.get('CD_NM', '')) for k in ('자동', '총량', '수질측정망')): menus.append(m)
    for m in menus:
        tag = re.sub(r'\W', '_', f"{m.get('CD_NM')}_{m.get('CD_NO')}")
        try:
            tabs = post('/web/totalSearch/tabsInfoListJson', {'gubun': m['CD_NO']}).text
            save(f'mulmoa_tabs_{tag}.json', tabs)
            form = post('/web/totalSearch/searchForm', {'gubun': m['CD_NO'], 'data_code': m['CD_NO'], 'tbmn_sn': m.get('TAB_SN', ''),
                                                       'maxYmd': m.get('MAX_YMD', ''), 'initSYY': m.get('MAX_YYYY', ''), '_csrf': tok}).text
            save(f'mulmoa_form_{tag}.html', form)
        except Exception as e:
            save(f'mulmoa_form_{tag}.error.txt', str(e))


if __name__ == '__main__':
    for f in (wamis_wl, mulmoa_menus):
        f()
    sys.exit(0)
