"""물모아 개편(2026-10) 후 총량측정망 소양A(1012A20, 내린천2) 2026년 일자료 재수집.

요청 경로는 api_request_2026_redesign.md 참고. 결과는 이 폴더에만 저장한다.
사용법:  python -I fetch_naerincheon2_2026.py
"""
import truststore

truststore.inject_into_ssl()

import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent
BASE = "https://www.mulmoa.go.kr"
PAGE = BASE + "/web/totalSearch"
CODE = "1012A20"
DELAY = 3.5
MAX_TIMEOUTS = 2

s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152.0 Safari/537.36"
timeouts = 0
token = None


def get_page():
    global token
    r = call("GET", PAGE)
    token = re.search(r'name="_csrf" content="([^"]+)"', r.text).group(1)


def call(method, url, data=None):
    """요청 사이 DELAY초 대기, 시간초과 2회면 중단."""
    global timeouts
    time.sleep(DELAY)
    h = {"Referer": PAGE}
    if method == "POST":
        h.update({"X-CSRF-TOKEN": token, "X-Requested-With": "XMLHttpRequest"})
    try:
        r = s.request(method, url, data=data, headers=h, timeout=90)
    except (requests.Timeout, requests.ConnectionError) as e:
        timeouts += 1
        print(f"시간초과/접속오류 {timeouts}/{MAX_TIMEOUTS}: {url} {e}")
        if timeouts >= MAX_TIMEOUTS:
            sys.exit("시간초과 2회: 중단")
        return call(method, url, data)
    r.raise_for_status()
    return r


def post_json(path, data):
    r = call("POST", BASE + path, data)
    try:
        return r, r.json()
    except ValueError:
        raise RuntimeError(f"JSON 아님 {path}: {r.text[:300]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    get_page()

    # 메뉴: 수질(top 1) > 총량측정망
    _, groups = post_json("/web/totalSearch/getNewDataCodeListJson", {"top_data_code": "1"})
    menu = None
    for g in groups["list"]:
        if g["CD_NM"] == "총량측정망":
            menu = g
        elif int(g.get("CNT") or 0) > 0 and menu is None:
            _, sub = post_json("/web/totalSearch/dataCodeListJson", {"upNo": g["CD_NO"]})
            menu = next((d for d in sub["list"] if d["CD_NM"] == "총량측정망"), None)
    if menu is None:
        raise RuntimeError(f"총량측정망 메뉴 못 찾음: {[g['CD_NM'] for g in groups['list']]}")
    print("메뉴:", {k: menu.get(k) for k in ("CD_NO", "CD_NM", "TAB_SN", "MAX_YMD", "MAX_YYYY", "ZONE_CHC_NM", "INQ_PRD_SE_NM")})

    # 화면이 하는 순서 그대로: 탭 목록 -> 검색폼 (세션 상태 설정 가능성)
    post_json("/web/totalSearch/tabsInfoListJson", {"gubun": menu["CD_NO"]})
    call("POST", PAGE + "/searchForm", {"gubun": menu["CD_NO"], "data_code": menu["CD_NO"], "tbmn_sn": menu["TAB_SN"],
                                        "maxYmd": menu["MAX_YMD"], "initSYY": menu["MAX_YYYY"], "_csrf": token})

    # MAX_YMD(메뉴 최신월)는 늦게 갱신될 수 있어 연말까지 요청
    end_ym = "202612"
    params = {"data_code": menu["CD_NO"], "tbmn_sn": menu["TAB_SN"], "pLgubun": "02",
              "STATION_CODES": json.dumps([CODE]), "pDateGun": "1",
              "pSYY": "2026", "pEYY": end_ym[:4], "pSMM": "01", "pEMM": end_ym[4:],
              "pSDate": "202601", "pEDate": end_ym, "excel": "N"}
    r, data = post_json("/web/totalSearch/water/totAmount/mesureDataCall", params)
    (OUT / "tmdl_soyangA_1012A20_2026_mesureDataCall.json").write_bytes(r.content)
    (OUT / "request_used.json").write_text(json.dumps({"menu": menu, "params": params}, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = data["list"]
    print("행 수:", len(rows), "열:", list(rows[0].keys()) if rows else None)
    names = {c["COL_NM"]: c["SCRN_EXPRS_NM"] for c in data["columns"]}
    ss = next(k for k, v in names.items() if v.startswith("부유물질"))
    q = next(k for k, v in names.items() if v.startswith("유량"))
    print("열 이름:", names)
    with open(OUT / "naerincheon2_2026_latest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "ss_mgL", "flow_m3s"])
        for row in sorted(rows, key=lambda x: x["INSP_YMD"]):
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["INSP_YMD"]) and row["WQMN_CD"] == CODE
            w.writerow([row["INSP_YMD"], row[ss] or "", row[q] or ""])


if __name__ == "__main__":
    main()
