"""자료원 접속 점검: 각 API에 작은 요청을 보내 상태코드와 응답 앞부분을 출력한다(인증키는 가린다)."""
import datetime as dt, json, os, re, time
import requests

KEY = os.environ.get("KMA_API_KEY", "")
now = dt.datetime.utcnow() + dt.timedelta(hours=9)
ymd = (now - dt.timedelta(days=1)).strftime("%Y%m%d")
today = now.strftime("%Y%m%d")
K = "https://apihub.kma.go.kr"
TESTS = {
    "kma_asos_day_211": f"{K}/api/typ01/url/kma_sfcdd.php?tm={ymd}&stn=211&help=1&authKey={KEY}",
    "kma_aws_day_557": f"{K}/api/typ01/url/sfc_aws_day.php?tm2={ymd}&stn=557&obs=rn_day&disp=0&help=1&authKey={KEY}",
    "kma_aws_min_557": f"{K}/api/typ01/cgi-bin/url/nph-aws2_min?tm2={ymd}2350&stn=557&disp=0&help=1&authKey={KEY}",
    "kma_vilage_fcst_83_136": f"{K}/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst?pageNo=1&numOfRows=50&dataType=JSON&base_date={today}&base_time=0500&nx=83&ny=136&authKey={KEY}",
    "kma_wrn_now": f"{K}/api/typ01/url/wrn_now_data.php?fe=f&disp=0&help=1&authKey={KEY}",
    "kma_wrn_met_history": f"{K}/api/typ01/url/wrn_met_data.php?wrn=A&reg=0&tmfc1={ymd[:6]}010000&tmfc2={ymd}2300&disp=0&help=1&authKey={KEY}",
    "wamis_flw_dtdata_1012650": "http://www.wamis.go.kr:8080/wamis/openapi/wkw/flw_dtdata?obscd=1012650&year=2026&output=json",
    "wamis_wl_dtdata_1012650": "http://www.wamis.go.kr:8080/wamis/openapi/wkw/wl_dtdata?obscd=1012650&startdt=20261001&enddt=20261029&output=json",
    "hrfco_root": "https://api.hrfco.go.kr/",
    "mulmoa_page": "https://www.mulmoa.go.kr/web/totalSearch",
}
for name, url in TESTS.items():
    for verify in (True,):
        try:
            r = requests.get(url, timeout=40, verify=verify, headers={"User-Agent": "Mozilla/5.0"})
            body = r.content[:900].decode(r.encoding or "utf-8", "replace")
            print(f"== {name} status={r.status_code} len={len(r.content)} ctype={r.headers.get('content-type')}")
        except Exception as e:
            body = ""
            print(f"== {name} ERROR {type(e).__name__}: {str(e)[:300]}")
        if KEY: body = body.replace(KEY, "***")
        print(body.replace("\r", ""))
    time.sleep(1.5)
