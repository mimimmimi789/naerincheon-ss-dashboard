# 내린천2 SS 대시보드

총량측정망 내린천2(소양A, 1012A20) 부유물질(SS)의 일별 추정과 1·3·5일 예측 대시보드. 캡스톤디자인2.

대시보드: `docs/index.html` (GitHub Pages로 공개)

## 매일 자동 갱신

PC의 Windows 작업 스케줄러가 매일 07:30에 `run_daily.bat`을 실행한다.
WAMIS는 해외(GitHub Actions)에서 접속되지 않아 PC에서 돌린다.

1. 자료 수집 (`pipeline/sources.py`, 결과는 `data/live/`)
   - WAMIS 원대교(1012650) 일유량, 시간수위 → 일 평균·최저·최고
   - 기상청 API허브 AWS 일강수 321 원통·557 기린·559 내면 → Thiessen 가중(0.0487/0.5113/0.4400)
   - 기상청 단기예보(D 23시 발표) 강수량 → D+1.. 일강수 예보
   - 기상청 특보 현황·올해 이력 (인제 관련 줄)
   - 물모아 총량측정망 SS·유량 실측
2. 입력 조립: `data/base/base_inputs_20140122_20260826.csv` 뒤에 수집 자료를 붙임 (유량 = 원대교 × 1.0414)
3. SS 추정 (`pipeline/model.py`): XGB(log SS) + 확장변수 + 수위 기반 유량보정 + 첨두수위. 매일 전체 실측으로 재학습.
4. 1·3·5일 예측 (`pipeline/forecast.py`): 2019년 이후 기준일별 표본외 예측과 마지막 기준일 운영 예측(기상청 예보강우, 0–120 mm 시나리오).
5. 잠금 평가: 2026-08-26까지 실측으로만 학습한 모델로 그 뒤 새 실측을 맞춘 성적.
6. 운영 예측 기록: 그날 낸 1·3·5일 예측(관측만·기상청 예보 반영)을 `docs/forecast_log.csv`에 쌓고, 목표일 실측이 생기면 실제 예보 성적(MAE, 월별 기준값 대비)을 매긴다.
7. `docs/index.html`, `docs/ss_daily.csv`, `docs/forecast_daily.csv`, `docs/forecast_log.csv`, `docs/latest.json` 생성 후 git push.

## PC 설정

```
pip install -r requirements.txt
```

기상청 키는 저장소 폴더 또는 그 상위 폴더의 `.env`(또는 `.env.txt`)에 `KMA_API_KEY=키값`으로 둔다. 공공데이터포털 단기예보 키를 쓰려면 `DATA_GO_KR_KEY=키값`을 추가한다. 이 파일들은 커밋하지 않는다(`.gitignore`).

작업 등록 (PowerShell에서 한 번):

```
powershell -ExecutionPolicy Bypass -File register_task.ps1
```

수동 실행: `python pipeline/run_daily.py` (수집 없이 다시 그리기만: `--no-fetch`, 올리기까지: `--push`). 로그는 `logs/run_daily.log`.

## 참고

- 모델 선택과 예측 성능표는 2026-10-30 분석(전진검증 2019–26)에서 고정했다(`data/base/forecast_static.json`).
- 예보강우 모델은 학습 때 실제 내린 비를 예보 자리에 넣었다(사후 계산). 예보가 틀리는 오차는 반영되지 않는다.
- 2026-11-03 감사(`audit/model_audit_20261103/`): 과거 기상청 발표 예보로 다시 계산한 성능을 `forecast_static.json`의 `tbl_real`에 두고 대시보드에 함께 표시한다. 기상청 예보가 예측 기간(h일)을 다 채우지 못하면(끝날 일부 시간만 있는 날 포함) 예보 반영값을 내지 않는다.
- 기본 입력의 2026년 최저수위는 수집된 WAMIS 시간수위로 채운다. 그래서 2026년 추정값은 2026-10-30 고정본과 조금 다를 수 있다.

## 홈 화면 추가 · 휴대폰 알림 (2026-11-06)
- 대시보드의 **홈 화면에 추가**(PC는 바탕화면에 추가) 버튼: 가능하면 브라우저 설치 창, 아니면 기기별 안내(아이폰·삼성·Chrome·Edge·카카오톡). 파일: `docs/manifest.webmanifest`, `docs/icons/`, `docs/sw.js`.
- **알림 받기**: 누르면 그 자리에서 오늘 요약 알림이 뜨고, 등록 정보가 암호화되어 ntfy.sh 우편함으로 갑니다. GitHub Actions(`.github/workflows/notify.yml`, `pipeline/notify.py`)가 10분마다 새 등록을 확인해 '등록 완료' 알림을 보내고, PC가 매일 갱신을 올리면(docs/latest.json 변경) 모든 기기에 1·3·5일 요약을 보냅니다. 알림 발송에는 PC가 필요 없습니다.
- 처음 한 번: 저장소 Settings → Secrets and variables → Actions → New repository secret, 이름 `PUSH_SECRET`, 값은 긴 임의 문자열. 그다음 Actions → 알림 보내기 → Run workflow(poll)로 `docs/push_key.json`을 만듭니다.
- 아이폰은 iOS 16.4 이상에서 홈 화면에 추가한 앱으로 열어야 알림을 받을 수 있습니다. 안드로이드(삼성 인터넷·Chrome)는 브라우저에서도 됩니다.
