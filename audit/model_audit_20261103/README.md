# 내린천2 SS 예측 모델 종합 기술 감사 (2026-11-03)

보고서는 6개입니다.

| 파일 | 내용 |
|---|---|
| 01_종합기술감사.md | 시스템 구조, 과제 A–N 진단표, 최종 10문 10답 |
| 02_통계및시계열검증.md | 지표 정의, 재현, 누수 점검표(C1–C11), 실제 강우 대입 vs 실제예보, 구간, 로그/원척도 |
| 03_수문학적타당성검토.md | 면적비, 수위–유량, 지연, 사례(2023-06-27, 2021-03-03), 강우 결측, 예보 범위 |
| 04_개선실험계획.md | 공통 평가체계, 후보별 가설·실험·도입 기준 |
| 05_모델비교결과.md | 실제 수행 실험의 성능표와 부트스트랩 |
| 06_개발우선순위.md | P0·P1·P2 로드맵 |

## 재현
```
pip install xgboost scikit-learn pandas numpy
export REPO=<naerincheon-ss-dashboard 경로>     # main 8151c7e 기준, pipeline/ 를 그대로 import
cd scripts
python r0_repro.py      # 일별 전진검증 재현 (~1분)
python r1_fc.py         # 1·3·7일 hindcast 재현 + 실제 예보 보관본 (~5분)
python r3_daily.py      # 앙상블·혼합비·보정 (~2분)
python r4_variants.py   # M0–M5, 유량보정, 수위–유량 as-of, 고농도 대안 (~9분)
python r5_interval.py   # 80% 구간 (r0, r1 결과 필요)
python r6_hydro.py      # 수문 점검 (r0 결과 필요)
python r7_asof.py       # 공개 지연 as-of (~35분)
python r7b_same_days.py # as-of vs 연도별 같은 날짜 비교 (r7 결과 필요)
python r8_bins.py       # 구간별·연도별 (r3 결과 필요)
```
- XGB 스레드가 겹치면 매우 느려지므로 하나씩 실행하세요.
- 예보 보관본 경로: `capstone/ss_short_term_forecast_20261102/data/raw/`
- 운영 파일(main, docs, data/base, data/live)은 읽기만 했습니다.
