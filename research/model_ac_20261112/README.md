# 모델 A+C 연구 코드 (2026-11-12, 운영 미반영)

- **C (일별 추정, 전이 미세조정)**: 가까운 6곳 실측 SS + 원주·경안천 탁도 보간 SS = 원천 7,312행(`source_nearbig_7312.csv`, 공통 18변수 + `yrel`=log SS−지점 중앙값, `grp` near/turb)으로 XGB 300그루×5 사전학습 → 내린천2 42변수(운영 40 + lQrel·Qpct)로 300그루 미세조정 → Ridge 반반 + 중앙값 보정 + 홍수일 보정 ¾. 코드: `scripts/t2.py`의 `FineTune`(GRP=`nearbig`), 공통변수·상대유량 `scripts/tl.py`.
- **A (두 단계 + 아침 07시)**: `scripts/d4_int_morn.py` 앞부분(1단계 XGB 25개, 아침 변수 H07·dH07·Hmx07·R07·Rmx07, 인제 ASOS 211 시간강우).
- **A+C**: `scripts/d6_ac_morn.py` (1단계 예측 한 번 → 2단계 운영/C/고농도만 섞기 교체). 결과 `d6_scores.csv`.
- 실행 위치: tl.py가 `src.pkl`을 찾음 → `source_nearbig_7312.csv`를 읽어 `src.pkl`로 저장하거나 `source_table()`을 바꿔 사용. 평가틀 `fw.py`는 `df.pkl`·`F.pkl`(운영 `rd.assemble()`/`model.features` 결과)을 씀.
- 주의: XGB를 두 프로세스로 동시에 돌리면 스레드 경합으로 매우 느려짐. 한 번에 하나씩.
