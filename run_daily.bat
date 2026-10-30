@echo off
rem 매일 자동 갱신: 최신 코드 받기 -> 수집·추정·예측·대시보드 생성 -> GitHub에 올리기
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist logs mkdir logs
git pull --rebase --autostash >> logs\run_daily.log 2>&1
python pipeline\run_daily.py --push >> logs\run_daily.log 2>&1
