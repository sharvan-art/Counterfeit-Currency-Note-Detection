@echo off
title CurrencyShield AI - Counterfeit Currency Detection
echo ========================================================
echo Starting CurrencyShield AI Web Application...
echo ========================================================
cd /d "%~dp0"
python -m streamlit run webapp\frontend\app.py --server.port 8501
pause
