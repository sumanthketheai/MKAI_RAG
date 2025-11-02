
@echo off
echo ==============================
echo Starting MK AI Bot RCA Assistant
echo ==============================
call .venv\Scripts\activate
start "" "C:\Program Files\Ollama\ollama.exe" serve
streamlit run app.py
pause
