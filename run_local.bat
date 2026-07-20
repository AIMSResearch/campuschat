@echo off
python scripts	rain_all.py
start cmd /k uvicorn app.api:app --reload --port 8000
streamlit run app\streamlit_app.py
