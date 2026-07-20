#!/usr/bin/env bash
set -e
python scripts/train_all.py
uvicorn app.api:app --reload --port 8000 &
API_PID=$!
streamlit run app/streamlit_app.py
kill $API_PID
