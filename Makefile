install:
	python -m pip install -r requirements.txt

train:
	python scripts/train_all.py

test:
	python -m pytest -q

api:
	uvicorn app.api:app --reload --port 8000

ui:
	streamlit run app/streamlit_app.py

compose:
	docker compose up --build
