# Campus Chat: Training to Production

**AAIN-7420 Data Engineering for AI — Chapter 7 Complete Example**

This project teaches students how to build a small AI chat application from raw data to a production-style local deployment. It is intentionally lightweight and requires **no paid API, no GPU, and no prior AI experience**.

## What the app does

Campus Chat combines two forms of AI:

1. **A trained intent classifier** using TF-IDF and Logistic Regression.
2. **A retrieval pipeline** that searches a governed campus-policy corpus.

The application then returns a grounded answer with source, owner, version, and effective-date information. Every request produces an evidence trace.

## Chapter 7 concepts implemented

- reproducible training pipeline;
- data and corpus snapshots;
- model, prompt, and index registries;
- benchmark lineage;
- release gate;
- deployment manifest;
- serving API and chat UI;
- monitoring/evidence traces;
- feedback capture design;
- rollback-ready application state.

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Rebuild the training and retrieval artifacts

```bash
python scripts/train_all.py
```

### 4. Start the API

```bash
uvicorn app.api:app --reload --port 8000
```

Open the API documentation at `http://localhost:8000/docs`.

### 5. Start the chat UI in a second terminal

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`.

## Docker option

```bash
docker compose up --build
```

## Project structure

```text
AAIN7420_CampusChat_Training_to_Production/
├── app/                    # FastAPI and Streamlit application
├── artifacts/              # Prebuilt model, retriever, registries, and manifests
├── data/                   # Synthetic training, corpus, benchmark, and feedback data
├── docs/                   # Architecture and teaching documentation
├── evidence/               # Request traces created during inference
├── notebooks/              # Detailed Colab/Jupyter walkthrough
├── scripts/                # One-command training pipeline
├── src/                    # Data, training, retrieval, inference, and evaluation code
├── tests/                  # Automated tests
├── .github/workflows/      # CI example
├── docker-compose.yml
├── requirements.txt
└── TUTORIAL.md
```

## Safe-use note

All data is fictional and synthetic. The app is a teaching demonstration, not an official source of university policy. It must not be used for real student decisions without institutional review, authoritative data, security controls, accessibility work, privacy review, and operational ownership.
