from pathlib import Path

from src.data_pipeline import validate_corpus
from src.inference import CampusChatEngine
import pandas as pd


def test_invalid_documents_are_quarantined():
    project_root = Path(__file__).resolve().parents[1]
    raw = pd.read_csv(project_root / "data/raw/campus_documents.csv").fillna("")
    approved, quarantine = validate_corpus(raw)
    assert len(approved) > 0
    assert len(quarantine) >= 3
    assert "DOC-BAD-002" in set(quarantine["document_id"])


def test_chat_returns_trace():
    engine = CampusChatEngine()
    result = engine.chat("How do I reset my password?")
    assert result.predicted_intent == "technical_support"
    assert result.sources
    assert Path(result.trace_path).exists()
