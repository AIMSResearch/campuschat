from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import ARTIFACT_DIR, PROJECT_ROOT
from src.data_pipeline import make_snapshot_id, validate_corpus

CHUNK_SIZE = 65
CHUNK_OVERLAP = 15
CHUNKING_VERSION = "word-window-v1"
RETRIEVAL_MODEL_VERSION = "tfidf-retriever-v1"


def chunk_text(text: str) -> list[str]:
    words = str(text).split()
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_SIZE]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + CHUNK_SIZE >= len(words):
            break
    return chunks


def build_retriever() -> dict:
    raw = pd.read_csv(PROJECT_ROOT / "data/raw/campus_documents.csv").fillna("")
    approved, quarantine = validate_corpus(raw)
    quarantine.to_csv(PROJECT_ROOT / "data/quarantine/quarantine_report.csv", index=False)
    content_keys = (
        approved["document_id"].astype(str) + "||" +
        approved["document_version"].astype(str) + "||" +
        approved["document_text"].astype(str)
    ).tolist()
    corpus_snapshot_id = make_snapshot_id(content_keys, "campus-corpus")

    rows: list[dict] = []
    for _, row in approved.iterrows():
        for position, chunk in enumerate(chunk_text(row["document_text"]), start=1):
            rows.append({
                "chunk_id": f"{row['document_id']}:{row['document_version']}:c{position:03d}",
                "document_id": row["document_id"],
                "document_version": row["document_version"],
                "title": row["title"],
                "intent": row["intent"],
                "owner": row["owner"],
                "effective_date": row["effective_date"],
                "source_url": row["source_url"],
                "chunk_text": chunk,
                "corpus_snapshot_id": corpus_snapshot_id,
                "chunking_version": CHUNKING_VERSION,
            })
    chunks = pd.DataFrame(rows)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks["chunk_text"])
    chunks.to_csv(ARTIFACT_DIR / "retrieval_chunks.csv", index=False)
    joblib.dump(vectorizer, ARTIFACT_DIR / "retrieval_vectorizer.joblib")
    sparse.save_npz(ARTIFACT_DIR / "retrieval_matrix.npz", matrix)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_snapshot_id": corpus_snapshot_id,
        "chunking_version": CHUNKING_VERSION,
        "retrieval_model_version": RETRIEVAL_MODEL_VERSION,
        "approved_document_count": int(len(approved)),
        "quarantined_document_count": int(len(quarantine)),
        "chunk_count": int(len(chunks)),
    }
    (ARTIFACT_DIR / "retriever_build.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_retriever(), indent=2))
