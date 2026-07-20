from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ARTIFACT_DIR, EVIDENCE_DIR, INTENT_CONFIDENCE_THRESHOLD, TOP_K


@dataclass
class ChatResult:
    request_id: str
    answer: str
    predicted_intent: str
    confidence: float
    sources: list[dict[str, Any]]
    trace_path: str


class CampusChatEngine:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        self.intent_model = joblib.load(artifact_dir / "intent_model.joblib")
        self.intent_vectorizer = joblib.load(artifact_dir / "intent_vectorizer.joblib")
        self.retrieval_vectorizer = joblib.load(artifact_dir / "retrieval_vectorizer.joblib")
        self.retrieval_matrix = sparse.load_npz(artifact_dir / "retrieval_matrix.npz")
        self.chunks = pd.read_csv(artifact_dir / "retrieval_chunks.csv").fillna("")
        self.manifest = json.loads((artifact_dir / "deployment_manifest.json").read_text(encoding="utf-8"))
        self.prompt_registry = json.loads((artifact_dir / "prompt_registry.json").read_text(encoding="utf-8"))
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    def classify_intent(self, message: str) -> tuple[str, float]:
        vector = self.intent_vectorizer.transform([message])
        probabilities = self.intent_model.predict_proba(vector)[0]
        index = int(probabilities.argmax())
        return str(self.intent_model.classes_[index]), float(probabilities[index])

    def retrieve(self, message: str, predicted_intent: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        query_vector = self.retrieval_vectorizer.transform([message])
        scores = cosine_similarity(query_vector, self.retrieval_matrix).flatten()
        order = scores.argsort()[::-1]
        results: list[dict[str, Any]] = []
        for position in order:
            row = self.chunks.iloc[int(position)]
            # Prefer the predicted intent, but allow broader retrieval when evidence is sparse.
            if row["intent"] != predicted_intent and len(results) < 1:
                continue
            results.append({
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "document_version": row["document_version"],
                "title": row["title"],
                "owner": row["owner"],
                "effective_date": row["effective_date"],
                "source_url": row["source_url"],
                "score": round(float(scores[int(position)]), 4),
                "chunk_text": row["chunk_text"],
            })
            if len(results) >= top_k:
                break
        return results

    def compose_answer(self, message: str, intent: str, confidence: float, sources: list[dict[str, Any]]) -> str:
        if intent == "emergency":
            return (
                "If there is an immediate threat to life or safety, contact local emergency services now. "
                "When it is safe, contact campus safety. This chat assistant should not be used as the only source of help during an emergency."
            )
        if intent == "greeting":
            return "Hello! I can help you find approved campus information about registration, advising, degree requirements, financial aid, services, and technical support."
        if confidence < INTENT_CONFIDENCE_THRESHOLD or not sources:
            return (
                "I am not confident enough to answer from the approved campus knowledge base. "
                "Please rephrase the question or contact the responsible campus office."
            )
        source = sources[0]
        return (
            f"According to **{source['title']}** (version {source['document_version']}, effective {source['effective_date']}), "
            f"{source['chunk_text']}\n\n"
            f"Source: {source['source_url']}\n"
            f"For decisions about your individual record, contact {source['owner']}."
        )

    def chat(self, message: str) -> ChatResult:
        if not message or not message.strip():
            raise ValueError("message must not be empty")
        request_id = "req-" + uuid.uuid4().hex[:12]
        response_id = "resp-" + uuid.uuid4().hex[:12]
        intent, confidence = self.classify_intent(message)
        sources = self.retrieve(message, intent)
        answer = self.compose_answer(message, intent, confidence, sources)
        trace = {
            "request_id": request_id,
            "response_id": response_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "user_message": message,
            "predicted_intent": intent,
            "intent_confidence": round(confidence, 4),
            "deployment_version": self.manifest["deployment_version"],
            "intent_model_version": self.manifest["intent_model_version"],
            "prompt_template_version": self.manifest["prompt_template_version"],
            "corpus_snapshot_id": self.manifest["corpus_snapshot_id"],
            "vector_index_version": self.manifest["vector_index_version"],
            "retrieved_chunks": [
                {
                    "chunk_id": s["chunk_id"],
                    "document_id": s["document_id"],
                    "document_version": s["document_version"],
                    "score": s["score"],
                }
                for s in sources
            ],
            "final_response": answer,
            "retention_class": "redacted_evidence_30_days",
        }
        trace_path = EVIDENCE_DIR / f"{request_id}.json"
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        return ChatResult(
            request_id=request_id,
            answer=answer,
            predicted_intent=intent,
            confidence=confidence,
            sources=sources,
            trace_path=str(trace_path),
        )
