from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ARTIFACT_DIR, PROJECT_ROOT
from src.inference import CampusChatEngine


def evaluate_application() -> dict:
    benchmark = pd.read_csv(PROJECT_ROOT / "data/eval/campus_benchmark.csv")
    engine = CampusChatEngine()
    rows = []
    intent_hits = 0
    document_hits = 0
    for _, item in benchmark.iterrows():
        result = engine.chat(item["question"])
        retrieved_docs = [source["document_id"] for source in result.sources]
        intent_hit = result.predicted_intent == item["expected_intent"]
        document_hit = item["expected_document_id"] in retrieved_docs[:3]
        intent_hits += int(intent_hit)
        document_hits += int(document_hit)
        rows.append({
            "benchmark_id": item["benchmark_id"],
            "question": item["question"],
            "expected_intent": item["expected_intent"],
            "predicted_intent": result.predicted_intent,
            "intent_hit": intent_hit,
            "expected_document_id": item["expected_document_id"],
            "retrieved_document_ids": "|".join(retrieved_docs),
            "document_hit_at_3": document_hit,
        })
    results = pd.DataFrame(rows)
    results.to_csv(ARTIFACT_DIR / "application_benchmark_results.csv", index=False)
    summary = {
        "intent_accuracy_on_benchmark": intent_hits / len(benchmark),
        "retrieval_hit_at_3_on_benchmark": document_hits / len(benchmark),
        "benchmark_count": int(len(benchmark)),
    }
    (ARTIFACT_DIR / "application_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate_application(), indent=2))
