from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.config import ARTIFACT_DIR, PROJECT_ROOT


def train_intent_classifier() -> dict:
    data = pd.read_csv(PROJECT_ROOT / "data/raw/campus_intents.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["intent"], test_size=0.25,
        random_state=42, stratify=data["intent"]
    )
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=5000, sublinear_tf=True
    )
    model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42, C=4.0)
    model.fit(vectorizer.fit_transform(X_train), y_train)
    prediction = model.predict(vectorizer.transform(X_test))
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro")),
        "classification_report": classification_report(
            y_test, prediction, output_dict=True, zero_division=0
        ),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "intent_model.joblib")
    joblib.dump(vectorizer, ARTIFACT_DIR / "intent_vectorizer.joblib")
    (ARTIFACT_DIR / "intent_training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_intent_classifier(), indent=2))
