from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_DOCUMENT_FIELDS = [
    "document_id", "title", "document_text", "owner",
    "effective_date", "source_url", "document_version"
]


def normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def validate_document(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    for field in REQUIRED_DOCUMENT_FIELDS:
        if not normalize_text(row.get(field, "")):
            reasons.append(f"missing_{field}")
    if normalize_text(row.get("access_classification", "")).lower() != "public":
        reasons.append("non_public_access")
    try:
        pd.to_datetime(row.get("effective_date", ""))
    except Exception:
        reasons.append("invalid_effective_date")
    if len(normalize_text(row.get("document_text", "")).split()) < 8:
        reasons.append("text_too_short")
    return sorted(set(reasons))


def validate_corpus(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.fillna("").copy()
    result["validation_reasons"] = result.apply(validate_document, axis=1)
    result["validation_status"] = result["validation_reasons"].map(
        lambda reasons: "approved" if not reasons else "quarantined"
    )
    result["quarantine_reason"] = result["validation_reasons"].map(
        lambda reasons: ";".join(reasons)
    )
    return (
        result.query("validation_status == 'approved'").copy(),
        result.query("validation_status == 'quarantined'").copy(),
    )


def make_snapshot_id(values: Iterable[str], prefix: str) -> str:
    payload = "||".join(sorted(values))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"
