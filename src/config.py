from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
LOG_DIR = PROJECT_ROOT / "logs"

INTENT_CONFIDENCE_THRESHOLD = 0.35
TOP_K = 3
