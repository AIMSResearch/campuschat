# Model Card: Campus Intent Classifier

## Model
TF-IDF text features with Logistic Regression classification.

## Intended use
Route fictional campus questions to a topic area before retrieval.

## Metrics from the bundled run
- Accuracy: 0.985
- Macro F1: 0.985

## Limitations
- Synthetic and intentionally easy training data.
- English-only.
- Confidence values are not calibrated for high-stakes use.
- The model does not understand individual student records.
- Intent classification does not guarantee retrieval or answer correctness.

## Required controls
Use confidence thresholds, retrieval evidence, source citations, benchmark tests, human escalation, monitoring, and a deployment manifest.
