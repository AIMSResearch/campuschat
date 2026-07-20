# Campus Chat Data Dictionary

## campus_intents.csv
| Field | Description |
|---|---|
| example_id | Stable synthetic example ID |
| text | User-style question |
| intent | Training label |

## campus_documents.csv
| Field | Description |
|---|---|
| document_id | Stable source ID |
| title | Document title |
| document_text | Fictional policy/support text |
| owner | Accountable office |
| effective_date | Policy effective date |
| last_reviewed | Last review date |
| source_url | Fictional source URL |
| access_classification | public or confidential |
| document_version | Source version |
| intent | Topic used for routing and evaluation |

## campus_benchmark.csv
| Field | Description |
|---|---|
| benchmark_id | Stable evaluation case ID |
| question | Benchmark question |
| expected_intent | Expected classifier result |
| expected_document_id | Expected authoritative source |

## sample_feedback.csv
| Field | Description |
|---|---|
| feedback_id | Stable feedback ID |
| timestamp_utc | Capture time |
| request_id | Link to served response |
| rating | User or reviewer signal |
| comment | Free-text feedback |
| review_status | Human review state |
| approved_use | Whether the record may be reused |
