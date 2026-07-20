from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import CampusChatEngine

app = FastAPI(
    title="Campus Chat API",
    version="1.0.0",
    description="A teaching application that demonstrates intent classification, retrieval, release manifests, and evidence traces.",
)
engine = CampusChatEngine()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "deployment_version": engine.manifest["deployment_version"],
        "release_status": engine.manifest["release_status"],
    }


@app.get("/manifest")
def manifest() -> dict:
    return engine.manifest


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    try:
        result = engine.chat(request.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "request_id": result.request_id,
        "answer": result.answer,
        "predicted_intent": result.predicted_intent,
        "confidence": round(result.confidence, 4),
        "sources": result.sources,
    }
