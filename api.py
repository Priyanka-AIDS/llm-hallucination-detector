from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded  # type: ignore

# 🔥 Import pipeline
from pipeline import run_pipeline

# =========================
# ⚙️ Setup
# =========================

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="RUC-Detect API",
    version="1.0.0",
    description="Production API for real-time hallucination detection."
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# =========================
# 📦 Schemas
# =========================

class DetectRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)
    response: str = Field(..., max_length=5000)
    model_id: Optional[str] = Field("llama-3-8b")


class Span(BaseModel):
    start: int
    end: int
    text: str
    confidence: float


class DetectResponse(BaseModel):
    hallucination_score: float
    is_hallucinated: bool
    explanation: str
    flagged_spans: List[Span]
    component_scores: Dict[str, float]


# =========================
# 🟢 Health Check
# =========================

@app.get("/health")
async def health_check():
    return {"status": "ok"}


# =========================
# 🚀 Detection Endpoint
# =========================

@app.post("/detect", response_model=DetectResponse)
@limiter.limit("50/minute")
async def detect_hallucination(request: Request, payload: DetectRequest):
    try:
        result = run_pipeline(payload.prompt, payload.response)

        return DetectResponse(
            hallucination_score=result["score"],
            is_hallucinated=result["label"],
            explanation=result["explanation"],
            flagged_spans=[
                Span(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    confidence=s["confidence"]
                )
                for s in result["spans"]
            ],
            component_scores=result["components"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# ▶️ Run Server
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)