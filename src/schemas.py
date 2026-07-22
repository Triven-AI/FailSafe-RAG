from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AuditRequest(BaseModel):
    query: str = Field(..., description="User query or transcript prompt to audit.")
    session_id: Optional[str] = Field(default="session-default", description="Tracking ID for caller session.")
    caller_id: Optional[str] = Field(default="caller-anon", description="Operator/Caller identifier.")

class AuditResponse(BaseModel):
    query: str
    response: str
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Factuality/Groundedness confidence score.")
    circuit_breaker_tripped: bool = Field(..., description="Indicates whether TTS safety fallback was triggered.")
    crag_triggered: bool = Field(default=False, description="Indicates whether query rewriting self-corrected retrieval.")
    latency_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthStatus(BaseModel):
    status: str
    service: str
    version: str