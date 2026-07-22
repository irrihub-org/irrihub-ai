from typing import Literal, Optional

from pydantic import BaseModel, Field


class Sender(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


class ChatRequest(BaseModel):
    
    conversation_id: str
    account_id: str
    message: str
    conversation_history: str | None = None
    channel: str = "web"
    sender: Optional[Sender] = None


class AIResponse(BaseModel):
    
    text: str = Field(description="User-facing response")
    intent: str = Field(default="unclassified", description="Classified intent")
    category: str = Field(default="", description="Sub-category")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    route: Literal["respond", "escalate", "deterministic"] = "respond"
    language: str = Field(default="en")
    sources: list[str] = Field(default_factory=list)
    crm_payload: dict = Field(default_factory=dict)