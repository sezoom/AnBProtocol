from __future__ import annotations
from typing import List, Optional, Literal, TypedDict, Dict, Any
from pydantic import BaseModel, Field

Role = Literal["Alice", "Bob", "Server", "Client", "Adversary"]

class Message(BaseModel):
    step: int
    sender: str
    receiver: str
    content: str
    channel: Optional[str] = None
    note: Optional[str] = None

class Flow(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

class Rendered(BaseModel):
    mermaid: str
    ascii: str
    json: Dict[str, Any]

class GraphState(TypedDict, total=False):
    raw_text: str
    flow: Flow
    score: float
    iter: int
    rendered: Rendered
