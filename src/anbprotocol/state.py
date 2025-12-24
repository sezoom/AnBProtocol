from __future__ import annotations
from typing import List, Optional, Literal, TypedDict, Dict, Any
from pydantic import BaseModel, Field
from .llm import make_llm
import os
from pathlib import Path

llm1=make_llm(os.getenv("LLM_OPTIMIZER"))
llm2=make_llm(os.getenv("LLM_EVALUATOR"))

GUIDELINES_PATH = Path(__file__).resolve().parent.parent.parent / "dataset" / "Notation_guidlines.md"

GUIDELINES = GUIDELINES_PATH.read_text()


class Message(BaseModel):
    step: int
    sender: str
    receiver: str
    content: str
    channel: Optional[str] = None
    note: Optional[str] = None #any fresh value

class Flow(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    feedback: List[str] = Field(default_factory=list)

## we will use raw flow which is not parsing the protocol each time so we will use it as string paragraph and postponding the parsing after completing the debate
class Flow_raw(BaseModel):
    messages: List[str] = Field(default_factory=list)
    feedback: List[str] = Field(default_factory=list)

class Rendered(BaseModel):
    mermaid: str
    ascii: str
    json: Dict[str, Any]

class GraphState(TypedDict, total=False):
    raw_text: str
    flow: Flow
    flow_raw: Flow_raw
    score: float
    iter: int
    rendered: Rendered

