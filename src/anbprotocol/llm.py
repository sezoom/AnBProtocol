from __future__ import annotations
import os
from typing import Optional
from langchain_openai import ChatOpenAI

def make_llm(model: Optional[str] = None, temperature: float = 0.2) -> ChatOpenAI:
    model = model or os.getenv("PROTOFLOW_MODEL", "gpt-4.1")
    return ChatOpenAI(model=model, temperature=temperature)
