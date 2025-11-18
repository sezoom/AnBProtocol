from __future__ import annotations
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
env_path = os.getenv('PATH')
load_dotenv(override=True)
dotenv_path = os.getenv('PATH')

if dotenv_path:
    os.environ['PATH'] = env_path + ':' + dotenv_path


def make_llm(model: Optional[str] = None, temperature: float = 0.1) -> ChatOpenAI:
    if "gemini" in model:
                return ChatGoogleGenerativeAI(
                    model=model,
                    temperature=0.1
                )
    else:
    #model = model or os.getenv("PROTOFLOW_MODEL", "gpt-4.1")
        return ChatOpenAI(model=model, temperature=temperature)
