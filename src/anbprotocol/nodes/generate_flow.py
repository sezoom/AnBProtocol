from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_core.prompts import PromptTemplate
from ..state import GraphState, Message, Flow
from ..llm import make_llm

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generate_flow.txt"

def _lines_to_messages(lines: List[str]) -> List[Message]:
    msgs = []
    step = 1
    for line in lines:
        s = line.strip()
        if not s or "->" not in s or ":" not in s:
            continue
        left, content = s.split(":", 1)
        sender, receiver = [x.strip() for x in left.split("->")]
        msgs.append(Message(step=step, sender=sender, receiver=receiver, content=content.strip()))
        step += 1
    return msgs

def generate_flow_node(state: GraphState) -> GraphState:
    llm = make_llm()
    prompt = PromptTemplate(
        template=PROMPT_PATH.read_text(),
        input_variables=["raw_text"],
    )
    raw_text = state["raw_text"]
    print("DebugMSG_prompt:",prompt)
    print("DebugMSG_spec:",raw_text)
    text = (prompt | llm).invoke({"raw_text": raw_text}).content
    print("DebugMSG_txt:",text)
    lines = text.splitlines()
    flow = Flow(messages=_lines_to_messages(lines))
    state["flow"] = flow
    state["iter"] = int(state.get("iter", 0))

    return state
