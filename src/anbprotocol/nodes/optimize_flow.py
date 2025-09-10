from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_core.prompts import PromptTemplate
from ..state import GraphState, Message, Flow
from ..llm import make_llm

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "optimize_flow.txt"

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

def optimize_flow_node(state: GraphState) -> GraphState:
    llm = make_llm(temperature=0.1)
    flow = state["flow"]
    text = "\n".join([f"{m.sender} -> {m.receiver}: {m.content}" for m in flow.messages])
    prompt = PromptTemplate(
        template=PROMPT_PATH.read_text(),
        input_variables=["flow_text"],
    )
    improved = (prompt | llm).invoke({"flow_text": text}).content
    lines = improved.splitlines()
    state["flow"] = Flow(messages=_lines_to_messages(lines), notes=flow.notes, warnings=flow.warnings)
    state["iter"] = int(state.get("iter", 0)) + 1
    return state
