from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_core.prompts import PromptTemplate
from ..state import GraphState, Message, Flow,Flow_raw,llm1
from rich.console import Console
from rich.panel import Panel

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
    flow = state["flow_raw"]
    #text = "\n".join([f"{m.sender} -> {m.receiver}: {m.content}" for m in flow.messages])
    # last message
    text=flow.messages[-1]
    #print("DebugMessage_flow_tex:", text)
    prompt = PromptTemplate(
        template=PROMPT_PATH.read_text(),
        input_variables=["flow_text"],
    )
    #take the last feedback
    feedback_text = flow.feedback[-1]
    improved = (prompt | llm1).invoke({"flow_text": text,"feedback_text":feedback_text}).content
    #lines = improved.splitlines()
    #state["flow"] = Flow(messages=_lines_to_messages(lines), notes=flow.notes)
    state["flow_raw"] = Flow_raw(messages=[improved])
    Console().print(Panel.fit(f"[bold]Agent 1:[/bold] {feedback}"))
    # print("DebugOptimize_prompt:",prompt)
    # print("DebugMessage_input:", text)
    # print("DebugMessage_output:", improved)
    state["iter"] = int(state.get("iter", 0)) + 1
    return state
