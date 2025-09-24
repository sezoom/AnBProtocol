from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_core.prompts import PromptTemplate
from ..state import GraphState, Message, Flow
from ..llm import make_llm
import re

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generate_flow.txt"

STEP_RE = re.compile(
    r'^\s*\[step(?P<num>\d+)\]\s*(?P<who>.+?)\s*:\s*(?P<inline>.*)$',
    flags=re.IGNORECASE
)

WHO_RE = re.compile(
    r'^\s*(?P<lhs>[^()]+?)(?:\s*\((?P<note>[^)]*)\))?\s*$'
)

def _clean_content(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*;\s*$', '', s)
    return s

def _parse_who(who: str) -> (str, Optional[str], Optional[str]):
    """
    Parse 'Alice -> Bob (note)' OR 'Both (note)' into (sender, receiver, note).
    If there is no '->', set receiver=None.
    """
    m = WHO_RE.match(who)
    if not m:
        return who.strip(), None, None

    lhs = m.group('lhs').strip()
    note = (m.group('note') or '').strip() or None

    if '->' in lhs:
        sender, receiver = [x.strip() for x in lhs.split('->', 1)]
    else:
        sender, receiver = lhs, None
    return sender, receiver, note

def _lines_to_messages(lines: List[str]) -> List[Dict]:
    msgs: List[Dict] = []
    i = 0
    n = len(lines)

    def is_step(line: str) -> bool:
        return STEP_RE.match(line) is not None

    while i < n:
        line = lines[i]
        if line.strip().lower() == 'end':
            break

        m = STEP_RE.match(line)
        if not m:
            i += 1
            continue

        step = int(m.group('num'))
        who = m.group('who')
        inline = (m.group('inline') or '').strip()

        sender, receiver, note = _parse_who(who)

        # Collect content: inline + following lines until next step or 'end'
        content_lines = []
        if inline:
            content_lines.append(inline)

        i += 1
        while i < n:
            nxt = lines[i]
            if nxt.strip().lower() == 'end' or is_step(nxt):
                break
            if nxt.strip():  # skip pure empty lines
                content_lines.append(nxt.strip())
            i += 1

        content = _clean_content(" ".join(content_lines))

        msgs.append(Message(step=step, sender=sender, receiver=receiver,note=note, content=content.strip()))
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
    print("DebugMSG_prompt1_output:",text)
    lines = text.splitlines()
    flow = Flow(messages=_lines_to_messages(lines))
    state["flow"] = flow
    state["iter"] = int(state.get("iter", 0))

    return state
