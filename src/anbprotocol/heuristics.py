from __future__ import annotations
from typing import Tuple, List, Dict, Callable
import re
from .state import Flow, Message,llm2
from pathlib import Path
from langchain_core.prompts import PromptTemplate


PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "evaluate_flow.txt"

def _lc(s: str) -> str:
    return (s or "").lower()

def _has_any(s: str, terms: List[str]) -> bool:
    s = _lc(s)
    return any(t.lower() in s for t in terms)

def _regex_search(s: str, pat: str) -> bool:
    return re.search(pat, s or "", flags=re.IGNORECASE) is not None

def _steps_consecutive(msgs: List[Message]) -> bool:
    if not msgs:
        return False
    steps = [m.step for m in msgs]
    # strictly increasing by 1 starting from the first step (usually 1)
    return all(steps[i] + 1 == steps[i+1] for i in range(len(steps)-1))

def _roles_ok(msgs: List[Message]) -> bool:
    parties = {m.sender for m in msgs} | {m.receiver for m in msgs}
    return "Alice" in parties and "Bob" in parties

def _directions(msgs: List[Message]) -> List[str]:
    return [f"{m.sender}->{m.receiver}" for m in msgs]

def _content_blob(msgs: List[Message]) -> str:
    return " ".join([_lc(m.content) for m in msgs])

def _notes_blob(msgs: List[Message]) -> str:
    return " ".join([_lc(m.note) for m in msgs if m.note])

# -------- Basic scoring --------

def basic_score(flow: Flow) -> Tuple[float, List[str]]:
    """
    Lightweight checklist:
      - >=3 messages
      - Both roles appear
      - Canonical first two directions: Alice->Bob then Bob->Alice
      - Mentions of NA/NB/kdf/senc/ACK anywhere in content or notes
    """
    msgs = flow.messages
    notes: List[str] = []
    score = 0.0
    # At least three messages
    if len(msgs) >= 3:
        score += 0.30
    else:
        notes.append("Fewer than 3 messages.")

    # Roles present
    if _roles_ok(msgs):
        score += 0.20
    else:
        notes.append("Roles Alice/Bob missing.")

    # Directions for first two steps
    dirs = _directions(msgs)
    if len(dirs) >= 1 and dirs[0].lower() == "alice->bob":
        score += 0.15
    else:
        notes.append("Step1 is not Alice->Bob.")

    if len(dirs) >= 2 and dirs[1].lower() == "bob->alice":
        score += 0.15
    else:
        notes.append("Step2 is not Bob->Alice.")

    # Cryptographic ingredients
    blob = _content_blob(msgs) + " " + _notes_blob(msgs)
    got_na = _has_any(blob, ["na"])
    got_nb = _has_any(blob, ["nb"])
    got_kdf = _has_any(blob, ["kdf"])
    got_senc_ack = (_has_any(blob, ["senc"]) and _has_any(blob, ["ack"]))

    if got_na | got_nb | got_kdf | got_senc_ack:
        score += 0.05
    else:
        notes.append("No symmetric encryption of ACK detected.")

    print("DebugMSG_score_output:",score)
    return min(1.0, round(score, 4)), notes

def MAD_score(state: GraphState) -> Tuple[float, list]:


    flow_raw = state["flow_raw"]

    prompt = PromptTemplate(
        template=PROMPT_PATH.read_text(),
        input_variables=["flow_text"],
    )
    feedback = (prompt | llm2).invoke({"flow_text": flow_raw}).content
    print("DebugMSG_MAD_output:",feedback)


    return feedback
