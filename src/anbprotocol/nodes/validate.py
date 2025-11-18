from __future__ import annotations
from ..state import GraphState,Flow_raw
from ..heuristics import basic_score, MAD_score
import re
_SCORE_RE = re.compile(
    r'^\s*(?:'
    r'<\s*Score\s*=\s*(?:>\s*)?(?P<num1>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*>?'  # <Score=...> or <Score=>...>
    r'|'
    r'\{\s*Score\s*=\s*(?P<num2>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\}'          # {Score=...}
    r'|'
    r'Score\s*=\s*(?P<num3>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'                    # Score = ...
    r')'
)
# read the score from the text {score=0.} which graded by the evaluator
def extract_score(text: str, default=None) -> float | None:
    m = _SCORE_RE.match(text)
    if not m:

        return default
    return float(m.group(1))


def validate_node(state: GraphState) -> GraphState:
    ## basic score will be disabled
    #score, notes = basic_score(flow)
    ## feedback from another LLM
    feedback=MAD_score(state)
    state["flow_raw"].feedback.append(feedback)
    state["score"] = extract_score(str(feedback)[:12])
    return state
