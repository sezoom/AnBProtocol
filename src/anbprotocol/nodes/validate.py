from __future__ import annotations
from ..state import GraphState
from ..heuristics import basic_score, MAD_score
import re
_SCORE_RE = re.compile(
    r'^\s*\{\s*Score\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\}'
)
# read the score from the text {score=0.} which graded by the evaluator
def extract_score(text: str, default=None) -> float | None:
    m = _SCORE_RE.match(text)
    if not m:
        return default
    return float(m.group(1))


def validate_node(state: GraphState) -> GraphState:
    flow_raw = state["flow_raw"]
    #score, notes = basic_score(flow)
    ## update the extra feedback from another LLM
    feedback1=MAD_score(state)
    flow_raw.feedback.extend(feedback1)
    state["score"] = extract_score(feedback1)
    return state
