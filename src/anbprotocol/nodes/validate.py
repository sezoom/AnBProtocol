from __future__ import annotations
from ..state import GraphState
from ..heuristics import basic_score, MAD_score

def validate_node(state: GraphState) -> GraphState:
    flow = state["flow"]
    score, notes = basic_score(flow)
    ## update the extra feedback from another LLM
    notes+=MAD_score(flow)
    flow.notes.extend(notes)
    state["score"] = score
    return state
