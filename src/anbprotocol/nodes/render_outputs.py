from __future__ import annotations
from ..state import GraphState
from ..render import render_all

def render_outputs_node(state: GraphState) -> GraphState:
    state["rendered"] = render_all(state["flow"])
    return state
