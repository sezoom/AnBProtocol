from __future__ import annotations
from ..state import GraphState
from ..render import render_all,render_flow_raw

def render_outputs_node(state: GraphState) -> GraphState:
    #state["rendered"] = render_all(state["flow"])
    state["rendered"] = render_flow_raw(state["flow_raw"])
    return state
