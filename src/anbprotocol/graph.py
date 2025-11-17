from __future__ import annotations
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import GraphState
from .nodes.generate_flow import generate_flow_node
from .nodes.optimize_flow import optimize_flow_node
from .nodes.validate import validate_node
from .nodes.render_outputs import render_outputs_node


SCORE_THRESHOLD = 0.8
MAX_ITERS = 3



def _router(state: GraphState) -> Literal["optimize", "render"]:
    score = float(state.get("score", 0.0))
    iters = int(state.get("iter", 0))
    if score >= SCORE_THRESHOLD or iters >= MAX_ITERS:
        return "render"
    return "optimize"

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("generate", generate_flow_node)
    graph.add_node("validate", validate_node)
    graph.add_node("optimize", optimize_flow_node)
    graph.add_node("render", render_outputs_node)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", _router, {"optimize": "optimize", "render": "render"})
    graph.add_edge("optimize", "validate")
    graph.add_edge("render", END)

    memory = MemorySaver()

    ### generating image for the current workflow
    # from IPython.display import Image, display
    # chain = graph.compile()
    # png_data = chain.get_graph().draw_mermaid_png()
    # output_file_path = "graph.png"
    # with open(output_file_path, "wb") as f:
    #     f.write(png_data)

    return graph.compile(checkpointer=memory)
