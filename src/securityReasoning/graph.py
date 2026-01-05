from __future__ import annotations
from typing import List, Optional, Literal, TypedDict, Dict, Any
import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from rich.console import Console
from rich.panel import Panel
import re
from .llm import extract_k2_think_answer,make_llm
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver


llm=make_llm(os.getenv("LLM_REASONING"),temperature=0.1)

Console().print(f"Model Name: [green]{llm.model_name}[/green]")
Console().print(f"Temperature: [green]{llm.temperature}[/green]")

PROMPT_1_PATH = Path(__file__).resolve().parent / "securityGoals" / "1.txt"
PROMPT_2_PATH = Path(__file__).resolve().parent/ "securityGoals" / "2.txt"
PROMPT_3_PATH = Path(__file__).resolve().parent/ "securityGoals" / "3.txt"
PROMPT_4_PATH = Path(__file__).resolve().parent/ "securityGoals" / "4.txt"
PROMPT_5_PATH = Path(__file__).resolve().parent/ "securityGoals" / "5.txt"
PROMPT_6_PATH = Path(__file__).resolve().parent/ "securityGoals" / "6.txt"
PROMPT_7_PATH = Path(__file__).resolve().parent/ "securityGoals" / "7.txt"
PROMPT_8_PATH = Path(__file__).resolve().parent/ "securityGoals" / "8.txt"
PROMPT_9_PATH = Path(__file__).resolve().parent/ "securityGoals" / "9.txt"
PROMPT_10_PATH = Path(__file__).resolve().parent / "securityGoals" / "10.txt"

def parse_goal_output(text: str):
    """
    Expected format:
        <yes/no><optional whitespace><optional newline><justification>

    Returns (result, reason), where result is 'yes' or 'no' (lowercase),
    and reason is the remaining text (stripped), possibly empty.
    """
    if text is None:
        return "no", "Empty model output."

    s = text.strip()

    # Grab first token as yes/no
    m = re.match(r"^(yes|no)\b(.*)$", s, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        # Fallback: if we can't find yes/no, treat as 'no' with explanation
        return "no", f"Output not in expected format: {text!r}"

    result = m.group(1).lower()
    reason = m.group(2).strip()
    return result, reason

# Graph state
class State(TypedDict):
    protocol: str
    adversaryModel:str
    adversaryModel: str
    resultForGoal1: str
    reasonForGoal1: str
    resultForGoal2: str
    reasonForGoal2: str
    resultForGoal3: str
    reasonForGoal3: str
    resultForGoal4: str
    reasonForGoal4: str
    resultForGoal5: str
    reasonForGoal5: str
    resultForGoal6: str
    reasonForGoal6: str
    resultForGoal7: str
    reasonForGoal7: str
    resultForGoal8: str
    reasonForGoal8: str
    resultForGoal9:str
    reasonForGoal9: str
    resultForGoal10: str
    reasonForGoal10: str
    combined_output: str

# Nodes
def call_llm_1(state: State):
    prompt = PromptTemplate(
        template=PROMPT_1_PATH.read_text()
    )

    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": state['adversaryModel']})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal1":result,"reasonForGoal1": reason}



def call_llm_2(state: State):
    prompt = PromptTemplate(
        template=PROMPT_2_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": state['adversaryModel']})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal2":result,"reasonForGoal2": reason}


def call_llm_3(state: State):
    prompt = PromptTemplate(
        template=PROMPT_3_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": state['adversaryModel']})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal3":result,"reasonForGoal3": reason}

def call_llm_4(state: State):
    prompt = PromptTemplate(
        template=PROMPT_4_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": state['adversaryModel']})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal4":result,"reasonForGoal4": reason}

def call_llm_5(state: State):
    prompt = PromptTemplate(
        template=PROMPT_5_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": state['adversaryModel']})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal5":result,"reasonForGoal5": reason}

def call_llm_6(state: State):
    prompt = PromptTemplate(
        template=PROMPT_6_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": state['adversaryModel']})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal6":result,"reasonForGoal6": reason}

def call_llm_7(state: State):
    prompt = PromptTemplate(
        template=PROMPT_7_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": state['adversaryModel']})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal7":result,"reasonForGoal7": reason}

def call_llm_8(state: State):
    prompt = PromptTemplate(
        template=PROMPT_8_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": state['adversaryModel']})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal8":result,"reasonForGoal8": reason}

def call_llm_9(state: State):
    prompt = PromptTemplate(
        template=PROMPT_9_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": state['adversaryModel']})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal9":result,"reasonForGoal9": reason}

def call_llm_10(state: State):
    prompt = PromptTemplate(
        template=PROMPT_10_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": state['adversaryModel']})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal10":result,"reasonForGoal10": reason}



def aggregator(state: State):
    """
    Aggregate all per-goal results into a single ASCII report string.
    Produces a 'combined_output' field in the state with:
      - summary statistics at the top
      - per-goal title, result, and reasoning
    """
    GOAL_TITLES = {
        1: "Correctness",
        2: "Secrecy of transport data & traffic keys",
        3: "Forward Secrecy",
        4: "Mutual Authentication",
        5: "Resistance against Key Compromise Impersonation (KCI)",
        6: "Resistance against Identity mis-binding / unknown key-share",
        7: "Resistance against Replay Attack",
        8: "Session Uniqueness",
        9: "Channel Binding",
        10: "Identity Hiding",
    }

    total = 0
    yes_count = 0
    no_count = 0

    # First pass: compute stats
    for i in range(1, 11):
        res = (state.get(f"resultForGoal{i}") or "").strip()
        if not res and not (state.get(f"reasonForGoal{i}") or "").strip():
            # Goal not evaluated / not applicable
            continue

        total += 1
        res_norm = res.lower()
        if res_norm == "yes":
            yes_count += 1
        elif res_norm == "no":
            no_count += 1


    # Build summary header
    header_lines = []
    header_lines.append("=== Security Evaluation Summary ===")
    header_lines.append(f"Total goals evaluated : {total}")
    header_lines.append(f"Goals satisfied (yes)  : {yes_count}")
    header_lines.append(f"Goals violated (no)    : {no_count}")

    if total > 0:
        yes_pct = 100.0 * yes_count / total
        no_pct = 100.0 * no_count / total
        header_lines.append(
            f"Success rate           : {yes_pct:.1f}% yes / {no_pct:.1f}% no"
        )

    header_lines.append("=" * 40)
    header_lines.append("")


    header_lines.append(state['protocol'])
    header_lines.append("")

    header_lines.append("=== Security Evaluation ===")

    # Second pass: per-goal details
    detail_lines = []
    for i in range(1, 11):
        res = (state.get(f"resultForGoal{i}") or "").strip()
        reason = (state.get(f"reasonForGoal{i}") or "").strip()

        if not res and not reason:
            continue  # skip goals with no data

        title = GOAL_TITLES.get(i)
        detail_lines.append(f"---{i}: {title} ---")
        detail_lines.append(f"Result: {res or 'N/A'}")

        if reason:
            detail_lines.append("Reasoning:")
            detail_lines.append(reason)
        else:
            detail_lines.append("Reasoning: (not provided)")

        detail_lines.append("")  # blank line between goals

    combined_output = "\n".join(header_lines + detail_lines).rstrip() + "\n"
    return {"combined_output": combined_output}

# Build workflow
def build_graph() -> StateGraph:
    parallel_builder = StateGraph(State)

    # Add nodes
    parallel_builder.add_node("call_llm_1", call_llm_1)
    parallel_builder.add_node("call_llm_2", call_llm_2)
    parallel_builder.add_node("call_llm_3", call_llm_3)
    parallel_builder.add_node("call_llm_4", call_llm_4)
    parallel_builder.add_node("call_llm_5", call_llm_5)
    parallel_builder.add_node("call_llm_6", call_llm_6)
    parallel_builder.add_node("call_llm_7", call_llm_7)
    parallel_builder.add_node("call_llm_8", call_llm_8)
    parallel_builder.add_node("call_llm_9", call_llm_9)
    parallel_builder.add_node("call_llm_10", call_llm_10)
    parallel_builder.add_node("aggregator", aggregator)

    # Add edges to connect nodes
    parallel_builder.add_edge(START, "call_llm_1")
    parallel_builder.add_edge(START, "call_llm_2")
    parallel_builder.add_edge(START, "call_llm_3")
    parallel_builder.add_edge(START, "call_llm_4")
    parallel_builder.add_edge(START, "call_llm_5")
    parallel_builder.add_edge(START, "call_llm_6")
    parallel_builder.add_edge(START, "call_llm_7")
    parallel_builder.add_edge(START, "call_llm_8")
    parallel_builder.add_edge(START, "call_llm_9")
    parallel_builder.add_edge(START, "call_llm_10")
    parallel_builder.add_edge("call_llm_1", "aggregator")
    parallel_builder.add_edge("call_llm_2", "aggregator")
    parallel_builder.add_edge("call_llm_3", "aggregator")
    parallel_builder.add_edge("call_llm_4", "aggregator")
    parallel_builder.add_edge("call_llm_5", "aggregator")
    parallel_builder.add_edge("call_llm_6", "aggregator")
    parallel_builder.add_edge("call_llm_7", "aggregator")
    parallel_builder.add_edge("call_llm_8", "aggregator")
    parallel_builder.add_edge("call_llm_9", "aggregator")
    parallel_builder.add_edge("call_llm_10", "aggregator")
    parallel_builder.add_edge("aggregator", END)
    memory = MemorySaver()
    parallel_workflow = parallel_builder.compile(checkpointer=memory)
    return parallel_workflow
# # Show workflow
# display(Image(parallel_workflow.get_graph().draw_mermaid_png()))
### generating image for the current workflow
# from IPython.display import Image, display
# png_data = parallel_workflow.get_graph().draw_mermaid_png()
# output_file_path = "graphReasoning.png"
# with open(output_file_path, "wb") as f:
#     f.write(png_data)
# Invoke

# PROTOCOL_1_PATH = Path(__file__).resolve().parent.parent.parent / "dataset" /"anb"/ "1.anb"
# state = parallel_workflow.invoke({"protocol": PROTOCOL_1_PATH.read_text()})
#
# print(state['combined_output'])