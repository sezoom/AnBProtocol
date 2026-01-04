from __future__ import annotations
from typing import List, Optional, Literal, TypedDict, Dict, Any
import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from rich.console import Console
from rich.panel import Panel
import re
from llm import extract_k2_think_answer,make_llm
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display



llm=make_llm(os.getenv("LLM_REASONING"))

ADVERSARY_MODEL_PATH = Path(__file__).resolve().parent / "adversaryModel" / "active.txt"
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
    adversaryModel: str
    resultForGoal1: bool
    reasonForGoal1: str
    resultForGoal2: bool
    reasonForGoal2: str
    resultForGoal3: bool
    reasonForGoal3: str
    resultForGoal4: bool
    reasonForGoal4: str
    resultForGoal5: bool
    reasonForGoal5: str
    resultForGoal6: bool
    reasonForGoal6: str
    resultForGoal7: bool
    reasonForGoal7: str
    resultForGoal8: bool
    reasonForGoal8: str
    resultForGoal9:bool
    reasonForGoal9: str
    resultForGoal10: bool
    reasonForGoal10: str
    combined_output: str

# Nodes
def call_llm_1(state: State):
    prompt = PromptTemplate(
        template=PROMPT_1_PATH.read_text()
    )

    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": ADVERSARY_MODEL_PATH.read_text()})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal1":result,"reasonForGoal1": reason}



def call_llm_2(state: State):
    prompt = PromptTemplate(
        template=PROMPT_2_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": ADVERSARY_MODEL_PATH.read_text()})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal2":result,"reasonForGoal2": reason}


def call_llm_3(state: State):
    prompt = PromptTemplate(
        template=PROMPT_3_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": ADVERSARY_MODEL_PATH.read_text()})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal3":result,"reasonForGoal3": reason}

def call_llm_4(state: State):
    prompt = PromptTemplate(
        template=PROMPT_4_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": ADVERSARY_MODEL_PATH.read_text()})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal4":result,"reasonForGoal4": reason}

def call_llm_5(state: State):
    prompt = PromptTemplate(
        template=PROMPT_5_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": ADVERSARY_MODEL_PATH.read_text()})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal5":result,"reasonForGoal5": reason}

def call_llm_6(state: State):
    prompt = PromptTemplate(
        template=PROMPT_6_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": ADVERSARY_MODEL_PATH.read_text()})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal6":result,"reasonForGoal6": reason}

def call_llm_7(state: State):
    prompt = PromptTemplate(
        template=PROMPT_7_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": ADVERSARY_MODEL_PATH.read_text()})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal7":result,"reasonForGoal7": reason}

def call_llm_8(state: State):
    prompt = PromptTemplate(
        template=PROMPT_8_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": ADVERSARY_MODEL_PATH.read_text()})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal8":result,"reasonForGoal8": reason}

def call_llm_9(state: State):
    prompt = PromptTemplate(
        template=PROMPT_9_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": ADVERSARY_MODEL_PATH.read_text()})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal9":result,"reasonForGoal9": reason}

def call_llm_10(state: State):
    prompt = PromptTemplate(
        template=PROMPT_10_PATH.read_text()
    )
    output = (prompt | llm).invoke({"protocol": state['protocol'], "attackerModel": ADVERSARY_MODEL_PATH.read_text()})
    result, reason = parse_goal_output(output.content)
    return {"resultForGoal10":result,"reasonForGoal10": reason}



def aggregator(state: State):
    pass


# Build workflow
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
parallel_workflow = parallel_builder.compile()

# # Show workflow
# display(Image(parallel_workflow.get_graph().draw_mermaid_png()))
#
# Invoke

PROTOCOL_1_PATH = Path(__file__).resolve().parent.parent.parent / "dataset" /"anb"/ "1.anb"
state = parallel_workflow.invoke({"protocol": PROTOCOL_1_PATH.read_text()})
print(state)