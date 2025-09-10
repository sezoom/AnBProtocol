from __future__ import annotations
from typing import Dict, Any
from .state import Flow, Rendered

#TODO: update the rendering to support ASCII only first then extend it to other formats, JSON for debuging purpose
def to_mermaid(flow: Flow) -> str:
    lines = ["sequenceDiagram", "    participant Alice", "    participant Bob"]
    for m in flow.messages:
        lines.append(f"    {m.sender} ->> {m.receiver}: {m.content}")
        if m.note:
            lines.append(f"    Note over {m.sender},{m.receiver}: {m.note}")
    return "\n".join(lines)

def to_ascii(flow: Flow) -> str:
    left = "Alice"; right = "Bob"; width = 40
    lines = [f"{left:<10}|{' '*(width-20)}|{right:>10}", "-"*(width)]
    for m in flow.messages:
        if m.sender == "Alice" and m.receiver == "Bob":
            lines.append(f"Alice ----> Bob : {m.content}")
        elif m.sender == "Bob" and m.receiver == "Alice":
            lines.append(f"Bob   ----> Alice : {m.content}")
        else:
            lines.append(f"{m.sender} -> {m.receiver}: {m.content}")
    return "\n".join(lines)

def to_json(flow: Flow) -> Dict[str, Any]:
    return {"messages": [m.model_dump() for m in flow.messages], "notes": flow.notes, "warnings": flow.warnings}

def render_all(flow: Flow) -> Rendered:
    return Rendered(mermaid=to_mermaid(flow), ascii=to_ascii(flow), json=to_json(flow))
