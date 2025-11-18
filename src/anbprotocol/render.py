from __future__ import annotations
from typing import Dict, Any
from .state import Flow, Rendered,Flow_raw

#TODO: update the rendering to support raw flow only for  JSON and mermaid
def to_mermaid(flow) -> str:
    if type(flow) == Flow_raw:
        return "" ## to be implemented
    else:
        lines = ["sequenceDiagram", "    participant Alice", "    participant Bob"]
        for m in flow.messages:
            lines.append(f"    {m.sender} ->> {m.receiver}: {m.content}")
            if m.note:
                lines.append(f"    Note over {m.sender},{m.receiver}: {m.note}")
        return "\n".join(lines)

def to_ascii(flow) -> str:
    if type(flow) == Flow_raw:
        # return the last message of the debate only
        return flow.messages[-1]
    else:
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

def to_json(flow) -> Dict[str, Any]:
    if type(flow) == Flow_raw:
        return {} #TODO:To be implemented
    else:
        return {"messages": [m.model_dump() for m in flow.messages], "notes": flow.notes}

def render_all(flow: Flow) -> Rendered:
    return Rendered(mermaid=to_mermaid(flow), ascii=to_ascii(flow), json=to_json(flow))

def render_flow_raw(flow_raw: Flow_raw) -> Rendered:
    return Rendered(mermaid=to_mermaid(flow_raw),ascii=to_ascii(flow_raw),json=to_json(flow_raw))