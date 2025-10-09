from __future__ import annotations
import json
import pathlib
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from .graph import build_graph

app = typer.Typer(add_completion=False)
console = Console()
ROOT = pathlib.Path("AnBProtocol")

@app.command()
def run(
    in_: Path = typer.Option(..., "--in", help="Path to protocol description .txt"),
    out: Path = typer.Option("out.md", "--out", help="Output Markdown path"),
    thread: str = typer.Option("default", "--thread", help="Thread id for checkpointing"),
):
    text = in_.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("[protoflow] The input file is empty. Please provide a protocol description.")
    graph = build_graph()
    state = {"raw_text": text}
    #print("DebugMSG_raw_text:",state)
    result = graph.invoke(state, config={"configurable": {"thread_id": thread}})

    rendered = result["rendered"]
    #spec = result["spec"]
    flow = result["flow"]

    md = []
    md.append("# Alice–Bob Flow\n")
    md.append("## Mermaid\n```mermaid\n")
    md.append(rendered.mermaid)
    md.append("\n```\n")
    md.append("## ASCII\n```\n" + rendered.ascii + "\n```\n")
    md.append("## JSON\n```json\n" + json.dumps(rendered.json, indent=2) + "\n```\n")
    #md.append("## Spec\n```json\n" + spec.model_dump_json(indent=2) + "\n```\n")
    md.append("## Notes\n")
    if flow.notes:
        for n in flow.notes:
            md.append(f"- {n}")
    else:
        md.append("- (none)")
    out.write_text("\n".join(md), encoding="utf-8")
    console.print(Panel.fit(f"Done. Wrote [bold]{out}[/bold]"))

if __name__ == "__main__":
    app()
