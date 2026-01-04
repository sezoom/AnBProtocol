from __future__ import annotations
import pathlib
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from .graph import build_graph
import os

app = typer.Typer(add_completion=False)
console = Console()
ROOT = pathlib.Path("AnBProtocol")

@app.command()
def single(
    in_: Path = typer.Option(..., "--in", help="Path to protocol description .anb"),
    out: Path = typer.Option("out.txt", "--out", help="Output txt path"),
):
    text = in_.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("[protoflow] The input file is empty. Please provide a protocol description.")
    graph = build_graph()
    state = {"protocol": text}

    result = graph.invoke(state, config={"configurable": {"thread_id": in_.stem}})

    rendered = result["result"]


## simple output for text file
    if out.suffix.lower() != ".txt":
        out = out.with_suffix(".txt")
    out.write_text(rendered.ascii, encoding="utf-8")
    console.print(Panel.fit(f"Done. Wrote [bold]{out}[/bold]"))


if __name__ == "__main__":
    app()
