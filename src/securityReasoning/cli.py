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

ADVERSARY_MODEL_ACTIVE_PATH = Path(__file__).resolve().parent / "adversaryModel" / "active.txt"
ADVERSARY_MODEL_PASSIVE_PATH = Path(__file__).resolve().parent / "adversaryModel" / "passive.txt"

@app.command()
def single(
    in_: Path = typer.Option(..., "--in", help="Path to protocol description .anb"),
    adv_: Path = typer.Option("P", "--adv", help="P: Passive Adversary or A: Active Adversary"),
    out: Path = typer.Option("out.txt", "--out", help="Output txt path"),
):
    text = in_.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("[protoflow] The input file is empty. Please provide a protocol description.")

    if adv_.stem.lower().strip() == "a":
        iniState = {"adversaryModel": ADVERSARY_MODEL_ACTIVE_PATH.read_text()}
        console.print("Adversary Type: [green]Active[/green]")
    else:
        iniState = {"adversaryModel": ADVERSARY_MODEL_PASSIVE_PATH.read_text()}
        console.print("Adversary Type: [green]Passive[/green]")
    graph = build_graph()
    iniState.update({"protocol": text})

    state = graph.invoke(iniState, config={"configurable": {"thread_id": in_.stem}})

    rendered = state['combined_output']


## simple output for text file
    if out.suffix.lower() != ".txt":
        out = out.with_suffix(".txt")
    out.write_text(rendered, encoding="utf-8")
    console.print(Panel.fit(f"Done. Wrote [bold]{out}[/bold]"))


if __name__ == "__main__":
    app()
