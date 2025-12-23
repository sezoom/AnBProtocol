from __future__ import annotations
import json
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
    in_: Path = typer.Option(..., "--in", help="Path to protocol description .txt"),
    out: Path = typer.Option("out.txt", "--out", help="Output txt path"),
):
    text = in_.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("[protoflow] The input file is empty. Please provide a protocol description.")
    graph = build_graph()
    state = {"raw_text": text}

    result = graph.invoke(state, config={"configurable": {"thread_id": in_.stem}})

    rendered = result["rendered"]
    #flow = result["flow_raw"]

#### output using MD format ####
    # md = []
    # md.append("# Alice–Bob Flow\n")
    # md.append("## Mermaid\n```mermaid\n")
    # md.append(rendered.mermaid)
    # md.append("\n```\n")
    # md.append("## ASCII\n```\n" + rendered.ascii + "\n```\n")
    # md.append("## JSON\n```json\n" + json.dumps(rendered.json, indent=2) + "\n```\n")
    # #md.append("## Spec\n```json\n" + spec.model_dump_json(indent=2) + "\n```\n")
    # md.append("## Notes\n")
    # if flow.feedback:
    #     for n in flow.feedback:
    #         md.append(f"- {n}")
    # else:
    #     md.append("- (none)")
    # out.write_text("\n".join(md), encoding="utf-8")

## simple output for text file
    if out.suffix.lower() != ".txt":
        out = out.with_suffix(".txt")
    out.write_text(rendered.ascii, encoding="utf-8")
    console.print(Panel.fit(f"Done. Wrote [bold]{out}[/bold]"))

@app.command()
def batch(
    dataset: Path = typer.Option(Path("dataset/natural_language"), "--dataset", help="Folder with input .txt files"),
    output: Path  = typer.Option(Path("output"), "--output", help="Folder to write .anb files"),
    pattern: str  = typer.Option("*.txt", "--pattern", help="Inputs extensions"),
):
    """Process all files in a dataset folder and write .anb outputs to /output."""

    output.mkdir(parents=True, exist_ok=True)
    (output / "afterDebate").mkdir(parents=True, exist_ok=True)
    (output/"beforeDebate").mkdir(parents=True, exist_ok=True)

    graph = build_graph()
    files = sorted(dataset.rglob(pattern))

    if not files:
        raise SystemExit(f"[protoflow] No files matching {pattern!r} under {dataset}")

    for i, inp in enumerate(files, 1):
        try:
            text = inp.read_text(encoding="utf-8")
            if not text.strip():
                console.print(Panel.fit(f"Skip empty file: {inp}", border_style="yellow"))
                continue

            # isolate checkpoints per file so histories don't mix
            thread_id = f"{inp.stem}"
            state = {"raw_text": text}
            result = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
            rendered = result["rendered"]

            out_path = output/ "afterDebate" / (inp.stem + ".anb")
            out_path.write_text(rendered.ascii, encoding="utf-8")

            console.print(Panel.fit(f"[{i}/{len(files)}] Wrote [bold]{out_path}[/bold]", border_style="blue"))
        except Exception as e:
            console.print(Panel.fit(f"Error on {inp} -> {e}", border_style="red"))
    os.system(f"mv ./outputWithoutDebate/* {output / 'beforeDebate/'}")

if __name__ == "__main__":
    app()
