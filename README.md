# AnBProtocol Agent (LangGraph)

An AI agent (built with **LangGraph**) that reads a natural-language description of a **cryptographic protocol** and produces an **Alice–Bob** style flow.
It runs a generator → optimizer loop, validates with simple heuristics, and renders Mermaid/ASCII/JSON.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export PROTOFLOW_MODEL="gpt-4.1"
cd src/
python -m AnBProtocol.cli  --in ../examples/tls_like.txt --out out.md
```
