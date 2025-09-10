from __future__ import annotations
from typing import Tuple
from .state import Flow

def basic_score(flow: Flow) -> Tuple[float, list]:
    msgs = flow.messages
    notes = []
    score = 0.0
    if len(msgs) >= 3:
        score += 0.3
    else:
        notes.append("Fewer than 3 messages.")
    parties = set([m.sender for m in msgs] + [m.receiver for m in msgs])
    if "Alice" in parties and "Bob" in parties:
        score += 0.2
    else:
        notes.append("Roles Alice/Bob missing in messages.")
    if msgs and msgs[0].sender == "Alice" and msgs[0].receiver == "Bob":
        score += 0.2
    else:
        notes.append("First message is not Alice->Bob.")
    content_blob = " ".join([m.content.lower() for m in msgs])
    if any(tok in content_blob for tok in ["nonce", "na", "nb", "kdf", "session key", "ephemeral", "dh"]):
        score += 0.3
    else:
        notes.append("No clear mention of nonces/key exchange.")
    return min(1.0, score), notes

def advance_score(flow: Flow) -> Tuple[float, list]:
    msgs = flow.messages
    #TODO: use LLM to to score the output
    notes = []
    score = 0.0
    return min(1.0, score), notes
