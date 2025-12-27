from pathlib import Path
import re, pandas as pd, numpy as np
import math
from collections import Counter
from dataclasses import dataclass

# ---------------- Normalization (full) ----------------
_comment_block_re = re.compile(r"/\*.*?\*/", re.S)
_comment_line1_re = re.compile(r"//.*?$", re.M)
_comment_line2_re = re.compile(r"#.*?$", re.M)
_comment_line3_re = re.compile(r"%.*?$", re.M)

HEADER_KEYS = (
    "declarations","types","knowledge","public","private",
    "actions","goals","channelkeys"
)

# Keywords that must NOT be abstracted
KEYWORDS = set(HEADER_KEYS) | {
    "protocol", "end",
    "agent", "number", "nonce","data",
    "symmetric_key", "function",
    "secret", "between",
    "public", "private", "knowledge", "actions", "goals", "channelkeys",
    "senc", "aenc", "hash","pk", "sk","k","g","kdf","K","h"
}

def strip_comments(s: str) -> str:
    s = _comment_block_re.sub(" ", s)
    s = _comment_line1_re.sub("", s)
    s = _comment_line2_re.sub("", s)
    s = _comment_line3_re.sub("", s)
    return s

def canonical_whitespace_and_tokens(s: str) -> str:
    s = s.lower().replace("\r\n", "\n").replace("\r", "\n")
    # unify [m1]/[step1] (incl. spaced variants)
    s = re.sub(r"\[\s*(?:m|step)\s*(\d+)\s*\]", r"[m\1]", s)
    s = strip_comments(s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([,:;{}()\[\].])\s*", r"\1", s)  # tighten punctuation spacing (incl '.')
    s = re.sub(r"\s*->\s*\*\s*", "->*", s)
    s = re.sub(r"\s*->\s*", "->", s)
    s = re.sub(r"\s*\.\s*", ".", s).strip()
    return s.strip()

# ---------------- Helpers for structured normalization ----------------

_id_re = re.compile(r"[a-z_][a-z0-9_]*")


def split_top_level(s: str, seps: str = ",", openers: str = "{([", closers: str = "})]") -> list[str]:
    """
    Split s on any char in `seps`, but only at top-level
    (not inside braces/parentheses/brackets).
    """
    res = []
    cur = []
    depth = 0
    opener_set = set(openers)
    closer_set = set(closers)
    for ch in s:
        if ch in opener_set:
            depth += 1
        elif ch in closer_set and depth > 0:
            depth -= 1
        if ch in seps and depth == 0:
            seg = "".join(cur).strip()
            if seg:
                res.append(seg)
            cur = []
        else:
            cur.append(ch)
    seg = "".join(cur).strip()
    if seg:
        res.append(seg)
    return res

def strip_outer_parens(expr: str) -> str:
    """
    Remove redundant outer parentheses that wrap the whole expression.
    E.g.  '((f1(n1,n2)))' -> 'f1(n1,n2)'
          '(x)'           -> 'x'
          'f1(n1,n2)'     -> 'f1(n1,n2)'  (unchanged)
    """
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                # if we close the outermost paren before the end, we can't strip
                if depth == 0 and i != len(expr) - 1:
                    ok = False
                    break
        if not ok:
            break
        # outer parens wrap whole expr; strip them and repeat
        expr = expr[1:-1].strip()
    return expr

def normalize_senc_payload(payload: str) -> str:
    """
    Normalize 'senc{X}(Y)', 'senc{X}((Y))', 'senc{X}Y' to the same form:
       'senc{X}Y'
    Only touches patterns starting with 'senc{...}'.
    """
    m = re.match(r"^(senc\{[^{}]*\})(.*)$", payload.strip())
    if not m:
        return payload
    prefix, rest = m.groups()
    rest = rest.strip()
    if not rest:
        return payload
    rest_norm = strip_outer_parens(rest)
    return prefix + rest_norm

def parse_types_line(content: str, type_map: dict[str,str]) -> None:
    """
    Parse a single Types line, e.g.:
      'agent a,b,s'
      'number na,nb'
      'symmetric_key kas,kbs,kab'
    Fill type_map[name] = typename.
    """
    m = re.match(r"^([a-z_][a-z0-9_]*)\s+(.+)$", content)
    if not m:
        return
    typ = m.group(1).strip()
    names_str = m.group(2).strip()
    if not names_str:
        return
    names = split_top_level(names_str, seps=",")
    for n in names:
        n = n.strip()
        if n:
            type_map[n] = typ

def build_rename_map(type_map: dict[str,str]) -> dict[str,str]:
    """
    Build canonical abstract names per type:
      agent -> a1, a2, ...
      number/nonce -> n1, n2, ...
      symmetric_key -> k1, k2, ...
      data -> m1, ...
      function -> f1, ...
    """
    names_by_type: dict[str, list[str]] = {}
    for name, typ in type_map.items():
        names_by_type.setdefault(typ, []).append(name)

    prefix_by_type = {
        "agent": "a",
        "number": "n",
        "symmetric_key": "k",
        "data": "m",
        "function": "f"
    }

    rename_map: dict[str,str] = {}
    for typ, names in names_by_type.items():
        base = prefix_by_type.get(typ, "id")
        # preserve first-occurrence order per type
        seen = set()
        ordered = []
        for n in names:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
        for idx, n in enumerate(ordered, start=1):
            rename_map[n] = f"{base}{idx}"
    return rename_map

def make_abstractifier(rename_map: dict[str,str]):
    """
    Return a function that, given a text, converts all identifiers to
    abstract names except KEYWORDS and those in rename_map.
    """
    extra: dict[str,str] = {}
    counter = [0]
    counter_actions = [0]

    def transform(text: str, sec: str | None = None) -> str:
        def repl(m: re.Match) -> str:
            tok = m.group(0)
            if tok in KEYWORDS:
                return tok
            if tok in rename_map:
                return rename_map[tok]
            if tok in extra:
                return extra[tok]

            # first extra token → 'proto'
            if counter[0] == 0:
                extra[tok] = "proto"
                counter[0] += 1
                return extra[tok]

            # special naming inside Actions section
            if sec == "actions":
                extra[tok] = f"m{counter_actions[0]}"
                counter_actions[0] += 1
                return extra[tok]

            # default naming for all other sections
            new = f"i{counter[0]}"
            counter[0] += 1
            extra[tok] = new
            return new

        return _id_re.sub(repl, text)

    return transform

def sort_list_whole(line: str, seps: str = ",") -> str:
    """
    Treat the whole line as a (possibly structured) list and sort
    the top-level items.
    """
    items = [x.strip() for x in split_top_level(line, seps=seps) if x.strip()]
    items_sorted = sorted(items)
    return ",".join(items_sorted)

def sort_list_after_colon(line: str, seps: str = ",") -> str:
    """
    For 'x: a,b,c' sort a,b,c at top-level.
    """
    if ":" not in line:
        return line
    left, right = line.split(":", 1)
    items = [x.strip() for x in split_top_level(right, seps=seps) if x.strip()]
    items_sorted = sorted(items)
    return f"{left}:{','.join(items_sorted)}"

# ---------------- Binder-free AC term machinery ----------------

@dataclass(frozen=True)
class Term:
    op: str
    args: tuple  # children; () for leaves

# '.' is associative + commutative
AC_FUNCS = {"."}

def find_matching(s: str, start: int, open_ch="{", close_ch="}") -> int:
    """
    Given s[start] == open_ch, find index of its matching close_ch.
    Raises ValueError if not found.
    """
    assert s[start] == open_ch
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"No matching {close_ch} for {open_ch} in {s!r}")

def parse_term(s: str) -> Term:
    """
    Parse a (normalized) message term string into a Term AST, supporting:
      - '.' as an AC operator at top level
      - senc{payload}key, aenc{payload}key, hash{payload}
      - generic function calls f(arg1, arg2, ...)
      - identifiers / constants as leaves
    """
    s = strip_outer_parens(s.strip())
    if not s:
        return Term("•empty•", ())

    # 1) Top-level '.' (AC operator)
    parts = split_top_level(s, seps=".", openers="{([", closers="})]")
    if len(parts) > 1:
        children = tuple(parse_term(p) for p in parts if p)
        return Term(".", children)

    # 2) Curly crypto: senc{payload}key, aenc{payload}key, hash{payload}
    crypto_syms = ["senc", "aenc", "hash"]
    m = re.match(r"^(" + "|".join(crypto_syms) + r")\{", s)
    if m:
        fname = m.group(1)
        brace_idx = len(fname)
        close_idx = find_matching(s, brace_idx, "{", "}")
        inner = s[brace_idx + 1 : close_idx]   # inside {...}
        rest  = s[close_idx + 1 :].strip()     # after '}' (key or empty)

        payload_t = parse_term(inner) if inner.strip() else Term("•empty•", ())
        args = [payload_t]
        if rest:
            args.append(parse_term(rest))
        return Term(fname, tuple(args))

    # 3) Parenthesized function calls: f(args)
    m = re.match(r"^([a-z_][a-z0-9_]*)\((.*)\)$", s)
    if m:
        fname = m.group(1)
        inner = m.group(2)
        arg_parts = split_top_level(inner, seps=",", openers="{([", closers="})]")
        args = tuple(parse_term(p) for p in arg_parts if p.strip())
        return Term(fname, args)

    # 4) Bare identifier / constant
    return Term(s, ())

def term_to_string(t: Term) -> str:
    """
    Canonical pretty-printer consistent with parse_term.
    """
    if not t.args:
        return t.op

    # AC '.' printer
    if t.op == ".":
        return ".".join(term_to_string(c) for c in t.args)

    # Curly crypto
    if t.op in {"senc", "aenc", "hash"}:
        if len(t.args) == 1:
            return f"{t.op}{{{term_to_string(t.args[0])}}}"
        elif len(t.args) == 2:
            return f"{t.op}{{{term_to_string(t.args[0])}}}{term_to_string(t.args[1])}"

    # Generic function call
    return f"{t.op}(" + ",".join(term_to_string(c) for c in t.args) + ")"

def normalize_term(t: Term) -> Term:
    """
    Normalize a term modulo associativity+commutativity of operators in AC_FUNCS.

      - For op in AC_FUNCS:
          * recursively normalize children
          * flatten nested nodes with same op
          * sort children by their canonical string

      - For other ops:
          * recursively normalize children, keep order
    """
    if not t.args:
        return t

    norm_children = [normalize_term(c) for c in t.args]

    if t.op in AC_FUNCS:
        flat = []
        for c in norm_children:
            if c.op == t.op:
                flat.extend(c.args)
            else:
                flat.append(c)
        flat.sort(key=term_to_string)
        return Term(t.op, tuple(flat))

    return Term(t.op, tuple(norm_children))

### ALPHA: placeholders iN / mN → canonical v1,v2,… (α-conversion on abstract variables)
_placeholder_re = re.compile(r"^(i\d+|m\d+)$")

def alpha_normalize_term(t: Term) -> Term:
    """
    α-normalization for abstract placeholders:
      - Any leaf/function name matching i[0-9]+ or m[0-9]+ is treated as
        a bound/abstract variable.
      - We rename them to v1, v2, ... in first-occurrence order within the term.

    Two terms that are identical up to renaming of these placeholders
    normalize to the same α-normal form.
    """
    mapping: dict[str, str] = {}
    counter = [1]

    def visit(node: Term) -> Term:
        op = node.op
        if _placeholder_re.match(op):
            if op not in mapping:
                mapping[op] = f"v{counter[0]}"
                counter[0] += 1
            new_op = mapping[op]
        else:
            new_op = op

        if not node.args:
            return Term(new_op, ())
        new_args = tuple(visit(a) for a in node.args)
        return Term(new_op, new_args)

    return visit(t)

def canonical_term_string(s: str) -> str:
    """
    Take a raw payload/term string and return a canonical string
    modulo:
      - AC of '.',
      - senc{X}(Y) / senc{X}((Y)) / senc{X}Y → senc{X}Y,
      - α-renaming of abstract placeholders iN/mN → v1,v2,... .
    """
    s = s.strip()
    if not s:
        return s
    # normalize whitespace around '.' etc
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\.\s*", ".", s)
    s = normalize_senc_payload(s)
    t = parse_term(s)
    t_norm = normalize_term(t)
    t_alpha = alpha_normalize_term(t_norm)  # <<< α-conversion step
    return term_to_string(t_alpha)

# ---------------- Clause-level canonicalization using AC+α ----------------

_action_re = re.compile(
    r"^(?P<label>\[[^\]]*\])?"
    r"(?P<src>[^-]+)->(?P<dst>[^(:]+)"
    r"(?:\((?P<fresh>[^)]*)\))?"
    r":(?P<payload>.+)$"
)

def canonicalize_action_line(line: str) -> str:
    """
    For Actions:
      [p] A->B(Na,Nb):payload

      - We IGNORE the fresh list in the canonical form so
        '[i3]a1->a2:senc{n3}f1(n1,n2)' and
        '[i3]a1->a2(n3):senc{n3}f1(n1,n2)' are treated as equal.

      - Payloads are fully normalized as binder-free AC terms with α-conversion:
          - '.' is AC (flatten + sort at any depth),
          - senc{X}(Y), senc{X}((Y)), senc{X}Y unify to senc{X}Y,
          - abstract placeholders iN/mN are α-renamed to v1,v2,...
    """
    m = _action_re.match(line)
    if not m:
        return line  # fallback

    label   = (m.group("label") or "").strip()
    src     = m.group("src").strip()
    dst     = m.group("dst").strip()
    # fresh is intentionally ignored in the output (AC equality on payload only)
    payload = m.group("payload").strip()

    payload_canon = canonical_term_string(payload)

    out = ""
    if label:
        out += label
    out += f"{src}->{dst}:{payload_canon}"
    return out

def canonicalize_goal_line(line: str) -> str:
    """
    For Goals:
      - If 'secret between', sort the principals after 'between'
      - Otherwise, for 'A->B:KAB.NA.NB', normalize the RHS as an AC+α term.
    """
    if "secret between" in line:
        m = re.match(r"^([a-z0-9_]+)\s+secret between\s+(.+)$", line)
        if not m:
            return line
        what = m.group(1).strip()
        who  = m.group(2).strip()
        items = [x.strip() for x in split_top_level(who, seps=",") if x.strip()]
        items = sorted(items)
        return f"{what} secret between {','.join(items)}"

    if ":" not in line:
        return line
    left, right = line.split(":", 1)
    right_canon = canonical_term_string(right)
    return f"{left}:{right_canon}"

def canonicalize_channelkey_line(line: str) -> str:
    """
    For ChannelKeys:
      - Normalize the RHS after ':' as an AC+α term
        (usually a multiset of keys, '.' is AC).
    """
    if ":" not in line:
        return line
    left, right = line.split(":", 1)
    right_canon = canonical_term_string(right)
    return f"{left}:{right_canon}"

# ---------------- New clauses_from_text with your 5 steps ----------------

def clauses_from_text(s: str) -> list[str]:
    """
    Normalization that:
      1) abstracts identifiers (except keywords) via type-based renaming,
      2) treats each section as a list,
      3) sorts sub-lists for sections except Actions/Goals/ChannelKeys,
      4) in Actions, normalizes payload as binder-free AC+α term (fresh ignored),
      5) in Goals & ChannelKeys, normalizes RHS as binder-free AC+α term,
      6) Handles empty sections correctly, e.g.:
        Public:

        Private:
            Ks, M;
    """
    # ---------- Preprocess: lowercase, strip comments, normalize per line ----------
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = strip_comments(s).lower()

    lines_raw = s.split("\n")
    norm_lines: list[str] = []

    for ln in lines_raw:
        # unify [m1]/[step1]
        ln = re.sub(r"\[\s*(?:m|step)\s*(\d+)\s*\]", r"[m\1]", ln)
        # collapse whitespace
        ln = re.sub(r"\s+", " ", ln)
        # tighten punctuation spacing (incl '.')
        ln = re.sub(r"\s*([,:;{}()\[\]\.])\s*", r"\1", ln)
        # normalize arrows
        ln = re.sub(r"\s*->\s*\*\s*", "->*", ln)
        ln = re.sub(r"\s*->\s*", "->", ln)
        ln = ln.strip()
        if ln:
            norm_lines.append(ln)

    section: str | None = None
    structured: list[tuple[str | None, str]] = []
    buffer = ""

    sec_re = re.compile(rf"^({'|'.join(HEADER_KEYS)})\s*:(.*)$")

    # ---------- First pass: detect sections & build (section, statement) list ----------
    for ln in norm_lines:
        # 'end' line
        if ln == "end":
            # flush any pending buffer as a statement (shouldn't happen in valid files)
            if buffer.strip():
                structured.append((section, buffer.strip()))
                buffer = ""
            structured.append((None, "end"))
            section = None
            break

        # Section headers (with optional same-line body)
        msec = sec_re.match(ln)
        if msec:
            # flush any previous buffered statement for old section
            if buffer.strip():
                structured.append((section, buffer.strip()))
                buffer = ""

            section = msec.group(1)
            rest = msec.group(2).strip()

            # Empty section header (e.g., "public:") → no statement yet
            if not rest:
                continue

            # Non-empty rest: treat as content for this section
            local = rest
        else:
            # Regular line within current section (or None)
            local = ln

        if not local:
            continue

        # Accumulate into buffer for this section
        if buffer:
            buffer += " " + local
        else:
            buffer = local

        # Split buffer by ';' into complete statements
        while ";" in buffer:
            before, sep, after = buffer.partition(";")
            stmt = before.strip()
            if stmt:
                structured.append((section, stmt))
            buffer = after.strip()

    # If file ended without 'end' and there's leftover buffer, flush it (defensive)
    if buffer.strip():
        structured.append((section, buffer.strip()))

    # ---------- Second pass (1): build type map from Types section ----------
    type_map: dict[str, str] = {}
    for sec, content in structured:
        if sec == "types":
            # e.g. "agent a,b,s" or "number na,nb"
            parse_types_line(content, type_map)

    # ---------- (1) abstract identifiers according to types ----------
    rename_map = build_rename_map(type_map)
    abstractify = make_abstractifier(rename_map)

    clauses: list[str] = []

    # ---------- (2)+(3): aggregate Types section as linked-lists, in order ----------
    if type_map:
        names_by_type: dict[str, list[str]] = {}
        for name, typ in type_map.items():
            names_by_type.setdefault(typ, []).append(name)

        # iterate types in insertion order (no sorted)
        for typ, names in names_by_type.items():
            seen = set()
            canon_names = []
            for n in names:
                if n in seen:
                    continue
                seen.add(n)
                canon_names.append(rename_map[n])
            clauses.append(f"types:{typ}:{','.join(canon_names)}")

    # ---------- Second pass (2)-(5): all other sections ----------
    for sec, content in structured:
        # Types already handled globally above
        if sec == "types":
            continue

        # 'end' stays as is
        if content == "end":
            clauses.append("end")
            continue

        # Abstractify identifiers in this line (once)
        line = abstractify(content, sec)

        # Outside known sections: keep as-is (e.g., protocol line)
        if sec is None or sec not in HEADER_KEYS:
            clauses.append(line)
            continue

        # Declarations: just attach section tag
        if sec == "declarations":
            clauses.append(f"{sec}:{line}")
            continue

        # Public/Private: sort the whole identifier list (empty section produces no clauses)
        if sec in ("public", "private"):
            # Guard: if line is just the header echo, skip
            if not line:
                continue
            sorted_line = sort_list_whole(line, seps=",")
            clauses.append(f"{sec}:{sorted_line}")
            continue

        # Knowledge: sort identifiers after ':'
        if sec == "knowledge":
            sorted_line = sort_list_after_colon(line, seps=",")
            clauses.append(f"{sec}:{sorted_line}")
            continue

        # Actions: normalize payload as binder-free AC+α term (fresh ignored)
        if sec == "actions":
            canon = canonicalize_action_line(line)
            clauses.append(f"{sec}:{canon}")
            continue

        # Goals: normalize RHS as binder-free AC+α term (plus 'secret between' handling)
        if sec == "goals":
            canon = canonicalize_goal_line(line)
            clauses.append(f"{sec}:{canon}")
            continue

        # ChannelKeys: normalize RHS as binder-free AC+α term
        if sec == "channelkeys":
            canon = canonicalize_channelkey_line(line)
            clauses.append(f"{sec}:{canon}")
            continue

        # Fallback (shouldn't hit, but just in case)
        clauses.append(f"{sec}:{line}")

    # ---------- Final cleanup ----------
    clauses = [c.strip() for c in clauses if c.strip()]
    return sorted(set(clauses))

def load_clauses(path: Path) -> list[str]:
    with open(path, "r", errors="ignore") as f:
        return clauses_from_text(f.read())

# ---------------- ROUGE implementations ----------------

def _lcs_length(a, b) -> int:
    """
    Longest common subsequence length between sequences a and b.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        ai = a[i]
        row = dp[i]
        row_next = dp[i + 1]
        for j in range(n):
            if ai == b[j]:
                row_next[j + 1] = row[j] + 1
            else:
                row_next[j + 1] = max(row[j + 1], row_next[j])
    return dp[m][n]

def rouge_l_f1(reference_tokens, candidate_tokens) -> float:
    """
    ROUGE-L F1 based on LCS between reference_tokens and candidate_tokens.
    """
    ref = list(reference_tokens)
    hyp = list(candidate_tokens)
    if not ref or not hyp:
        return 0.0
    lcs_len = _lcs_length(ref, hyp)
    r = lcs_len / len(ref)
    p = lcs_len / len(hyp)
    if r + p == 0:
        return 0.0
    return 2 * r * p / (r + p)

# ---------------- IO helpers ----------------
def build_name_map(d, allowed_ext={".anb"}):
    """
    Accepts str or Path; returns dict {lowercased filename -> Path}.
    Filters out AppleDouble files and .DS_Store; restricts to allowed_ext (case-insensitive).
    """
    d = Path(d)
    if not d.exists():
        raise FileNotFoundError(f"Path does not exist: {d}")
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    out = {}
    for p in d.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name == ".DS_Store" or name.startswith("._"):
            continue
        if allowed_ext and p.suffix.lower() not in allowed_ext:
            continue
        out[name.lower()] = p
    return out

# ---------------- Metrics ----------------
def compute_metrics(gt_dir, pred_dir):
    """
    Compute EC, BER, Jaccard,  ROUGE-L between GT and a prediction directory,
    using full normalization.
    Returns: df (per-file), agg (dict), details (dict)
    """
    gt_map  = build_name_map(gt_dir)
    pred_map = build_name_map(pred_dir)

    common = sorted(set(gt_map) & set(pred_map))
    if not common:
        gt_only  = sorted(set(gt_map) - set(pred_map))[:10]
        pred_only = sorted(set(pred_map) - set(gt_map))[:10]
        raise ValueError(
            "No common files found between GT and prediction.\n"
            f"GT dir: {gt_dir} ({len(gt_map)} files)\n"
            f"Pred dir: {pred_dir} ({len(pred_map)} files)\n"
            f"Examples GT-only: {gt_only}\n"
            f"Examples Pred-only: {pred_only}\n"
            "Check directory paths and filename normalization."
        )

    rows, details = [], {}
    for key in common:
        gt = set(load_clauses(gt_map[key]))
        pr = set(load_clauses(pred_map[key]))
        tp = len(gt & pr)
        fn = len(gt - pr)
        fp = len(pr - gt)
        gtn, pn = len(gt), len(pr)
        ec  = tp / gtn if gtn else np.nan
        ber = (fp + fn) / (gtn + pn) if (gtn + pn) else np.nan
        jac = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan

        #  ROUGE on tokenized normalized clauses
        gt_tokens = " ".join(sorted(gt)).split()
        pr_tokens = " ".join(sorted(pr)).split()
        rougeL = rouge_l_f1(gt_tokens, pr_tokens)

        rows.append({
            "file": key,
            "gt_clauses": gtn, "pred_clauses": pn,
            "TP": tp, "FN": fn, "FP": fp,
            "ExactCoverage_EC": ec,
            "BoundedErrorRate_BER": ber,
            "JaccardIndex": jac,
            "ROUGE_L": rougeL,
        })
        details[key] = {
            "gt_only": sorted(gt - pr),
            "pred_only": sorted(pr - gt),
            "intersection": sorted(gt & pr),
        }
######Debug################################
        if 1 or key == "signed_dh.anb":
            print("\n>>>",key)
            for i in details[key]:
                print(f"\n{i}: {details[key][i]}")
            print("TP:",tp)
            print("FN:",fn)
            print("FP:",fp)
            print("|gt|",len(gt))
            print("|pr|",len(pr))
            print("ExactCoverage_EC:",ec)
            print("ROUGE_L:", rougeL)
        #     # exit()

    cols = ["file","gt_clauses","pred_clauses","TP","FN","FP",
            "ExactCoverage_EC","BoundedErrorRate_BER","JaccardIndex","ROUGE_L"]
    df = pd.DataFrame(rows, columns=cols).sort_values("ExactCoverage_EC",ascending=False).reset_index(drop=True)

    total_tp = int(df["TP"].sum())
    total_fn = int(df["FN"].sum())
    total_fp = int(df["FP"].sum())
    total_gt = int(df["gt_clauses"].sum())
    total_pr = int(df["pred_clauses"].sum())

    agg = {
        "num_files": len(common),
        "EC_weighted_by_GT": float(total_tp / total_gt) if total_gt else np.nan,
        "EC_macro_avg": float(df["ExactCoverage_EC"].mean()),
        "BER_weighted": float((total_fp + total_fn) / (total_gt + total_pr)) if (total_gt + total_pr) else np.nan,
        "BER_macro_avg": float(df["BoundedErrorRate_BER"].mean()),
        "Jaccard_weighted": float(total_tp / (total_tp + total_fp + total_fn)) if (total_tp + total_fp + total_fn) else np.nan,
        "Jaccard_macro": float(df["JaccardIndex"].mean()),
        "ROUGE_L_macro_avg": float(df["ROUGE_L"].mean()),
    }
    return df, agg, details

if __name__ == "__main__":
    WORK = Path("work_ec_ber_new_repro")
    WORK.mkdir(exist_ok=True)

    ## set up the path to folders
    GT = Path("../dataset/anb")
    BD = Path("benchmark/outputResults_gemini-2.5-pro_gpt5.1_run3/beforeDebate/")
    AD = Path("benchmark/outputResults_gemini-2.5-pro_gpt5.1_run3/afterDebate/")
    # BD = Path("benchmark/outputResults_gpt4.1_gpt5.1_run3/beforeDebate/")
    # AD = Path("benchmark/outputResults_gpt4.1_gpt5.1_run3/afterDebate/")
    # BD = Path("benchmark/outputResults_k2-think_gpt5.1_run3/beforeDebate/")
    # AD = Path("benchmark/outputResults_k2-think_gpt5.1_run3/afterDebate/")
    # BD = Path("benchmark/outputResults_gpt-5-mini_gpt5.1_run3/beforeDebate/")
    # AD = Path("benchmark/outputResults_gpt-5-mini_gpt5.1_run3/afterDebate/")

    bd_df, bd_agg, bd_details = compute_metrics(GT, BD)
    ad_df, ad_agg, ad_details = compute_metrics(GT, AD)

    print("First Output without Debate:\n", bd_df.to_string(), "\n")
    print(bd_agg)
    print("---------------")
    print("\nFinal Output after Debate:\n", ad_df.to_string(), "\n")
    print("\n", ad_agg)