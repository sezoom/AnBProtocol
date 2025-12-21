from pathlib import Path
import re, pandas as pd, numpy as np
import math
from collections import Counter

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
    "senc", "aenc", "hash","pk", "sk","k","g","kdf","K"
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

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok in KEYWORDS:
            return tok
        if tok in rename_map:
            return rename_map[tok]
        if tok in extra:
            return extra[tok]
        if counter[0] == 0:
            extra[tok] = "proto"
            counter[0] += 1
            return extra[tok]
        new = f"i{counter[0]}"
        counter[0] += 1
        extra[tok] = new
        return new

    def transform(text: str) -> str:
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

      - We normalize payloads of the form:
          senc{n3}(f1(n1,n2)), senc{n3}((f1(n1,n2))), senc{n3}f1(n1,n2)
        to a single canonical form:
          senc{n3}f1(n1,n2)

      - We sort top-level payload components (split by '.').
    """
    m = _action_re.match(line)
    if not m:
        return line  # fallback

    label   = (m.group("label") or "").strip()
    src     = m.group("src").strip()
    dst     = m.group("dst").strip()
    fresh   = (m.group("fresh") or "").strip()
    payload = m.group("payload").strip()

    # (1) Normalize payload around senc{...} to remove redundant parentheses
    payload = normalize_senc_payload(payload)

    # (2) (optional) still sort fresh identifiers internally,
    #     but we DO NOT include them in the output string.
    #     This makes '[i3]a1->a2:senc{n3}...' and
    #     '[i3]a1->a2(n3):senc{n3}...' identical.
    if fresh:
        f_items = [x.strip() for x in split_top_level(fresh, seps=",.") if x.strip()]
        f_items = sorted(f_items)
        fresh_sorted = ",".join(f_items)
    else:
        fresh_sorted = ""

    # (3) Sort payload at top-level by '.'
    p_items = [x.strip() for x in split_top_level(payload, seps=".") if x.strip()]
    p_items = sorted(p_items)
    payload_sorted = ".".join(p_items)

    # (4) Build canonical form (NO fresh list in header)
    out = ""
    if label:
        out += label
    out += f"{src}->{dst}"
    # We intentionally drop the '(fresh_sorted)' part here
    out += f":{payload_sorted}"
    return out

def canonicalize_goal_line(line: str) -> str:
    """
    For Goals:
      - If 'secret between', sort the principals after 'between'
      - Otherwise, for 'A->B:KAB.NA.NB', sort identifiers after ':' only
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
    items = [x.strip() for x in split_top_level(right, seps=".") if x.strip()]
    items = sorted(items)
    return f"{left}:{'.'.join(items)}"

def canonicalize_channelkey_line(line: str) -> str:
    """
    For ChannelKeys:
      - Sort identifiers after ':' only (treat ',' or '.' as separators)
        e.g., K(A,B): K1.K2  -> K(A,B):K1,K2 (sorted)
    """
    if ":" not in line:
        return line
    left, right = line.split(":", 1)
    items = [x.strip() for x in split_top_level(right, seps=",.") if x.strip()]
    items = sorted(items)
    return f"{left}:{','.join(items)}"

# ---------------- New clauses_from_text with your 5 steps ----------------

def clauses_from_text(s: str) -> list[str]:
    """
    Normalization that:
      1) abstracts identifiers (except keywords) via type-based renaming,
      2) treats each section as a list,
      3) sorts sub-lists for sections except Actions/Goals/ChannelKeys,
      4) in Actions, sorts fresh IDs and payload only,
      5) in Goals & ChannelKeys, sorts IDs after ':' only.

    IMPORTANT: Handles empty sections correctly, e.g.:

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
        line = abstractify(content)

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

        # Actions: sort fresh identifiers and payload only
        if sec == "actions":
            canon = canonicalize_action_line(line)
            clauses.append(f"{sec}:{canon}")
            continue

        # Goals: sort identifiers after ':' only (plus 'secret between' handling)
        if sec == "goals":
            canon = canonicalize_goal_line(line)
            clauses.append(f"{sec}:{canon}")
            continue

        # ChannelKeys: sort identifiers after ':' only
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
######Debug example################################
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
            # exit()

    cols = ["file","gt_clauses","pred_clauses","TP","FN","FP",
            "ExactCoverage_EC","BoundedErrorRate_BER","JaccardIndex","ROUGE_L"]
    df = pd.DataFrame(rows, columns=cols).sort_values("file").reset_index(drop=True)

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
    # BD = Path("benchmark/outputResults_gemini-2.5-pro_gpt5.1/beforeDebate/")
    # AD = Path("benchmark/outputResults_gemini-2.5-pro_gpt5.1/afterDebate/")
    # BD = Path("benchmark/outputResults_gpt4.1_gpt5.1_run3/beforeDebate/")
    # AD = Path("benchmark/outputResults_gpt4.1_gpt5.1_run3/afterDebate/")
    # BD = Path("benchmark/outputResults_k2-think_gpt5.1/beforeDebate/")
    # AD = Path("benchmark/outputResults_k2-think_gpt5.1/beforeDebate/")
    BD = Path("benchmark/outputResults_k2-think_gpt5.1_run3/beforeDebate/")
    AD = Path("benchmark/outputResults_k2-think_gpt5.1_run3/afterDebate/")
    # BD = Path("benchmark/outputResults_gpt-5-mini_gpt5.1/beforeDebate/")
    # AD = Path("benchmark/outputResults_gpt-5-mini_gpt5.1/afterDebate/")

    bd_df, bd_agg, bd_details = compute_metrics(GT, BD)
    ad_df, ad_agg, ad_details = compute_metrics(GT, AD)

    print("First Output without Debate:\n", bd_df.to_string(), "\n")
    print(bd_agg)
    print("---------------")
    print("\nFinal Output after Debate:\n", ad_df.to_string(), "\n")
    print("\n", ad_agg)