
from pathlib import Path
import re, pandas as pd, numpy as np

# ---------------- Normalization (full) ----------------
_comment_block_re = re.compile(r"/\*.*?\*/", re.S)
_comment_line1_re = re.compile(r"//.*?$", re.M)
_comment_line2_re = re.compile(r"#.*?$", re.M)

HEADER_KEYS = (
    "declarations","types","knowledge","public","private",
    "actions","goals","channelkeys"
)

def strip_comments(s: str) -> str:
    s = _comment_block_re.sub(" ", s)
    s = _comment_line1_re.sub("", s)
    s = _comment_line2_re.sub("", s)
    return s

def canonical_whitespace_and_tokens(s: str) -> str:
    s = s.lower().replace("\r\n", "\n").replace("\r", "\n")
    # unify [m1]/[step1] (incl. spaced variants)
    s = re.sub(r"\[\s*(?:m|step)\s*(\d+)\s*\]", r"[m\1]", s)
    s = strip_comments(s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([,:;{}()\[\]])\s*", r"\1", s)  # tighten punctuation spacing
    s = re.sub(r"\s*->\s*\*\s*", "->*", s)
    s = re.sub(r"\s*->\s*", "->", s)
    s = re.sub(r"\s*\.\s*", ".", s).strip()
    return s.strip()

def clauses_from_text(s: str) -> list[str]:
    """Turn text into canonical 'clauses' to neutralize newlines/semicolons."""
    s = canonical_whitespace_and_tokens(s)
    parts = [p.strip() for p in s.split(";")]
    parts = [p for p in parts if p]
    clauses = []
    for chunk in parts:
        for piece in [pp.strip() for pp in chunk.split("\n") if pp.strip()]:
            # protocol <name>:
            mprot = re.match(r"^protocol\s+([^:]+)\s*:(.*)$", piece)
            if mprot:
                name = mprot.group(1).strip()
                tail = mprot.group(2).strip()
                clauses.append("protocol:")
                if name: clauses.append(name)
                if tail: clauses.append(tail)
                continue
            # protocol:
            mprot2 = re.match(r"^protocol\s*:(.*)$", piece)
            if mprot2:
                clauses.append("protocol:")
                tail = mprot2.group(1).strip()
                if tail: clauses.append(tail)
                continue
            # section headers (with optional same-line body)
            msec = re.match(rf"^({'|'.join(HEADER_KEYS)})\s*:(.*)$", piece)
            if msec:
                hdr = msec.group(1); body = msec.group(2).strip()
                clauses.append(f"{hdr}:")
                if body: clauses.append(body)
                continue
            # end
            if piece == "end":
                clauses.append("end"); continue
            # default
            clauses.append(piece)
    clauses = [c.strip() for c in clauses if c.strip()]
    return sorted(set(clauses))

def load_clauses(path: Path) -> list[str]:
    with open(path, "r", errors="ignore") as f:
        return clauses_from_text(f.read())

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
    Compute EC & BER between GT and a prediction directory, using full normalization.
    Returns: df (per-file), agg (dict), details (dict)
    """
    gt_map  = build_name_map(gt_dir)
    pred_map = build_name_map(pred_dir)

    common = sorted(set(gt_map) & set(pred_map))
    if not common:
        # Provide immediate, actionable diagnostics
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
        rows.append({
            "file": key,
            "gt_clauses": gtn, "pred_clauses": pn,
            "TP": tp, "FN": fn, "FP": fp,
            "ExactCoverage_EC": ec, "BoundedErrorRate_BER": ber, "JaccardIndex": jac
        })
        details[key] = {
            "gt_only": sorted(gt - pr),
            "pred_only": sorted(pr - gt),
            "intersection": sorted(gt & pr),
        }
        ######Debug################################
        if 1:
            print("\n>>>", key)
            for i in details[key]:
                print(f"\n{i}: {details[key][i]}")
            print("TP:", tp)
            print("FN:", fn)
            print("FP:", fp)
            print("|gt|", len(gt))
            print("|pr|", len(pr))
            print("ExactCoverage_EC:", ec)
        #     # exit()
    # Explicit columns prevent KeyError even if rows is empty (it won't be if we passed the check above).
    cols = ["file","gt_clauses","pred_clauses","TP","FN","FP",
            "ExactCoverage_EC","BoundedErrorRate_BER","JaccardIndex"]
    df = pd.DataFrame(rows, columns=cols).sort_values("file").reset_index(drop=True)


    # Weighted/micro aggregates
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


    print("First Output without Debate:\n",bd_df.to_string(),"\n")
    print(bd_agg)
    print("---------------")
    print("\nFinal Output after Debate:\n",ad_df.to_string(),"\n")
    print("\n",ad_agg)
