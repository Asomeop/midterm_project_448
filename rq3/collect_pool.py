"""Rebuild the RQ1 example pool with domain labels attached (RQ3, step 3).

Usage (from the repo root):
    python rq3/collect_pool.py

Streams the dataset with the same filters and order as export_splits.py, so the
2,100 examples are the same ones behind train/val/test.csv, and adds is_code,
creative_writing, math, and a mutually exclusive `domain` column.

Writes rq3/data/pool_with_domains.csv and prints counts per (LLM, domain).
Does not touch train.csv / val.csv / test.csv.
"""

import os

import pandas as pd
from datasets import load_dataset

TARGET_MODELS = {
    "claude-opus-4-20250514": "Claude",
    "gemini-2.5-flash": "Gemini",
    "chatgpt-4o-latest-20250326": "ChatGPT",
}

SAMPLES_PER_MODEL = 700
MIN_WORDS = 50
MAX_WORDS = 800
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, "data")
OUT_PATH = os.path.join(DATA_DIR, "pool_with_domains.csv")


# Copied from export_splits.py rather than imported: that script has no
# __main__ guard, so importing it would re-run it and overwrite the RQ1 CSVs.
def extract_strict_single_turn(row, model_id):
    full_conv = row.get("full_conversation")
    if isinstance(full_conv, list) and len(full_conv) > 2:
        return None, None

    meta = row.get("conv_metadata")
    if isinstance(meta, dict) and meta.get("turns", 1) > 1:
        return None, None

    if row.get("model_a") == model_id:
        conv = row.get("conversation_a")
    elif row.get("model_b") == model_id:
        conv = row.get("conversation_b")
    else:
        return None, None

    if not isinstance(conv, list) or len(conv) != 2:
        return None, None

    turn1, turn2 = conv[0], conv[1]
    if (
        not isinstance(turn1, dict)
        or not isinstance(turn2, dict)
        or turn1.get("role") != "user"
        or turn2.get("role") != "assistant"
    ):
        return None, None

    prompt = turn1.get("content")
    resp = turn2.get("content")

    if isinstance(prompt, list):
        prompt = " ".join(
            b.get("text", "") for b in prompt if isinstance(b, dict) and "text" in b
        )
    if isinstance(resp, list):
        resp = " ".join(
            b.get("text", "") for b in resp if isinstance(b, dict) and "text" in b
        )

    if isinstance(prompt, str) and isinstance(resp, str):
        return prompt.strip(), resp.strip()

    return None, None


def get_flags(row):
    """Return (is_code, creative_writing, math); None where a flag is missing."""
    tag = row.get("category_tag")
    tag = tag if isinstance(tag, dict) else {}
    creative = (tag.get("creative_writing_v0.1") or {}).get("creative_writing")
    math = (tag.get("math_v0.1") or {}).get("math")
    return row.get("is_code"), creative, math


def assign_domain(is_code, creative, math):
    # A missing flag means we cannot rule a domain in or out, so the row is unusable
    if is_code is None or creative is None or math is None:
        return "unknown"
    n_true = int(bool(is_code)) + int(bool(creative)) + int(bool(math))
    if n_true > 1:
        return "multi"  # excluded from RQ3
    if is_code:
        return "code"
    if creative:
        return "creative"
    if math:
        return "math"
    return "other"


def main():
    collected = {m: [] for m in TARGET_MODELS}
    dataset = load_dataset(
        "lmarena-ai/arena-human-preference-140k", split="train", streaming=True
    )

    for n_seen, row in enumerate(dataset, start=1):
        if all(len(collected[m]) >= SAMPLES_PER_MODEL for m in TARGET_MODELS):
            break
        if n_seen % 5000 == 0:
            print(n_seen, {TARGET_MODELS[m]: len(v) for m, v in collected.items()}, flush=True)

        if row.get("language") != "en":
            continue

        for m_id in TARGET_MODELS:
            if len(collected[m_id]) < SAMPLES_PER_MODEL and (
                row.get("model_a") == m_id or row.get("model_b") == m_id
            ):
                prompt, resp = extract_strict_single_turn(row, m_id)
                if not prompt or not resp:
                    continue
                if not (MIN_WORDS <= len(resp.split()) <= MAX_WORDS):
                    continue

                is_code, creative, math = get_flags(row)
                collected[m_id].append(
                    {
                        "LLM_name": TARGET_MODELS[m_id],
                        "LLM_Input": prompt,
                        "LLM_output": resp,
                        "is_code": is_code,
                        "creative_writing": creative,
                        "math": math,
                        "domain": assign_domain(is_code, creative, math),
                        "row_id": row.get("id"),  # battle id; shared by both sides of a battle
                    }
                )

    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame([r for rows in collected.values() for r in rows])
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_PATH}\n")

    order = ["code", "creative", "math", "other", "multi", "unknown"]
    table = pd.crosstab(df["LLM_name"], df["domain"]).reindex(columns=order, fill_value=0)
    table["total"] = table.sum(axis=1)
    print(table.to_string())

    # Raw flag overlap, to show what the "multi" bucket is made of
    print("\nFlag combinations (is_code, creative_writing, math):")
    print(df.groupby(["is_code", "creative_writing", "math"], dropna=False).size().to_string())

    # Sanity check: the pool should be exactly the examples in the RQ1 CSVs
    root = os.path.dirname(OUT_DIR)
    rq1 = pd.concat(pd.read_csv(os.path.join(root, f)) for f in ("train.csv", "val.csv", "test.csv"))
    key = lambda d: set(zip(d["LLM_name"], d["LLM_output"].astype(str).str.strip()))
    same = key(df) & key(rq1)
    print(f"\nOverlap with RQ1 CSVs: {len(same)} of {len(key(rq1))} unique RQ1 examples")

    code = df[df["domain"] == "code"]
    held = df[df["domain"].isin(["creative", "math"])]
    shared = set(code["LLM_Input"]) & set(held["LLM_Input"])
    print(f"Prompts shared between code and creative/math: {len(shared)}")
    print(f"Prompts appearing under more than one LLM: {(df.groupby('LLM_Input')['LLM_name'].nunique() > 1).sum()}")


if __name__ == "__main__":
    main()
