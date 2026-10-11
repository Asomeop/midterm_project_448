"""Collect every usable RQ3 example from the whole dataset (RQ3, step 4).

Usage (from the repo root):
    python rq3/collect_full.py

Same filters as export_splits.py (English, strictly single-turn, 50 to 800 word
responses, same three LLMs) but with no per-LLM cap: it streams the full
dataset once and keeps every example, labeled with its domain.

Writes rq3/data/pool_full.csv and prints counts per (LLM, domain). Use it with:
    python rq3/run_rq3.py
"""

import os

import pandas as pd
from datasets import load_dataset

from collect_pool import (DATA_DIR, MAX_WORDS, MIN_WORDS, TARGET_MODELS,
                          assign_domain, extract_strict_single_turn, get_flags)

OUT_PATH = os.path.join(DATA_DIR, "pool_full.csv")
SAVE_EVERY = 20000  # rows streamed between partial saves, so a dropped connection loses little


def save(rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    return df


def main():
    rows = []
    dataset = load_dataset(
        "lmarena-ai/arena-human-preference-140k", split="train", streaming=True
    )

    n_seen = 0
    for n_seen, row in enumerate(dataset, start=1):
        if n_seen % SAVE_EVERY == 0:
            df = save(rows)
            print(n_seen, df.groupby("domain")["LLM_name"].value_counts().unstack(fill_value=0)
                  .loc[["code", "creative", "math"]].to_dict("index"), flush=True)

        if row.get("language") != "en":
            continue

        for m_id in TARGET_MODELS:
            if row.get("model_a") != m_id and row.get("model_b") != m_id:
                continue
            prompt, resp = extract_strict_single_turn(row, m_id)
            if not prompt or not resp:
                continue
            if not (MIN_WORDS <= len(resp.split()) <= MAX_WORDS):
                continue

            is_code, creative, math = get_flags(row)
            rows.append(
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

    df = save(rows)
    print(f"\nStreamed {n_seen} rows. Wrote {len(df)} examples to {OUT_PATH}\n")

    order = ["code", "creative", "math", "other", "multi", "unknown"]
    table = pd.crosstab(df["LLM_name"], df["domain"]).reindex(columns=order, fill_value=0)
    table["total"] = table.sum(axis=1)
    print(table.to_string())

    dupes = df.duplicated(subset=["LLM_name", "LLM_Input", "LLM_output"]).sum()
    print(f"\nExact duplicate (LLM, prompt, response) rows: {dupes}")


if __name__ == "__main__":
    main()
