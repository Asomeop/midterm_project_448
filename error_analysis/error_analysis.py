"""Error analysis: which test responses the CNN and RNN get wrong, and why.

Usage (from the repo root):
    python error_analysis/error_analysis.py

Uses the RQ1 output-only setup (train.csv / test.csv). Loads the RNN checkpoint from
results/rnn_LLM_output_seed42/best.pt and the CNN checkpoint from best_textcnn.pt;
if the CNN checkpoint is missing it is trained first with train_cnn.py.

Writes to error_analysis/:
    predictions.csv   one row per test example: both models' predictions and
                      probabilities, length/truncation, style features, text
    summary.md        error counts and how errors relate to length and style
    examples.md       the most confident errors, with the features that make
                      each response look like the wrong LLM
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT_DIR)
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # the RQ1 scripts use paths relative to the repo root

import train_cnn  # noqa: E402
import train_rnn  # noqa: E402
from dataset import LABEL_MAP, get_dataloaders, simple_tokenizer  # noqa: E402
from model_cnn import TextCNN  # noqa: E402
from model_rnn import LSTMClassifier  # noqa: E402
from style_profiler import extract_linguistic_features  # noqa: E402

LLMS = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]
RNN_CKPT = os.path.join("results", "rnn_LLM_output_seed42", "best.pt")
CNN_CKPT = train_cnn.SAVE_PATH
DEVICE = train_cnn.device
N_EXAMPLES = 3  # confident errors shown per (true, predicted) pair in examples.md


def predict(kind):
    """Return (probabilities [n, 3], unknown-token rate per example) on the test set."""
    hp = train_cnn if kind == "cnn" else train_rnn
    _, _, test_loader, vocab = get_dataloaders(
        max_len=hp.MAX_LEN, batch_size=hp.BATCH_SIZE, max_vocab_size=hp.VOCAB_SIZE
    )
    if kind == "cnn":
        if not os.path.exists(CNN_CKPT):
            print("No CNN checkpoint found; training one with train_cnn.py (seed 42)...")
            train_rnn.set_seed(42)
            train_cnn.main()
        state = torch.load(CNN_CKPT, map_location=DEVICE)
        # Size the embedding table from the checkpoint so it loads however it was trained
        model = TextCNN(
            vocab_size=state["embedding.weight"].shape[0], embed_dim=hp.EMBED_DIM,
            num_classes=len(LLMS), filter_sizes=hp.FILTER_SIZES, num_filters=hp.NUM_FILTERS,
            dropout=hp.DROPOUT, pad_idx=vocab.pad_idx,
        )
    else:
        state = torch.load(RNN_CKPT, map_location=DEVICE)
        model = LSTMClassifier(
            vocab_size=len(vocab), num_classes=len(LLMS), embed_dim=hp.EMBED_DIM,
            hidden_dim=hp.HIDDEN_DIM, dropout=hp.DROPOUT, pad_idx=vocab.pad_idx,
        )
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            probs.append(torch.softmax(model(x.to(DEVICE)), dim=1).cpu().numpy())
    x_all = np.array(test_loader.dataset.features)
    unk = (x_all == vocab.unk_idx).sum(axis=1) / np.maximum((x_all != vocab.pad_idx).sum(axis=1), 1)
    return np.concatenate(probs), 100.0 * unk


def style_table(train_df, test_df):
    """Style features per test example, a style-only prediction, and per-LLM z-scores."""
    train_feat = extract_linguistic_features(train_df, return_raw=True)
    test_feat = extract_linguistic_features(test_df, return_raw=True)
    cols = [c for c in train_feat.columns if c != "model"]

    # Same style-only classifier as style_profiler.train_style_classifier
    scaler = StandardScaler().fit(train_feat[cols])
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(train_feat[cols]), train_feat["model"])
    style_pred = clf.predict(scaler.transform(test_feat[cols]))

    # z[llm][feature]: how far each test response is from that LLM's training average,
    # in units of that LLM's own spread
    z = {}
    for llm in LLMS:
        ref = train_feat.loc[train_feat["model"] == llm, cols]
        z[llm] = (test_feat[cols] - ref.mean()) / (ref.std() + 1e-9)
    return test_feat[cols], style_pred, z, cols


def looks_like(z, cols, i, true, pred, k=3):
    """Features where response i is far from its true LLM's norm and nearer the predicted one's."""
    rows = []
    for c in cols:
        zt, zp = z[true].iloc[i][c], z[pred].iloc[i][c]
        if abs(zt) > abs(zp):
            rows.append((abs(zt) - abs(zp), c, zt, zp))
    return sorted(rows, reverse=True)[:k]


def main():
    train_df, test_df = pd.read_csv("train.csv"), pd.read_csv("test.csv")
    true = test_df["LLM_name"].to_numpy()
    out = pd.DataFrame({"true": true})

    for kind in ("cnn", "rnn"):
        probs, unk = predict(kind)
        out[f"{kind}_pred"] = [LLMS[i] for i in probs.argmax(axis=1)]
        out[f"{kind}_conf"] = probs.max(axis=1).round(4)
        for j, llm in enumerate(LLMS):
            out[f"{kind}_p_{llm}"] = probs[:, j].round(4)
        out[f"{kind}_unk_pct"] = unk.round(1)
        out[f"{kind}_correct"] = out[f"{kind}_pred"] == out["true"]
        print(f"{kind.upper()} test accuracy: {100 * out[f'{kind}_correct'].mean():.2f}%")

    n_tok = np.array([len(simple_tokenizer(str(t))) for t in test_df["LLM_output"]])
    out["n_tokens"] = n_tok
    out["truncated"] = n_tok > train_rnn.MAX_LEN

    feats, style_pred, z, cols = style_table(train_df, test_df)
    out["style_pred"] = style_pred
    out = pd.concat([out, feats.round(3)], axis=1)
    out["LLM_Input"] = test_df["LLM_Input"]
    out["LLM_output"] = test_df["LLM_output"]
    out.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index_label="test_index")

    # ---- summary.md ----
    lines = ["# Error analysis summary", "",
             f"Test set: {len(out)} responses ({len(out) // len(LLMS)} per LLM), RQ1 output-only setup.", ""]
    both_wrong = (~out["cnn_correct"] & ~out["rnn_correct"]).sum()
    lines += [
        "## Error counts", "",
        "| | Count |", "|---|---|",
        f"| CNN errors | {(~out['cnn_correct']).sum()} |",
        f"| RNN errors | {(~out['rnn_correct']).sum()} |",
        f"| Wrong for both models | {both_wrong} |",
        f"| Wrong for CNN only | {(~out['cnn_correct'] & out['rnn_correct']).sum()} |",
        f"| Wrong for RNN only | {(out['cnn_correct'] & ~out['rnn_correct']).sum()} |",
        f"| Both wrong with the same wrong label | "
        f"{(~out['cnn_correct'] & ~out['rnn_correct'] & (out['cnn_pred'] == out['rnn_pred'])).sum()} |",
        "",
    ]
    for kind in ("cnn", "rnn"):
        ok, bad = out[out[f"{kind}_correct"]], out[~out[f"{kind}_correct"]]
        pairs = bad.groupby(["true", f"{kind}_pred"]).size().sort_values(ascending=False)
        lines += [
            f"## {kind.upper()}", "",
            "| True -> predicted | Errors |", "|---|---|",
            *[f"| {t} -> {p} | {n} |" for (t, p), n in pairs.items()], "",
            "| | Correct | Wrong |", "|---|---|---|",
            f"| Examples | {len(ok)} | {len(bad)} |",
            f"| Median length (tokens) | {ok['n_tokens'].median():.0f} | {bad['n_tokens'].median():.0f} |",
            f"| Truncated at {train_rnn.MAX_LEN} | {100 * ok['truncated'].mean():.1f}% | {100 * bad['truncated'].mean():.1f}% |",
            f"| Mean unknown tokens | {ok[f'{kind}_unk_pct'].mean():.1f}% | {bad[f'{kind}_unk_pct'].mean():.1f}% |",
            f"| Mean confidence | {ok[f'{kind}_conf'].mean():.3f} | {bad[f'{kind}_conf'].mean():.3f} |",
            f"| Style-only classifier also wrong | {100 * (ok['style_pred'] != ok['true']).mean():.1f}% | "
            f"{100 * (bad['style_pred'] != bad['true']).mean():.1f}% |",
            f"| Style-only classifier predicts the same label as the {kind.upper()} | n/a | "
            f"{100 * (bad['style_pred'] == bad[f'{kind}_pred']).mean():.1f}% |",
            "",
        ]
        # Per true LLM: do the errors sit at an unusual length for that LLM?
        lines += ["Median length (tokens) by true LLM:", "",
                  "| True LLM | Correct | Wrong |", "|---|---|---|"]
        for llm in LLMS:
            a, b = ok[ok["true"] == llm]["n_tokens"], bad[bad["true"] == llm]["n_tokens"]
            lines.append(f"| {llm} | {a.median():.0f} (n={len(a)}) | "
                         + (f"{b.median():.0f} (n={len(b)}) |" if len(b) else "none |"))
        lines.append("")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines))

    # ---- examples.md ----
    ex = ["# Most confident errors", "",
          "z = distance from an LLM's training average for that feature, in that LLM's standard deviations.",
          "Listed features are those where the response is far from its true LLM and closer to the predicted one.", ""]
    for kind in ("cnn", "rnn"):
        bad = out[~out[f"{kind}_correct"]]
        ex += [f"## {kind.upper()}", ""]
        for (t, p), grp in bad.groupby(["true", f"{kind}_pred"]):
            for i, r in grp.sort_values(f"{kind}_conf", ascending=False).head(N_EXAMPLES).iterrows():
                other = "rnn" if kind == "cnn" else "cnn"
                ex += [
                    f"### Test #{i}: true {t}, predicted {p} (confidence {r[f'{kind}_conf']:.2f})", "",
                    f"- {other.upper()} predicted {r[f'{other}_pred']}; style-only classifier predicted {r['style_pred']}",
                    f"- Length {r['n_tokens']} tokens ({'truncated' if r['truncated'] else 'not truncated'}); "
                    f"unknown tokens {r[f'{kind}_unk_pct']:.1f}%",
                ]
                for _, c, zt, zp in looks_like(z, cols, i, t, p):
                    ex.append(f"- `{c}` = {r[c]:.2f}: z = {zt:+.1f} vs {t}, {zp:+.1f} vs {p}")
                prompt = " ".join(str(r["LLM_Input"]).split())[:200]
                resp = str(r["LLM_output"])[:500]
                ex += ["", f"Prompt: {prompt}", "", "Response (first 500 characters):", "", "~~~~", resp, "~~~~", ""]
    with open(os.path.join(OUT_DIR, "examples.md"), "w") as f:
        f.write("\n".join(ex))

    print("\n".join(lines))
    print(f"Saved predictions.csv, summary.md, examples.md to {OUT_DIR}")


if __name__ == "__main__":
    main()
