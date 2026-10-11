"""RQ3: train on one domain, test on that domain and on held-out domains.

Usage (from the repo root):
    python rq3/run_rq3.py                                   # train on code, test on creative + math
    python rq3/run_rq3.py --train_domain creative --test_domains code
    python rq3/run_rq3.py --pool rq3/data/pool_with_domains.csv --seeds 42

Needs a pool CSV with a `domain` column; the default is rq3/data/pool_full.csv
(built by rq3/collect_full.py).
Reuses the RQ1 models, training loops, and hyperparameters unchanged.

Writes to rq3/results/train_<domain>/:
    summary.md / summary.csv   mean over seeds, one row per (model, test set)
    per_seed.csv               the same metrics for every seed
    confusion_<model>.png      confusion matrices summed over seeds
    config.json                seeds, hyperparameters, split sizes
and the split CSVs to rq3/data/splits/train_<domain>/seed<seed>/.
"""

import argparse
import copy
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed; only saving PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

RQ3_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(RQ3_DIR))  # repo root, for the RQ1 modules

import train_cnn  # noqa: E402
import train_rnn  # noqa: E402
from dataset import LABEL_MAP, LLMDataset, get_dataloaders  # noqa: E402
from model_cnn import TextCNN  # noqa: E402
from model_rnn import LSTMClassifier  # noqa: E402

LLMS = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]
CHANCE = 100.0 / len(LLMS)
VAL_FRAC = TEST_FRAC = 0.15  # same 70/15/15 split as RQ1
DEVICE = train_cnn.device    # same MPS > CUDA > CPU choice as the RQ1 scripts


def make_splits(pool, train_domain, test_domains, seed, out_dir):
    """Write train/val/test CSVs for one seed and return {name: path}, sizes."""
    rng = np.random.RandomState(seed)
    dom_a = pool[pool["domain"] == train_domain]

    # Balance the training domain: every LLM is cut down to the smallest LLM's count
    n_a = dom_a["LLM_name"].value_counts().reindex(LLMS).min()
    dom_a = pd.concat(
        dom_a[dom_a["LLM_name"] == llm].sample(n=n_a, random_state=rng) for llm in LLMS
    )

    # A prompt answered by two LLMs must not sit in train and in val/test at once,
    # so every row with a repeated prompt is forced into train.
    repeated = dom_a["LLM_Input"].duplicated(keep=False)
    n_hold = int(round(n_a * TEST_FRAC))
    parts = {"train": [], "val": [], "test_" + train_domain: []}
    for llm in LLMS:
        rows = dom_a[dom_a["LLM_name"] == llm]
        free = rows[~repeated.loc[rows.index]].sample(frac=1.0, random_state=rng)
        parts["val"].append(free.iloc[:n_hold])
        parts["test_" + train_domain].append(free.iloc[n_hold: 2 * n_hold])
        parts["train"].append(pd.concat([free.iloc[2 * n_hold:], rows[repeated.loc[rows.index]]]))
    splits = {k: pd.concat(v) for k, v in parts.items()}

    train_prompts = set(splits["train"]["LLM_Input"])
    dropped = {}
    for dom in test_domains:
        dom_b = pool[pool["domain"] == dom]
        leak = dom_b["LLM_Input"].isin(train_prompts)
        dropped[dom] = int(leak.sum())
        dom_b = dom_b[~leak]
        n_b = dom_b["LLM_name"].value_counts().reindex(LLMS).min()
        # "all": every held-out example we can use while staying balanced
        all_b = pd.concat(
            dom_b[dom_b["LLM_name"] == llm].sample(n=n_b, random_state=rng) for llm in LLMS
        )
        splits[f"test_{dom}_all"] = all_b
        # "matched": cut to the in-domain test size so both tests are equally noisy
        n_m = min(n_hold, n_b)
        splits[f"test_{dom}_matched"] = pd.concat(
            all_b[all_b["LLM_name"] == llm].iloc[:n_m] for llm in LLMS
        )

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, df in splits.items():
        paths[name] = os.path.join(out_dir, name + ".csv")
        df[["LLM_name", "LLM_Input", "LLM_output", "domain"]].to_csv(paths[name], index=False)
    sizes = {name: len(df) for name, df in splits.items()}
    sizes["leaked_prompts_dropped"] = dropped
    return paths, sizes


def build_model(kind, vocab):
    """Same architectures and hyperparameters as train_cnn.py / train_rnn.py."""
    if kind == "cnn":
        model = TextCNN(
            vocab_size=len(vocab), embed_dim=train_cnn.EMBED_DIM, num_classes=len(LABEL_MAP),
            filter_sizes=train_cnn.FILTER_SIZES, num_filters=train_cnn.NUM_FILTERS,
            dropout=train_cnn.DROPOUT, pad_idx=vocab.pad_idx,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cnn.LR)
    else:
        model = LSTMClassifier(
            vocab_size=len(vocab), num_classes=len(LABEL_MAP), embed_dim=train_rnn.EMBED_DIM,
            hidden_dim=train_rnn.HIDDEN_DIM, dropout=train_rnn.DROPOUT,
            embed_dropout=train_rnn.EMBED_DROPOUT, pad_idx=vocab.pad_idx,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_rnn.LR, weight_decay=train_rnn.WEIGHT_DECAY
        )
    return model.to(DEVICE), optimizer


def unk_rate(dataset, vocab):
    """Percent of real (non-pad) tokens the model sees as unknown."""
    x = np.array(dataset.features)
    return 100.0 * (x == vocab.unk_idx).sum() / max((x != vocab.pad_idx).sum(), 1)


def run_one(kind, paths, train_domain, seed):
    """Train one model on the training domain; evaluate on every test set."""
    hp = train_cnn if kind == "cnn" else train_rnn
    train_rnn.set_seed(seed)
    in_domain = "test_" + train_domain

    # Vocabulary comes from the training split only; held-out domains never touch it
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        train_csv=paths["train"], val_csv=paths["val"], test_csv=paths[in_domain],
        max_len=hp.MAX_LEN, batch_size=hp.BATCH_SIZE, max_vocab_size=hp.VOCAB_SIZE,
    )
    model, optimizer = build_model(kind, vocab)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, hp.EPOCHS + 1):
        if kind == "cnn":
            train_cnn.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, _, _ = train_cnn.evaluate(model, val_loader, criterion)
        else:
            train_rnn.train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, _, _ = train_rnn.evaluate(model, val_loader, criterion, DEVICE)
        # Checkpoint selection uses the in-domain validation split only
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    print(f"  {kind} seed {seed}: best epoch {best_epoch}, val loss {best_val:.4f}", flush=True)

    rows, cms = [], {}
    for name, path in paths.items():
        if not name.startswith("test_"):
            continue
        ds = test_loader.dataset if name == in_domain else LLMDataset(path, vocab, max_len=hp.MAX_LEN)
        loader = DataLoader(ds, batch_size=hp.BATCH_SIZE, shuffle=False)
        if kind == "cnn":
            _, acc, preds, labels = train_cnn.evaluate(model, loader, criterion)
        else:
            _, acc, preds, labels = train_rnn.evaluate(model, loader, criterion, DEVICE)
        per_llm = f1_score(labels, preds, labels=list(range(len(LLMS))), average=None, zero_division=0)
        row = {
            "model": kind.upper(), "seed": seed, "test_set": name[len("test_"):], "n": len(ds),
            "accuracy": acc, "macro_f1": 100.0 * per_llm.mean(),
            "unk_rate": unk_rate(ds, vocab), "best_epoch": best_epoch,
        }
        row.update({f"f1_{llm}": 100.0 * f for llm, f in zip(LLMS, per_llm)})
        rows.append(row)
        cms[name[len("test_"):]] = confusion_matrix(labels, preds, labels=list(range(len(LLMS))))
    return rows, cms, len(vocab)


def plot_confusions(cms, kind, train_domain, n_seeds, path):
    """One panel per test set; cell text is the count, shade is the row share."""
    fig, axes = plt.subplots(1, len(cms), figsize=(3.4 * len(cms), 3.9), squeeze=False)
    for ax, (name, cm) in zip(axes[0], cms.items()):
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(LLMS)):
            for j in range(len(LLMS)):
                ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center",
                        fontsize=9, color="white" if share[i, j] > 0.55 else "#1a1a1a")
        acc = 100.0 * np.trace(cm) / cm.sum()
        ax.set_title(f"{name}\nacc {acc:.1f}% (n={cm.sum()})", fontsize=10)
        ax.set_xticks(range(len(LLMS)), LLMS, fontsize=9)
        ax.set_yticks(range(len(LLMS)), LLMS, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0][0].set_ylabel("True", fontsize=9)
    fig.suptitle(
        f"{kind.upper()} trained on {train_domain} (summed over {n_seeds} seeds; chance {CHANCE:.1f}%)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize(per_seed, train_domain):
    """Mean over seeds per (model, test set), plus the drop from the in-domain test."""
    metrics = ["accuracy", "macro_f1"] + [f"f1_{llm}" for llm in LLMS] + ["unk_rate"]
    in_dom = per_seed[per_seed["test_set"] == train_domain].set_index(["model", "seed"])["accuracy"]
    per_seed = per_seed.copy()
    # Drop is computed within a seed (same trained model), then averaged
    per_seed["acc_drop"] = [
        in_dom[(m, s)] - a for m, s, a in zip(per_seed["model"], per_seed["seed"], per_seed["accuracy"])
    ]
    grouped = per_seed.groupby(["model", "test_set"], sort=False)
    out = grouped[metrics + ["acc_drop"]].mean()
    out.insert(0, "n", grouped["n"].first())
    out.insert(2, "accuracy_std", grouped["accuracy"].std(ddof=0))
    out = out.reset_index()
    out.insert(2, "type", ["in-domain" if t == train_domain else "held-out" for t in out["test_set"]])
    return out.round(1)


def to_markdown(summary, train_domain, seeds, sizes):
    head = ["Model", "Test set", "Type", "n", "Acc % (±sd)", "Macro F1"] + \
           [f"F1 {llm}" for llm in LLMS] + ["Acc drop (pts)", "Unknown tokens %"]
    lines = [
        f"# RQ3: trained on {train_domain}",
        "",
        f"Mean over {len(seeds)} seeds ({', '.join(map(str, seeds))}); each seed redraws the split. "
        f"Chance level is {CHANCE:.1f}% (three balanced classes).",
        f"Per seed: {sizes['train']} train / {sizes['val']} val examples, all from {train_domain}. "
        "Held-out domains are used for testing only.",
        "\"matched\" = held-out set cut to the in-domain test size; \"all\" = every balanced held-out example.",
        "Acc drop = in-domain accuracy minus this row's accuracy, in percentage points.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for _, r in summary.iterrows():
        cells = [r["model"], r["test_set"], r["type"], str(r["n"]),
                 f"{r['accuracy']:.1f} (±{r['accuracy_std']:.1f})", f"{r['macro_f1']:.1f}"]
        cells += [f"{r[f'f1_{llm}']:.1f}" for llm in LLMS]
        cells += ["n/a" if r["type"] == "in-domain" else f"{r['acc_drop']:.1f}", f"{r['unk_rate']:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=os.path.join(RQ3_DIR, "data", "pool_full.csv"))
    parser.add_argument("--train_domain", default="code")
    parser.add_argument("--test_domains", nargs="+", default=["creative", "math"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    pool = pd.read_csv(args.pool)
    tag = f"train_{args.train_domain}"
    out_dir = os.path.join(RQ3_DIR, "results", tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {DEVICE} | pool: {args.pool} | seeds: {args.seeds}")
    print(pd.crosstab(pool["LLM_name"], pool["domain"]).to_string(), "\n")

    all_rows, cm_sum, sizes_by_seed, vocab_sizes = [], {"cnn": {}, "rnn": {}}, {}, {}
    for seed in args.seeds:
        paths, sizes = make_splits(
            pool, args.train_domain, args.test_domains, seed,
            os.path.join(RQ3_DIR, "data", "splits", tag, f"seed{seed}"),
        )
        sizes_by_seed[seed] = sizes
        print(f"seed {seed} split sizes: {sizes}")
        for kind in ("cnn", "rnn"):
            rows, cms, vocab_sizes[kind] = run_one(kind, paths, args.train_domain, seed)
            all_rows.extend(rows)
            for name, cm in cms.items():
                cm_sum[kind][name] = cm_sum[kind].get(name, 0) + cm

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(os.path.join(out_dir, "per_seed.csv"), index=False)
    summary = summarize(per_seed, args.train_domain)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    md = to_markdown(summary, args.train_domain, args.seeds, sizes_by_seed[args.seeds[0]])
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)
    for kind in ("cnn", "rnn"):
        plot_confusions(cm_sum[kind], kind, args.train_domain, len(args.seeds),
                        os.path.join(out_dir, f"confusion_{kind}.png"))

    hp_names = ["BATCH_SIZE", "MAX_LEN", "VOCAB_SIZE", "EMBED_DIM", "DROPOUT", "LR", "EPOCHS"]
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "pool": os.path.relpath(args.pool, os.path.dirname(RQ3_DIR)),
            "train_domain": args.train_domain, "test_domains": args.test_domains,
            "seeds": args.seeds, "device": str(DEVICE), "chance_accuracy": CHANCE,
            "cnn": {**{k: getattr(train_cnn, k) for k in hp_names},
                    "NUM_FILTERS": train_cnn.NUM_FILTERS, "FILTER_SIZES": train_cnn.FILTER_SIZES,
                    "vocab_size_actual": vocab_sizes["cnn"]},
            "rnn": {**{k: getattr(train_rnn, k) for k in hp_names},
                    "HIDDEN_DIM": train_rnn.HIDDEN_DIM, "CLIP_NORM": train_rnn.CLIP_NORM,
                    "vocab_size_actual": vocab_sizes["rnn"]},
            "split_sizes_by_seed": sizes_by_seed,
        }, f, indent=2)

    print("\n" + md)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
