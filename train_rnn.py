"""Train and evaluate the LSTM classifier (RNN).

Usage:
    python train_rnn.py                            # RQ1 / RQ2 output-only
    python train_rnn.py --text_column LLM_Input    # RQ2 input-only
    python train_rnn.py --seed 1 --device cpu

Saves to results/rnn_<text_column>_seed<seed>/:
    config.json   hyperparameters, device, vocab size, dataset sizes
    history.json  per-epoch train/val loss and accuracy
    metrics.json  test loss, accuracy, per-class report, confusion matrix
    best.pt       checkpoint with the lowest validation loss
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dataset import LABEL_MAP, get_dataloaders
from model_rnn import LSTMClassifier

# Shared settings match train_cnn.py so CNN vs RNN differences come from the model, not the setup
BATCH_SIZE = 32
MAX_LEN = 384        # tokens kept per text; longer texts are truncated (bugs.md 2.1)
VOCAB_SIZE = 20000
EMBED_DIM = 128
HIDDEN_DIM = 64      # size of the LSTM memory vector; was 128, reduced for overfitting (NOTES.md)
DROPOUT = 0.5
LR = 1e-3
EPOCHS = 15
CLIP_NORM = 1.0      # RNN-only: gradients through 384 time steps can explode; cap their norm


def set_seed(seed):
    # Fixes weight init and shuffle order so a run can be reproduced exactly
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name):
    # "auto" prefers Apple GPU (MPS), then CUDA (e.g. Colab), then CPU
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()  # enables dropout
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        # Must sit between backward() (computes gradients) and step() (applies them)
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        # Weight by batch size so the epoch average is per example (last batch may be smaller)
        total_loss += loss.item() * len(y)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += len(y)

    return total_loss / total, (correct / total) * 100.0


@torch.no_grad()  # no gradients needed for evaluation: faster, less memory
def evaluate(model, loader, criterion, device):
    model.eval()  # disables dropout
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * len(y)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    return total_loss / total, (correct / total) * 100.0, all_preds, all_labels


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    # LLM_output = RQ1 and RQ2 output-only; LLM_Input = RQ2 input-only.
    # RQ2 input+output is not possible yet: dataset.py reads a single column.
    parser.add_argument("--text_column", default="LLM_output",
                        choices=["LLM_output", "LLM_Input"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    # One folder per setting + seed so runs never overwrite each other
    out_dir = os.path.join("results", f"rnn_{args.text_column}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best.pt")
    print(f"Device: {device} | Text column: {args.text_column} | Output: {out_dir}")

    # Vocabulary is built from the training split only, so val/test words stay unseen
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        text_column=args.text_column,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        max_vocab_size=VOCAB_SIZE,
    )

    # NOTE: this assumes the pad/unk token bug in dataset.py is fixed (bugs.md 1.1).
    # While pad_token and unk_token are both "", len(vocab) is one too small, so the
    # largest token id exceeds the embedding table and training crashes with an
    # index-out-of-range error. That bug must be fixed in dataset.py before running.
    model = LSTMClassifier(
        vocab_size=len(vocab),
        num_classes=len(LABEL_MAP),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        pad_idx=vocab.pad_idx,
    ).to(device)

    # Cross-entropy applies softmax internally, so the model outputs raw logits
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Saved before training so the setup is recorded even if a run crashes
    save_json({
        "model": "LSTMClassifier (1-layer, unidirectional)",
        "text_column": args.text_column,
        "seed": args.seed,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN,
        "max_vocab_size": VOCAB_SIZE,
        "vocab_size": len(vocab),
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "lr": LR,
        "optimizer": "Adam",
        "epochs": EPOCHS,
        "clip_norm": CLIP_NORM,
        "num_params": sum(p.numel() for p in model.parameters()),
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset),
    }, os.path.join(out_dir, "config.json"))

    best_val_loss = float("inf")
    history = []  # per-epoch curves, used later for training-curve figures

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        msg = (
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )
        # Early-stopping by checkpoint: keep the epoch with the lowest val loss,
        # so later overfitting epochs don't affect the test result. [*] marks a save.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            msg += " [*]"
        print(msg)
        # Rewritten every epoch so the curves survive even if a run is stopped early
        save_json(history, os.path.join(out_dir, "history.json"))

    # Test set is used exactly once, on the best-val checkpoint, never for model selection
    print(f"\nLoading {ckpt_path} for test evaluation...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    # Class names ordered by label id so report/matrix rows line up with LABEL_MAP
    target_names = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n")
    print(classification_report(labels, preds, target_names=target_names, digits=4))
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix (rows = true, cols = predicted):")
    print(cm)

    save_json({
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "labels": target_names,
        "classification_report": classification_report(
            labels, preds, target_names=target_names, digits=4, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }, os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
