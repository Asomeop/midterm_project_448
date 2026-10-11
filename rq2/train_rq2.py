"""Train and evaluate the CNN or RNN on one RQ2 setting.

Usage (from anywhere):
    python rq2/train_rq2.py --model cnn --setting input
    python rq2/train_rq2.py --model rnn --setting both --seed 43

Reuses the shared code without changing it: the models (model_cnn.py, model_rnn.py),
each model's hyperparameters and train/evaluate functions (train_cnn.py, train_rnn.py),
and the data pipeline (dataset.py, via rq2_data.py). Only the text setting changes.

Saves to rq2/results/<model>_<setting>_seed<seed>/:
    config.json, history.json, metrics.json, best.pt (gitignored)
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from rq2_data import MAX_INPUT_TOKENS, SETTINGS, get_rq2_dataloaders  # also puts repo root on sys.path

import train_cnn  # noqa: E402
import train_rnn  # noqa: E402
from dataset import LABEL_MAP  # noqa: E402
from model_cnn import TextCNN  # noqa: E402
from model_rnn import LSTMClassifier  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def build_cnn(vocab):
    c = train_cnn
    model = TextCNN(vocab_size=len(vocab), embed_dim=c.EMBED_DIM, num_classes=len(LABEL_MAP),
                    filter_sizes=c.FILTER_SIZES, num_filters=c.NUM_FILTERS,
                    dropout=c.DROPOUT, pad_idx=vocab.pad_idx)
    optimizer_fn = lambda params: torch.optim.Adam(params, lr=c.LR)
    return model, optimizer_fn


def build_rnn(vocab):
    c = train_rnn
    model = LSTMClassifier(vocab_size=len(vocab), num_classes=len(LABEL_MAP),
                           embed_dim=c.EMBED_DIM, hidden_dim=c.HIDDEN_DIM, dropout=c.DROPOUT,
                           embed_dropout=c.EMBED_DROPOUT, pad_idx=vocab.pad_idx)
    optimizer_fn = lambda params: torch.optim.AdamW(params, lr=c.LR, weight_decay=c.WEIGHT_DECAY)
    return model, optimizer_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "rnn"])
    parser.add_argument("--setting", required=True, choices=list(SETTINGS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Each model keeps its own config and train/eval functions from its training script
    cfg = train_cnn if args.model == "cnn" else train_rnn
    if args.model == "cnn":
        device = train_cnn.device  # train_cnn's functions use its module-level device
        run_epoch = train_cnn.train_epoch
        run_eval = train_cnn.evaluate
    else:
        device = train_rnn.pick_device("auto")
        run_epoch = lambda m, l, c, o: train_rnn.train_epoch(m, l, c, o, device)  # includes grad clipping
        run_eval = lambda m, l, c: train_rnn.evaluate(m, l, c, device)

    train_rnn.set_seed(args.seed)

    out_dir = os.path.join(RESULTS_DIR, f"{args.model}_{args.setting}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best.pt")
    print(f"Device: {device} | Model: {args.model} | Setting: {args.setting} | Output: {out_dir}")

    train_loader, val_loader, test_loader, vocab = get_rq2_dataloaders(
        args.setting, max_len=cfg.MAX_LEN, batch_size=cfg.BATCH_SIZE, max_vocab_size=cfg.VOCAB_SIZE)

    model, optimizer_fn = (build_cnn if args.model == "cnn" else build_rnn)(vocab)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters())

    train_rnn.save_json({
        "model": args.model,
        "setting": args.setting,
        "text": SETTINGS[args.setting] or f"LLM_Input[:{MAX_INPUT_TOKENS} tokens] <sep> LLM_output",
        "seed": args.seed,
        "device": str(device),
        "max_len": cfg.MAX_LEN,
        "max_vocab_size": cfg.VOCAB_SIZE,
        "vocab_size": len(vocab),
        "epochs": cfg.EPOCHS,
        "lr": cfg.LR,
        "num_params": sum(p.numel() for p in model.parameters()),
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset),
    }, os.path.join(out_dir, "config.json"))

    # Same checkpoint rule as both training scripts: keep the lowest val loss, test once
    best_val_loss = float("inf")
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = run_eval(model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        msg = (f"Epoch {epoch:02d}/{cfg.EPOCHS:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            msg += " [*]"
        print(msg)
        train_rnn.save_json(history, os.path.join(out_dir, "history.json"))

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc, preds, labels = run_eval(model, test_loader, criterion)

    target_names = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n")
    print(classification_report(labels, preds, target_names=target_names, digits=4))
    cm = confusion_matrix(labels, preds)
    print(cm)

    train_rnn.save_json({
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
