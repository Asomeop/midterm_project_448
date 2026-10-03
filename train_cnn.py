import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dataset import LABEL_MAP, get_dataloaders
from model_cnn import TextCNN

# Training config
BATCH_SIZE = 32
MAX_LEN = 384
VOCAB_SIZE = 20000
EMBED_DIM = 128
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
DROPOUT = 0.5
LR = 1e-3
EPOCHS = 15
SAVE_PATH = "best_textcnn.pt"

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / total, (correct / total) * 100.0


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
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


def main():
    print(f"Device: {device}")
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        max_vocab_size=VOCAB_SIZE,
    )

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_classes=len(LABEL_MAP),
        filter_sizes=FILTER_SIZES,
        num_filters=NUM_FILTERS,
        dropout=DROPOUT,
        pad_idx=vocab.pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        msg = (
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            msg += " [*]"

        print(msg)

    # Evaluate on test set with best checkpoint
    print(f"\nLoading {SAVE_PATH} for test evaluation...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))

    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n")

    # Class breakdown
    target_names = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
    print(classification_report(labels, preds, target_names=target_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()