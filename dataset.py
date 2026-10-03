import re
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Label mappings
LABEL_MAP = {"Claude": 0, "Gemini": 1, "ChatGPT": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# Simple regex tokenizer to lowercase and split words/punctuation
def simple_tokenizer(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+|[^\w\s]", text.lower())


class Vocabulary:

    def __init__(self, max_size=20000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.pad_token = ""
        self.unk_token = ""
        self.pad_idx = 0
        self.unk_idx = 1

        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}

    # Build vocab strictly from training set
    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenizer(text))

        # Filter by min frequency and cap at max_size
        vocab_words = [
            word
            for word, freq in counter.most_common()
            if freq >= self.min_freq and word not in self.word2idx
        ]
        vocab_words = vocab_words[: self.max_size - 2]

        for idx, word in enumerate(vocab_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    # Map words to indices, then pad or truncate
    def encode(self, text, max_len=384):
        tokens = simple_tokenizer(text)
        indices = [self.word2idx.get(t, self.unk_idx) for t in tokens]

        # Truncate or pad to fixed length
        if len(indices) > max_len:
            indices = indices[:max_len]
        elif len(indices) < max_len:
            indices = indices + [self.pad_idx] * (max_len - len(indices))

        return indices

    def __len__(self):
        return len(self.word2idx)


class LLMDataset(Dataset):

    def __init__(
        self, csv_path, vocab, text_column="LLM_output", max_len=384
    ):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.text_column = text_column
        self.max_len = max_len

        # Pre-encode tokens and labels upfront
        self.features = [
            self.vocab.encode(str(t), self.max_len)
            for t in self.df[self.text_column]
        ]
        self.labels = [LABEL_MAP[name] for name in self.df["LLM_name"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def get_dataloaders(
    train_csv="train.csv",
    val_csv="val.csv",
    test_csv="test.csv",
    text_column="LLM_output",
    max_len=384,
    batch_size=32,
    max_vocab_size=20000,
):
    # Fit vocabulary on training split only
    train_df = pd.read_csv(train_csv)
    vocab = Vocabulary(max_size=max_vocab_size)
    vocab.build_vocab(train_df[text_column].tolist())

    # Build datasets
    train_ds = LLMDataset(
        train_csv, vocab, text_column=text_column, max_len=max_len
    )
    val_ds = LLMDataset(
        val_csv, vocab, text_column=text_column, max_len=max_len
    )
    test_ds = LLMDataset(
        test_csv, vocab, text_column=text_column, max_len=max_len
    )

    # Wrap into loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    # Quick sanity check on shapes
    train_loader, val_loader, test_loader, vocab = get_dataloaders()
    x_batch, y_batch = next(iter(train_loader))
    print(x_batch.shape, y_batch.shape)