"""RQ2 data loading: input only, output only, or input + output.

Builds on the shared ../dataset.py without changing it:
- "input" and "output" call dataset.get_dataloaders directly, so they are exactly
  the pipeline used by train_cnn.py / train_rnn.py.
- "both" needs the prompt and response joined with a separator token, which
  dataset.py cannot do (it reads one column), so it is built here from the same
  tokenizer, label map, and vocabulary rules.
"""

import os
import sys
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import LABEL_MAP, Vocabulary, get_dataloaders, simple_tokenizer  # noqa: E402

# RQ2 setting -> CSV column ("both" is built from the two columns)
SETTINGS = {"input": "LLM_Input", "output": "LLM_output", "both": None}
SEP_TOKEN = "<sep>"     # added after tokenizing, so real text can never produce it
MAX_INPUT_TOKENS = 128  # cap on the prompt so a long prompt can't crowd out the response
CSV_PATHS = {split: os.path.join(ROOT, f"{split}.csv") for split in ("train", "val", "test")}


# prompt tokens (capped) + <sep> + response tokens
def combined_tokens(prompt, response, max_input_tokens=MAX_INPUT_TOKENS):
    return (simple_tokenizer(str(prompt))[:max_input_tokens]
            + [SEP_TOKEN] + simple_tokenizer(str(response)))


class TokenVocabulary(Vocabulary):
    """dataset.Vocabulary with the same rules, but fed token lists instead of raw text."""

    def build_from_tokens(self, token_lists):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        # Same filter and cap as Vocabulary.build_vocab
        words = [w for w, f in counter.most_common()
                 if f >= self.min_freq and w not in self.word2idx]
        for idx, word in enumerate(words[: self.max_size - 2], start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode_tokens(self, tokens, max_len=384):
        # Same truncate / pad rule as Vocabulary.encode
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens][:max_len]
        return ids + [self.pad_idx] * (max_len - len(ids))


class CombinedDataset(Dataset):
    """Each example is `prompt <sep> response`; labels from dataset.LABEL_MAP."""

    def __init__(self, df, vocab, max_len=384):
        self.features = [
            vocab.encode_tokens(combined_tokens(p, r), max_len)
            for p, r in zip(df["LLM_Input"], df["LLM_output"])
        ]
        self.labels = [LABEL_MAP[name] for name in df["LLM_name"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


def get_rq2_dataloaders(setting, max_len=384, batch_size=32, max_vocab_size=20000):
    if SETTINGS[setting] is not None:
        # Unchanged shared pipeline
        return get_dataloaders(
            train_csv=CSV_PATHS["train"], val_csv=CSV_PATHS["val"], test_csv=CSV_PATHS["test"],
            text_column=SETTINGS[setting], max_len=max_len,
            batch_size=batch_size, max_vocab_size=max_vocab_size,
        )

    dfs = {split: pd.read_csv(path) for split, path in CSV_PATHS.items()}
    # Vocabulary from the training split only; <sep> is in every example, so it always makes the cut
    vocab = TokenVocabulary(max_size=max_vocab_size)
    vocab.build_from_tokens(
        combined_tokens(p, r) for p, r in zip(dfs["train"]["LLM_Input"], dfs["train"]["LLM_output"])
    )
    loaders = [
        DataLoader(CombinedDataset(dfs[split], vocab, max_len),
                   batch_size=batch_size, shuffle=(split == "train"))
        for split in ("train", "val", "test")
    ]
    return (*loaders, vocab)
