"""LSTM text classifier for identifying which LLM wrote a text.

Usage:
    from model_rnn import LSTMClassifier
    model = LSTMClassifier(vocab_size=..., num_classes=3, pad_idx=vocab.pad_idx)
    logits = model(x)   # x: [batch, seq_len] LongTensor of token ids, padded at the end

Run `python model_rnn.py` for a quick shape check.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):
    """Single-layer LSTM: tokens -> embeddings -> LSTM -> final hidden state -> logits."""

    def __init__(self, vocab_size, num_classes, embed_dim=128, hidden_dim=128,
                 dropout=0.5, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        # Map each token ID to a learned vector; padding stays a zero vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # Read the sequence left to right, one token at a time
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # Regularize, then map the final hidden state to one score per LLM
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens):
        # dataset.py pads only at the end, so real length = number of non-pad tokens
        lengths = (tokens != self.pad_idx).sum(dim=1).cpu()
        x = self.embedding(tokens)
        # Pack so the LSTM stops at each sequence's real end instead of running over padding
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n[-1] is the hidden state after the last real (non-truncated) token
        return self.fc(self.dropout(h_n[-1]))


if __name__ == "__main__":
    model = LSTMClassifier(vocab_size=20000, num_classes=3)
    x = torch.randint(1, 20000, (32, 384))
    x[:, 300:] = 0  # simulate end padding
    print(model(x).shape)
