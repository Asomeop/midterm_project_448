# RNN Project Notes

Decisions, problems, and observations, for the report's training-details and
lessons-learned sections. Newest at the bottom.

## 2026-10-09: Overfitting in first LSTM run

**Run:** `results/rnn_LLM_output_seed42` (unidirectional LSTM, embed 128, hidden 128,
dropout 0.5, Adam lr 1e-3, 15 epochs, MPS). Ran with the `dataset.py` pad/unk bug
still present (bugs.md 1.1); effect likely negligible.

**Observation:** train accuracy rose to 99.0% by epoch 15 while validation loss was
lowest at epoch 4 (0.679) and rose to 1.100 by epoch 15. Validation accuracy
plateaued around 70-77%. Test accuracy (epoch-4 checkpoint): 69.5%.

**Likely cause:** 2.56M of the model's 2.69M parameters (95%) are in the embedding
table (20,000 words x 128 dims), trained on only 1,470 examples, so the model can
memorize rare words per training example.

**Change:** `HIDDEN_DIM` 128 -> 64. Chosen because it is an RNN-only setting (the CNN
has no equivalent), so it is not a deviation from the CNN setup. Expected effect is
small: the LSTM layer is <5% of parameters and does not touch the embedding table.

**Considered but not applied** (each would change a setting shared with, or absent
from, the CNN, and none is guaranteed to help): smaller vocabulary, weight decay,
smaller embedding dim, embedding dropout, pretrained embeddings, more data.
