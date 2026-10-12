# RNN Project Notes

Decisions, problems, and observations, for the report's training-details and
lessons-learned sections. Newest at the bottom.

## 2026-10-09: Overfitting in first LSTM run (hidden 128)

**Run:** unidirectional LSTM, embed 128, hidden 128, dropout 0.5, Adam lr 1e-3,
15 epochs, MPS, seed 42. Ran with the `dataset.py` pad/unk bug still present;
effect likely negligible. Its results folder was later overwritten
by the hidden-64 run (see correction below); the numbers here were recorded from
its `history.json` / `metrics.json` before that.

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---|---|---|---|---|
| 1 | 1.013 | 47.9 | 0.938 | 57.5 |
| 4 | 0.627 | 75.5 | **0.679** (best) | 73.7 |
| 10 | 0.158 | 95.2 | 0.725 | 77.1 |
| 15 | 0.041 | 99.0 | 1.100 | 74.9 |

Test accuracy (epoch-4 checkpoint): 69.5%.

**Observation:** the model memorizes the training set after ~epoch 4; validation
loss rises from there while validation accuracy plateaus around 70-77%.

**Likely cause:** 2.56M of the model's 2.69M parameters (95%) are in the embedding
table (20,000 words x 128 dims), trained on only 1,470 examples, so the model can
memorize rare words per training example.

**Change:** `HIDDEN_DIM` 128 -> 64. Chosen because it is an RNN-only setting (the CNN
has no equivalent), so it is not a deviation from the CNN setup. Expected effect is
small: the LSTM layer is <5% of parameters and does not touch the embedding table.

## 2026-10-09: Hidden 64 run, and a correction

**Correction:** commit `3217454` describes `results/rnn_LLM_output_seed42` as the
hidden-128 run. It is actually the **hidden-64** run (`config.json`:
`"hidden_dim": 64`); the run finished before the commit was made.

**Result:** best val loss 0.689 at epoch 7, train acc 98.0% by epoch 15,
test accuracy 75.6%. Overfitting essentially unchanged (best val loss 0.689 vs
0.679), as expected since the embedding table was untouched. The higher test
accuracy (75.6 vs 69.5) comes from a different stopping epoch and is within
run-to-run noise for a 315-example test set.

## 2026-10-09: Anti-overfitting comparison

Added flags to `train_rnn.py` (`--vocab_size`, `--embed_dim`, `--embed_dropout`,
`--weight_decay`, `--tag`) and an `embed_dropout` option to `model_rnn.py`. All default
to the previous behavior. The optimizer is now AdamW, which is mathematically
identical to Adam when `weight_decay=0`.

Each candidate was changed alone, with 2 seeds (42, 43), compared on best
validation loss (not test). These runs used an in-memory fix of the
`dataset.py` pad/unk bug (the file itself was not changed), because with a
5,000-word vocabulary the out-of-range word is a common one and MPS silently
corrupts it, which would make the comparison unfair.

Results folders: `results/rnn_LLM_output_seed{42,43}_x_<config>/`. All runs: hidden 64,
dropout 0.5, lr 1e-3, 15 epochs, MPS, `LLM_output`. Mean of 2 seeds; "best" = epoch
with lowest val loss (the checkpoint that gets tested). Test accuracy shown for
completeness only; it was **not** used to choose anything.

| Config | Change | Best val loss (per seed) | Val acc @best | Train-val acc gap @best | Test acc |
|---|---|---|---|---|---|
| base | none (vocab 20000) | 0.691 (0.707, 0.674) | 74.0 | 13.2 | 75.2 |
| e64 | embed dim 64 | 0.698 (0.731, 0.665) | 75.2 | 9.0 | 75.7 |
| edrop | embed dropout 0.3 | 0.700 (0.727, 0.673) | 72.7 | 4.0 | 72.2 |
| wd | AdamW weight decay 0.1 | 0.653 (0.687, 0.618) | 75.2 | 10.6 | 70.5 |
| v5kwd | vocab 5000 + wd 0.1 | 0.651 (0.620, 0.683) | 74.4 | 8.1 | 74.4 |
| v5k | vocab 5000 | 0.666 (0.656, 0.675) | 75.1 | 12.9 | 77.0 |
| v2k | vocab 2000 | 0.601 (0.591, 0.610) | 78.6 | 12.2 | 81.6 |
| v1k | vocab 1000 | 0.534 (0.541, 0.527) | 79.8 | 8.8 | 81.1 |
| v500 | vocab 500 | 0.507 (0.493, 0.521) | 81.7 | 3.3 | 82.4 |
| **v250** | **vocab 250** | **0.461 (0.456, 0.465)** | **83.3** | 3.6 | 84.4 |
| v100 | vocab 100 | 0.474 (0.488, 0.459) | 82.7 | 3.6 | 84.3 |
| v50 | vocab 50 | 0.519 (0.516, 0.522) | 81.0 | -1.1 | 80.2 |

Epoch-15 val loss (how much it degrades when training continues): base 0.95 / 0.85,
v250 0.51 / 0.50.

**Findings**
- Embedding dropout, smaller embedding dim, and weight decay changed little;
  differences were within seed-to-seed noise (~±0.03 val loss).
- Vocabulary size was the dominant factor. Val loss fell steadily from 20000 to
  250 words, flattened at 100-250, and got worse at 50 (too little information).
- At 250 words the train-val gap fell from ~13 points to ~4, val loss stops rising
  sharply after the best epoch, and val accuracy rose ~9 points.
- The top-250 tokens are dominated by markdown and punctuation (`* - . , : ( ) " `` ` ``
  `# | = { }`, em/en dashes, curly quotes, `✅`) plus function words (`the and you
  your can if like`). Removing rarer, topic-specific words removes what the model
  was memorizing and keeps style/formatting signals. Suggests the LSTM's fingerprint
  is largely **structural/stylistic**, which is relevant evidence for RQ4.
- Side observation: some non-Latin characters (Tamil, Devanagari, full-width
  punctuation) appear in the top 250 despite the dataset's `language == "en"` filter.

**Decision:** default `VOCAB_SIZE` in `train_rnn.py` changed 20000 -> 250. This is a
**deviation from the CNN** (which uses 20000) and must be stated in the report,
justified by the table above. Either the CNN should be re-run with 250 too, or the
report should present the two models as tuned separately.

**Caveats:** 2 seeds per config; the gap between 100 and 250 is within noise. Runs used
the in-memory pad/unk fix, so a run of `train_rnn.py` with the unfixed `dataset.py`
will not match exactly. The hidden 64 / vocab 20000 run in `results/rnn_LLM_output_seed42/`
used the unfixed `dataset.py` and is not part of this comparison.
