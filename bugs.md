# Known Bugs and Issues

Found during code/data review on 2026-10-09. None of these have been fixed yet;
existing files are unchanged.

## 1. Bugs (will break or corrupt a run)

### 1.1 `pad_token` and `unk_token` are both the empty string — `dataset.py:24-25`
```python
self.pad_token = ""
self.unk_token = ""
```
Both special tokens are the same key, so in
`self.word2idx = {self.pad_token: 0, self.unk_token: 1}` the second entry
overwrites the first and `word2idx` ends up as `{"": 1}` (one entry, not two).

Consequence: real words are assigned indices `2 .. n+1`, but
`len(vocab)` returns `n + 1` instead of `n + 2`. The model builds
`nn.Embedding(len(vocab), ...)`, whose valid indices are `0 .. n`, so the last
(least frequent) vocabulary word, index `n+1`, is out of range. Affects both the CNN and the RNN.

Observed behavior (verified 2026-10-09, torch 2.8.0):
- **CPU:** crashes with `IndexError: index out of range in self` when a batch contains that word.
- **MPS (Apple GPU):** **no error.** The out-of-range lookup silently returns a
  vector (zeros in our test) instead of failing. This is undefined behavior.
  `train_rnn.py` ran to completion on MPS with this bug present
  (`results/rnn_LLM_output_seed42`, `vocab_size: 19999`).
- With the current `train.csv`, the out-of-range word is `alleviate`: it appears
  once in the first 384 tokens of train outputs and never in val/test, so its
  effect on that run's results is likely negligible. Runs should still be
  repeated after the fix.
- **With `VOCAB_SIZE = 250`** (the RNN's new default, see NOTES.md) the impact is much
  larger: the out-of-range word is `import`, which appears 173 times in train, 11 in
  val, and 15 in test (first 384 tokens). On CPU this crashes; on MPS those tokens are
  silently corrupted. The fix is needed before final RNN results are produced.

Fix: use distinct strings, e.g. `"<pad>"` and `"<unk>"`.

### 1.2 Environment cannot run training — `venv/`
The project `venv` (Python 3.9.6) has `datasets`, `pandas`, `numpy`, but **not**
`torch` or `scikit-learn`, so `train_cnn.py` fails at import. There was no
`requirements.txt` (one has now been added).

## 2. Data issues (do not crash, but affect results / interpretation)

### 2.1 58% of responses are truncated at `MAX_LEN = 384` tokens
With the `simple_tokenizer` in `dataset.py`, `LLM_output` lengths are:
median 473 tokens, 90th percentile 1086, max 3934. 57.9% of outputs exceed
384 tokens and lose their endings (where sign-offs / follow-up offers often appear,
which may be a useful fingerprint). Worth reporting, and worth one experiment
with a longer `MAX_LEN`.

### 2.2 Duplicate prompts across classes and splits — `export_splits.py`
Arena rows are head-to-head battles, so when two target models appear in the same
battle, the same prompt is saved once for each model.
- 57 distinct prompts appear more than once (117 rows total).
- 56 of them are paired with **different** LLM labels.
- 27 of them span **more than one split** (train/val/test).

Impact on RQ2 "input only": identical inputs with different labels are
unlearnable, and the same prompt in train and test is leakage. Small relative to
2,100 rows, but should be either deduplicated or disclosed in the report.

### 2.3 Very long prompts
`LLM_Input` is short in general (median 18 tokens, 90th percentile 130) but the
max is 30,020 tokens. Needs a sensible `max_len` for the input-only and
input+output settings.

## 3. Missing pieces / limitations (not bugs)

- **No random seeds** set in `train_cnn.py` (`torch`, `numpy`, `random`), so runs
  are not reproducible.
- **Results only printed to stdout.** No metrics, confusion matrices, training
  curves, or configs are saved to files.
- **`LLMDataset` reads a single text column**, so the RQ2 "input + output"
  setting is not supported yet.
- **No task/domain label** in the CSVs, which RQ3 requires.
- `export_splits.py` depends on the HF streaming order of
  `lmarena-ai/arena-human-preference-140k`; the exact rows selected may differ if
  the dataset is updated. The committed CSVs are the canonical copy.
