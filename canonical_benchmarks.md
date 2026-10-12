# Canonical benchmarks

One set of numbers for the whole project. Every value below is read from a saved results file;
nothing was retrained to produce this document. All runs use the same splits
(1,470 train / 315 validation / 315 test, balanced across the three LLMs), 384-token sequences,
batch size 32, learning rate 1e-3 and 15 epochs, with the checkpoint of lowest validation loss tested.
Chance accuracy is 33.3%.

| | CNN | RNN |
|---|---|---|
| Architecture | TextCNN, 100 filters each of widths 3, 4, 5 | 1-layer unidirectional LSTM, hidden size 64 |
| Vocabulary cap | 20,000 | 250 |
| Embedding size / dropout | 128 / 0.5 | 128 / 0.5 |
| Optimizer | Adam | AdamW, weight decay 0, gradient clipping 1.0 |
| Parameters | 2,714,803 | 81,859 |

Standard deviations are sample standard deviations (n - 1) over seeds 42, 43, 44 unless noted.

## 1. Headline (RQ1): identifying the LLM from its response

Source: `rq2/results/<model>_output_seed<seed>/` (the output-only setting, which is the RQ1 task).

| Metric | CNN | RNN |
|---|---|---|
| Test accuracy (%) | 94.29 ± 0.32 | 85.82 ± 0.73 |
| Macro F1 (%) | 94.29 ± 0.32 | 85.82 ± 0.72 |
| Test loss | 0.1474 ± 0.0032 | 0.4251 ± 0.0255 |
| Best validation loss | 0.1508 ± 0.0059 | 0.4506 ± 0.0159 |
| Validation accuracy at best epoch (%) | 94.92 ± 0.55 | 83.39 ± 1.11 |
| Training accuracy at best epoch (%) | 95.71 ± 0.14 | 87.82 ± 1.83 |
| Best epoch | 14.3 ± 0.6 | 12.0 ± 2.0 |

Per LLM (mean of 3 seeds, %):

| Model | LLM | Precision | Recall | F1 |
|---|---|---|---|---|
| CNN | Claude | 92.22 | 93.97 | 93.08 |
| CNN | Gemini | 94.81 | 92.70 | 93.74 |
| CNN | ChatGPT | 95.89 | 96.19 | 96.04 |
| RNN | Claude | 83.86 | 87.30 | 85.54 |
| RNN | Gemini | 85.50 | 82.22 | 83.82 |
| RNN | ChatGPT | 88.28 | 87.94 | 88.08 |

Per seed:

| Model | Seed | Test acc (%) | Macro F1 (%) | Test loss | Best val loss | Best epoch | Val acc at best (%) | Train acc at best (%) |
|---|---|---|---|---|---|---|---|---|
| CNN | 42 | 94.60 | 94.61 | 0.1491 | 0.1575 | 14 | 95.24 | 95.58 |
| CNN | 43 | 93.97 | 93.97 | 0.1438 | 0.1466 | 15 | 94.29 | 95.85 |
| CNN | 44 | 94.29 | 94.28 | 0.1495 | 0.1482 | 14 | 95.24 | 95.71 |
| RNN | 42 | 85.40 | 85.42 | 0.4119 | 0.4404 | 10 | 84.44 | 86.67 |
| RNN | 43 | 86.67 | 86.65 | 0.4088 | 0.4689 | 12 | 82.22 | 86.87 |
| RNN | 44 | 85.40 | 85.38 | 0.4544 | 0.4424 | 14 | 83.49 | 89.93 |

Confusion matrices, summed over the 3 seeds (rows = true LLM, columns = predicted; 315 per row):

CNN:

| | Claude | Gemini | ChatGPT |
|---|---|---|---|
| **Claude** | 296 | 11 | 8 |
| **Gemini** | 18 | 292 | 5 |
| **ChatGPT** | 7 | 5 | 303 |

RNN:

| | Claude | Gemini | ChatGPT |
|---|---|---|---|
| **Claude** | 275 | 23 | 17 |
| **Gemini** | 36 | 259 | 20 |
| **ChatGPT** | 17 | 21 | 277 |

## 2. RQ2: does the prompt help?

Source: `rq2/results/` (18 runs; summary in `rq2/results/rq2_summary.json`).

| Model | Setting | Test accuracy (%) | Macro F1 (%) | Test loss | F1 Claude | F1 Gemini | F1 ChatGPT |
|---|---|---|---|---|---|---|---|
| CNN | Input only | 33.97 ± 4.41 | 33.28 | 1.1208 ± 0.0022 | 37.14 | 33.58 | 29.13 |
| CNN | Output only | 94.29 ± 0.32 | 94.29 | 0.1474 ± 0.0032 | 93.08 | 93.74 | 96.04 |
| CNN | Input + output | 94.92 ± 1.14 | 94.92 | 0.1704 ± 0.0167 | 94.88 | 94.65 | 95.22 |
| RNN | Input only | 35.13 ± 1.32 | 30.23 | 1.0961 ± 0.0036 | 37.08 | 30.97 | 22.64 |
| RNN | Output only | 85.82 ± 0.73 | 85.82 | 0.4251 ± 0.0255 | 85.54 | 83.82 | 88.08 |
| RNN | Input + output | 83.07 ± 1.60 | 83.06 | 0.4807 ± 0.0293 | 82.61 | 83.01 | 83.55 |

## 3. RQ3: cross-domain generalization

Source: `rq3/results/train_code/`. Both models are trained on code responses only (942 train / 201 validation),
so these in-domain scores are not comparable to the RQ1 numbers above.
Held-out sets are the full balanced ones. `rq3/results/train_code/summary.md` reports the same means with
population standard deviations (n), which are smaller; the values here use n - 1 to match the rest of this file.

| Model | Test set | n | Accuracy (%) | Macro F1 (%) | F1 Claude | F1 Gemini | F1 ChatGPT | Drop from code (pts) | Unknown tokens (%) |
|---|---|---|---|---|---|---|---|---|---|
| CNN | Code (in-domain) | 201 | 92.9 ± 1.7 | 92.9 | 91.4 | 94.1 | 93.2 | n/a | 6.6 |
| CNN | Creative (held-out) | 573 | 78.8 ± 2.0 | 78.9 | 79.6 | 76.2 | 80.9 | 14.1 | 13.6 |
| CNN | Math (held-out) | 480 | 89.7 ± 0.4 | 89.8 | 86.8 | 87.2 | 95.5 | 3.1 | 7.4 |
| RNN | Code (in-domain) | 201 | 69.3 ± 3.3 | 69.1 | 64.8 | 73.1 | 69.5 | n/a | 34.3 |
| RNN | Creative (held-out) | 573 | 64.2 ± 2.1 | 63.2 | 67.0 | 66.8 | 56.0 | 5.2 | 45.7 |
| RNN | Math (held-out) | 480 | 70.1 ± 1.6 | 70.0 | 70.0 | 71.0 | 69.1 | -0.8 | 35.3 |

## 4. Preprocessing statistics (response text)

| Quantity | Value |
|---|---|
| Response length in tokens | median 473, 90th percentile 1,086, max 3,934 |
| Responses truncated at 384 tokens | 57.9% (Claude 26.3%, Gemini 74.3%, ChatGPT 73.1%) |
| Share of response tokens kept after truncation | 56.7% |
| Distinct tokens in training responses | 39,060 (21,855 appear at least twice) |
| Unknown-token rate on test, CNN vocabulary (20,000) | 5.3% |
| Unknown-token rate on test, RNN vocabulary (250) | 37.3% |

These were measured directly from `train.csv`, `val.csv` and `test.csv` with `dataset.py`; they are not stored in a results file.

## 5. RNN vocabulary sweep (validation loss, used to choose 250)

Source: `NOTES.md` and `results/rnn_LLM_output_seed{42,43}_x_<config>/`. Mean of 2 seeds (42, 43).

| Vocabulary cap | Best validation loss | Validation accuracy at best (%) |
|---|---|---|
| 20,000 | 0.691 | 74.0 |
| 5,000 | 0.666 | 75.1 |
| 2,000 | 0.601 | 78.6 |
| 1,000 | 0.534 | 79.8 |
| 500 | 0.507 | 81.7 |
| **250** | **0.461** | **83.3** |
| 100 | 0.474 | 82.7 |
| 50 | 0.519 | 81.0 |

## 6. Numbers that exist but should not be quoted alongside the ones above

These come from single runs of the same setups. They are valid, but mixing them with the 3-seed means would give
the report two different values for the same quantity.

| Source | What it is | Test accuracy | Test loss |
|---|---|---|---|
| `results/rnn_LLM_output_seed42/` | single RQ1 RNN run (seed 42), best epoch 13, best validation loss 0.4408 | 85.40% | 0.3961 |
| `error_analysis/` | CNN retrained once (seed 42) to extract misclassified examples; 15 test errors | 95.24% | 0.1511 |
| `error_analysis/` | RNN checkpoint from the run in the row above; 46 test errors | 85.40% | not recorded |

The style-only logistic regression (RQ4) is not saved to a results file. The report text gives 68.25% test accuracy;
re-running the same classifier during the error analysis gave 68.57% (one test example different).
