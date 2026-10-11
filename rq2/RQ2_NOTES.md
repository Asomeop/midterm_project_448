# RQ2 Notes: Does the user's prompt help identify the LLM?

Running log for the report's RQ2 section: what the assignment asks, how we set it
up, decisions and why, results, and interpretation. Newest entries at the bottom.

## What the assignment asks

> Compare classifiers using: Input only; Output only; Input + Output.
> i) Can the LLM be predicted from the user's prompt even before seeing its response?
> ii) Does including the input improve classification?
> iii) If input-only classification works surprisingly well, why might this happen?

## How to run

```bash
python rq2/train_rq2.py --model cnn --setting input --seed 42   # model: cnn|rnn, setting: input|output|both
python rq2/summarize_rq2.py                                      # mean ± std table over seeds 42 43 44
```

Results go to `rq2/results/<model>_<setting>_seed<seed>/` (config, per-epoch history,
test metrics) and `rq2/results/rq2_summary.json`. Logs in `rq2/results/logs/`.

**Environment used for the RQ2 runs:** Python 3.11.7 (Anaconda), torch 2.2.2,
scikit-learn 1.2.2, pandas 2.1.4, numpy 1.26.4, Apple MPS GPU. Nothing extra was
installed. These are older than `requirements.txt` (torch 2.8.0 etc.); the code runs
on both, but exact numbers may differ slightly across versions.

## Code structure (2026-10-11)

All RQ2 code lives in `rq2/` and **does not modify any shared file**. It imports:

| From | What |
|---|---|
| `dataset.py` | tokenizer, `Vocabulary`, `LABEL_MAP`, `get_dataloaders` |
| `model_cnn.py`, `model_rnn.py` | `TextCNN`, `LSTMClassifier` |
| `train_cnn.py` | CNN hyperparameters, `train_epoch`, `evaluate`, device |
| `train_rnn.py` | RNN hyperparameters, `train_epoch` (with grad clipping), `evaluate`, `set_seed`, `save_json` |

So if a teammate changes a model or its hyperparameters, RQ2 picks it up automatically.

- `rq2_data.py`: input only and output only call `dataset.get_dataloaders` unchanged,
  so they are exactly the RQ1 pipeline. Input + output is built here (see below).
- `train_rq2.py`: one training run for a given model, setting, and seed.
- `summarize_rq2.py`: aggregates the runs into the results table.

## Setup

| Setting | `--setting` | What the model sees |
|---|---|---|
| Input only | `input` | the user's prompt (`LLM_Input`) |
| Output only | `output` | the response (`LLM_output`, same as RQ1) |
| Input + Output | `both` | `prompt <sep> response` |

- **Same everything else.** Within each model the three settings use identical splits,
  hyperparameters, epochs, and checkpoint rule (lowest val loss, test evaluated once).
  Only the text changes, so differences come from the input, not the setup. The
  vocabulary is rebuilt from the training split of whichever text is used.
- **Each model keeps its own tuned settings**, taken from its training script.
  CNN: vocab 20,000, Adam. RNN: vocab 250, hidden 64, AdamW (wd 0), grad clip 1.0
  (tuned against overfitting, see NOTES.md). RQ2 compares settings *within* a model,
  so the two models don't need to match.
- **Seeds 42, 43, 44**; we report mean ± std. The test set has 315 examples
  (1 example = 0.32 points), so single runs are noisy. `train_cnn.py` sets no seed, so
  RQ2 seeds both models with `train_rnn.set_seed`.

## Implementation decisions (input + output)

- **Separator token `<sep>`** between prompt and response. Added *after* tokenizing, so
  real text can never produce it (the tokenizer would split the string `"<sep>"` into
  `<`, `sep`, `>`). It appears in every training example, so it always makes the
  vocabulary cut, even at 250 words.
- **`TokenVocabulary`** subclasses `dataset.Vocabulary` and applies the same frequency
  filter, cap, and pad/truncate rules, but takes token lists (needed to insert `<sep>`).
  The shared class is not changed.
- **Prompt capped at 128 tokens** (`MAX_INPUT_TOKENS`). Prompts are usually short (median
  18 tokens, 90th percentile 130), but 10.2% exceed 128 and the longest is 30,020.
  Without a cap, a long prompt would fill the 384-token window and the model would never
  see the response. 128 ≈ the 90th percentile, so 90% of prompts fit whole.
- **Truncation:** with the prompt added, 63.3% of examples exceed 384 tokens (vs 57.9%
  for output only), so some response text is pushed out of the window. Worth noting
  when comparing Input + Output with Output only.

## Data caveats relevant to RQ2

- **Duplicate prompts across models.** The data comes from LMArena "battles", where one
  prompt is sent to two models. When both models are in our set, the same prompt
  appears twice with different labels. 9 val rows and 17 test rows have a prompt that
  is also in train; in **25 of these 26 the train copy has a different label**. So this
  overlap cannot inflate input-only accuracy; if anything it pushes a memorizing model
  toward the wrong answer. Kept as-is (splits shared with the rest of the project) and
  disclosed here.
- **What we expect.** On LMArena the user writes the prompt *before* the two models are
  randomly assigned, so the prompt should carry no information about the model, and
  input only should be near chance (33.3%). Anything clearly above chance would point
  to a **sampling artifact**, not something the LLM did.
- **One possible artifact (noted before seeing results):** mean prompt length differs by
  label (ChatGPT 134 tokens, Claude 144, Gemini 57), although medians are similar
  (17, 21, 17). This could come from which battles each model appeared in (time period,
  availability, data-collection order in `export_splits.py`).
- **Small run-to-run nondeterminism on MPS.** Re-running with the same seed reproduced
  input-only and output-only CNN results exactly, but Input + Output differed by ~1 test
  example (seed 42: 93.65% vs 93.97%). The Apple GPU backend is not fully deterministic;
  this is far smaller than the seed-to-seed spread and changes no conclusion.

## Results (2026-10-11, all 18 runs complete)

Test accuracy, mean ± std over seeds 42/43/44 (`python rq2/summarize_rq2.py`). Chance = 33.3%.

| Model | Setting | Test acc (%) | Per seed | Macro-F1 |
|---|---|---|---|---|
| CNN | Input only | 34.0 ± 4.4 | 36.8 / 28.9 / 36.2 | 0.333 |
| CNN | Output only | 94.3 ± 0.3 | 94.6 / 94.0 / 94.3 | 0.943 |
| CNN | Input + Output | 94.9 ± 1.1 | 94.0 / 96.2 / 94.6 | 0.949 |
| RNN | Input only | 35.1 ± 1.3 | 33.7 / 35.6 / 36.2 | 0.302 |
| RNN | Output only | 85.8 ± 0.7 | 85.4 / 86.7 / 85.4 | 0.858 |
| RNN | Input + Output | 83.1 ± 1.6 | 84.8 / 82.9 / 81.6 | 0.831 |

Input-only confusion matrices, summed over the 3 seeds (rows = true Claude/Gemini/ChatGPT,
cols = predicted). Predictions are spread across all three classes, so the model is not
just collapsing to one class; it simply finds no usable signal:

```
CNN                     RNN
[[133,  91,  91],       [[145, 110,  60],
 [127, 104,  84],        [134, 118,  63],
 [118, 113,  84]]        [127, 119,  69]]
```

## Answers to the RQ2 sub-questions

**(i) Can the LLM be predicted from the prompt alone? No.** Both models stay at chance
(CNN 34.0%, RNN 35.1%, chance 33.3%), with macro-F1 ≈ 0.30-0.33 and no class predicted
reliably. The seed-to-seed spread (CNN 28.9-36.8%) is as large as any gap from chance.
This matches how the data was produced: on LMArena the user writes the prompt before the
two models are randomly assigned, so the prompt carries no information about which model
will answer.

**(ii) Does including the input improve classification? No meaningful improvement.**
CNN: 94.9 ± 1.1% with input + output vs 94.3 ± 0.3% with output only (+0.6 points, about
2 test examples), smaller than the seed-to-seed spread (seed 42 is actually lower with the
input: 94.0 vs 94.6). RNN: 83.1 ± 1.6% with input + output vs 85.8 ± 0.7% with output
only (-2.7 points), lower for every seed (84.8 vs 85.4, 82.9 vs 86.7, 81.6 vs 85.4), so for
the RNN adding the prompt slightly hurts. Since the prompt alone carries no signal, adding
it can at best give context for reading the response; in practice the identifying signal is
in the response. Possible reasons the RNN is hurt (not tested): its 250-token vocabulary is
now shared with prompt words, and the LSTM classifies from its state after the last token,
so losing more response endings to truncation may matter more for it than for the CNN. Adding the prompt also pushes more of the response past the 384-token
window (63.3% vs 57.9% of examples truncated), which could offset any small gain.

**(iii) Why might input-only work surprisingly well? It doesn't here, and the reason is
the data source.** Random model assignment after the prompt is written removes any
prompt-model link. Input-only classification could look good in other datasets if:
the prompts for each model were collected differently (different time periods, sources,
or users), prompts were written specifically for one model (e.g. "As ChatGPT…"), or the
same prompt appeared in train and test with the same label (leakage). We checked the last
one: the 26 val/test prompts that also appear in train are almost all (25/26) paired with
a *different* LLM, because they come from head-to-head battles, so they cannot inflate
input-only accuracy. The prompt-length difference noted above (Gemini's mean prompt is
shorter) did not translate into above-chance accuracy.
