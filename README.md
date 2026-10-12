# Who Wrote It? Identifying LLMs from Their Responses

**Team Lead:** Jacob Meert  
Kasra Ghadimi  
Jason McKinlay  
Bryan Miller  

CMPSC 448 midterm project. Given a piece of text, we try to tell which LLM wrote it:
Claude, Gemini, or ChatGPT. We train a CNN and an RNN (LSTM) on responses from the
LMArena human preference dataset and use them to answer four questions:

1. Can the model be identified from its response alone?
2. Does seeing the user's prompt help?
3. Does a classifier trained on one kind of task still work on another?
4. What actually makes the models different from each other?

## Setup

```
pip install -r requirements.txt
```

All scripts are run from the repo root.

## What's in here

**Data and shared code**
- `export_splits.py` pulls the data and writes `train.csv`, `val.csv`, `test.csv` (700 responses per LLM, split 70/15/15).
- `dataset.py` is the tokenizer, vocabulary and data loaders that everything else uses.
- `model_cnn.py`, `model_rnn.py` are the two models.
- `train_cnn.py`, `train_rnn.py` train and test them.

**Experiments**
- `rq2/` compares prompt only, response only, and both together. The response-only runs are also our main RQ1 results. See `rq2/RQ2_NOTES.md`.
- `rq3/` trains on coding responses and tests on creative writing and math. See `rq3/README.md`.
- `style_profiler.py` measures style features (length, formatting, hedging words and so on) and fits a simple classifier on them for RQ4.
- `error_analysis/` looks at the individual test responses the models got wrong.

**Results**
- `canonical_benchmarks.md` has the final numbers for every experiment in one place.
- `results/` holds the RNN runs, including the vocabulary size sweep described in `NOTES.md`.
- `rq2/results/` and `rq3/results/` hold the saved metrics for those experiments.

**Other**
- `NOTES.md` is a running log of decisions and problems from building the RNN.
- `ai_use.md` describes how we used AI tools.
