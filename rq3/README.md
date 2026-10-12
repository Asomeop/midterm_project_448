# RQ3: Cross-domain generalization

We train the CNN and the RNN on **code** responses only, then test them on unseen code
(in-domain) and on **creative writing** and **math** (held-out, never seen in training).

## Result

Mean of 3 seeds; chance is 33.3%.

| Model | Code (in-domain) | Creative (held-out) | Math (held-out) |
|---|---|---|---|
| CNN | 92.9% | 78.8% | 89.7% |
| RNN | 69.3% | 64.2% | 70.1% |

Full table: [`results/train_code/summary.md`](results/train_code/summary.md).
Confusion matrices: `results/train_code/confusion_cnn.png`, `confusion_rnn.png`.

## Files

| Path | What it is |
|---|---|
| `collect_pool.py` | adds domain labels to the RQ1 pool (used for counting) |
| `collect_full.py` | collects every usable example from the whole dataset |
| `run_rq3.py` | builds splits, trains both models, evaluates, saves results |
| `data/pool_full.csv` | the 8,482 labeled examples used for the results |
| `data/splits/` | the train/val/test files for each seed |
| `results/train_code/` | summary table, per-seed metrics, confusion matrices, config |

## Reproducing

From the repository root:

```
python rq3/collect_full.py    # optional; data/pool_full.csv is already included
python rq3/run_rq3.py
```
