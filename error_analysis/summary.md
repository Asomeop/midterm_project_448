# Error analysis summary

Test set: 315 responses (105 per LLM), RQ1 output-only setup.

## Error counts

| | Count |
|---|---|
| CNN errors | 15 |
| RNN errors | 46 |
| Wrong for both models | 10 |
| Wrong for CNN only | 5 |
| Wrong for RNN only | 36 |
| Both wrong with the same wrong label | 9 |

## CNN

| True -> predicted | Errors |
|---|---|
| Gemini -> Claude | 6 |
| ChatGPT -> Claude | 3 |
| Claude -> ChatGPT | 3 |
| ChatGPT -> Gemini | 1 |
| Claude -> Gemini | 1 |
| Gemini -> ChatGPT | 1 |

| | Correct | Wrong |
|---|---|---|
| Examples | 300 | 15 |
| Median length (tokens) | 509 | 410 |
| Truncated at 384 | 59.7% | 53.3% |
| Mean unknown tokens | 5.3% | 4.1% |
| Mean confidence | 0.941 | 0.739 |
| Style-only classifier also wrong | 29.0% | 80.0% |
| Style-only classifier predicts the same label as the CNN | n/a | 53.3% |

Median length (tokens) by true LLM:

| True LLM | Correct | Wrong |
|---|---|---|
| Claude | 286 (n=101) | 670 (n=4) |
| Gemini | 676 (n=98) | 272 (n=7) |
| ChatGPT | 652 (n=101) | 330 (n=4) |

## RNN

| True -> predicted | Errors |
|---|---|
| Gemini -> Claude | 13 |
| Claude -> Gemini | 12 |
| ChatGPT -> Claude | 8 |
| ChatGPT -> Gemini | 7 |
| Claude -> ChatGPT | 3 |
| Gemini -> ChatGPT | 3 |

| | Correct | Wrong |
|---|---|---|
| Examples | 269 | 46 |
| Median length (tokens) | 505 | 534 |
| Truncated at 384 | 59.1% | 60.9% |
| Mean unknown tokens | 37.7% | 38.2% |
| Mean confidence | 0.909 | 0.692 |
| Style-only classifier also wrong | 26.4% | 60.9% |
| Style-only classifier predicts the same label as the RNN | n/a | 30.4% |

Median length (tokens) by true LLM:

| True LLM | Correct | Wrong |
|---|---|---|
| Claude | 284 (n=90) | 529 (n=15) |
| Gemini | 675 (n=89) | 532 (n=16) |
| ChatGPT | 626 (n=90) | 535 (n=15) |
