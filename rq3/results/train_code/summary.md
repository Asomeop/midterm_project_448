# RQ3: trained on code

Mean over 3 seeds (42, 43, 44); each seed redraws the split. Chance level is 33.3% (three balanced classes).
Per seed: 942 train / 201 val examples, all from code. Held-out domains are used for testing only.
"matched" = held-out set cut to the in-domain test size; "all" = every balanced held-out example.
Acc drop = in-domain accuracy minus this row's accuracy, in percentage points.

| Model | Test set | Type | n | Acc % (±sd) | Macro F1 | F1 Claude | F1 Gemini | F1 ChatGPT | Acc drop (pts) | Unknown tokens % |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN | code | in-domain | 201 | 92.9 (±1.4) | 92.9 | 91.4 | 94.1 | 93.2 | n/a | 6.6 |
| CNN | creative_all | held-out | 573 | 78.8 (±1.7) | 78.9 | 79.6 | 76.2 | 80.9 | 14.1 | 13.6 |
| CNN | creative_matched | held-out | 201 | 78.9 (±1.3) | 79.0 | 80.3 | 76.3 | 80.3 | 13.9 | 13.2 |
| CNN | math_all | held-out | 480 | 89.7 (±0.4) | 89.8 | 86.8 | 87.2 | 95.5 | 3.1 | 7.4 |
| CNN | math_matched | held-out | 201 | 88.6 (±1.5) | 88.7 | 85.6 | 87.0 | 93.4 | 4.3 | 7.3 |
| RNN | code | in-domain | 201 | 69.3 (±2.7) | 69.1 | 64.8 | 73.1 | 69.5 | n/a | 34.3 |
| RNN | creative_all | held-out | 573 | 64.2 (±1.7) | 63.2 | 67.0 | 66.8 | 56.0 | 5.2 | 45.7 |
| RNN | creative_matched | held-out | 201 | 64.0 (±3.0) | 63.0 | 68.0 | 65.4 | 55.7 | 5.3 | 45.3 |
| RNN | math_all | held-out | 480 | 70.1 (±1.3) | 70.0 | 70.0 | 71.0 | 69.1 | -0.8 | 35.3 |
| RNN | math_matched | held-out | 201 | 68.7 (±1.8) | 68.5 | 69.4 | 70.0 | 66.3 | 0.7 | 35.2 |
