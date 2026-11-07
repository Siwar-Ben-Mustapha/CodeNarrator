### Protocol A: 30 epochs, patience 5  (3 seeds)

**test, familiar families**

| model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | Exact match | Entity recall |
|---|---|---|---|---|---|---|
| attention | 99.8 ± 0.0 | 99.7 ± 0.0 | 99.8 ± 0.0 | 99.6 ± 0.0 | 98.9 ± 0.0 | 100.0 ± 0.0 |
| baseline | 72.8 ± 38.7 | 63.6 ± 47.8 | 71.8 ± 40.4 | 61.3 ± 47.3 | 45.6 ± 39.5 | 56.0 ± 45.9 |

**test, novel identifiers**

| model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | Exact match | Entity recall |
|---|---|---|---|---|---|---|
| attention | 99.6 ± 0.0 | 99.2 ± 0.1 | 99.6 ± 0.0 | 98.8 ± 0.1 | 95.4 ± 0.2 | 97.8 ± 0.2 |
| baseline | 74.6 ± 38.5 | 65.8 ± 49.3 | 73.3 ± 40.6 | 64.2 ± 49.4 | 52.3 ± 44.7 | 64.0 ± 44.6 |

**test, UNSEEN families (zero-shot)**

| model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | Exact match | Entity recall |
|---|---|---|---|---|---|---|
| attention | 35.2 ± 4.4 | 20.3 ± 3.4 | 33.8 ± 4.2 | 15.8 ± 4.2 | 0.0 ± 0.0 | 16.5 ± 8.9 |
| baseline | 27.7 ± 4.9 | 8.8 ± 2.8 | 24.2 ± 3.7 | 6.4 ± 3.1 | 0.0 ± 0.0 | 21.3 ± 1.8 |

Exact match on the familiar test set per seed:

| model | 42 | 43 | 44 |
|---|---|---|---|
| attention | 98.9 | 98.9 | 98.8 |
| baseline | 68.6 | 0.0 | 68.1 |

### Protocol B: 60 epochs, patience 12  (3 seeds)

**test, familiar families**

| model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | Exact match | Entity recall |
|---|---|---|---|---|---|---|
| attention | 99.8 ± 0.0 | 99.7 ± 0.0 | 99.8 ± 0.0 | 99.6 ± 0.0 | 98.9 ± 0.0 | 100.0 ± 0.0 |
| baseline | 96.5 ± 0.8 | 93.5 ± 1.4 | 96.5 ± 0.8 | 91.3 ± 1.7 | 74.3 ± 5.0 | 87.0 ± 3.1 |

**test, novel identifiers**

| model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | Exact match | Entity recall |
|---|---|---|---|---|---|---|
| attention | 99.7 ± 0.0 | 99.4 ± 0.1 | 99.7 ± 0.0 | 99.1 ± 0.2 | 96.3 ± 0.7 | 98.2 ± 0.3 |
| baseline | 97.5 ± 0.2 | 95.7 ± 0.2 | 97.5 ± 0.2 | 94.5 ± 0.4 | 83.3 ± 2.4 | 91.7 ± 0.9 |

**test, UNSEEN families (zero-shot)**

| model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | Exact match | Entity recall |
|---|---|---|---|---|---|---|
| attention | 36.4 ± 2.7 | 21.2 ± 1.9 | 35.7 ± 3.0 | 15.6 ± 0.9 | 0.0 ± 0.0 | 16.8 ± 2.1 |
| baseline | 28.4 ± 1.9 | 9.6 ± 0.6 | 24.5 ± 1.1 | 8.3 ± 2.4 | 0.0 ± 0.0 | 18.1 ± 6.3 |

Exact match on the familiar test set per seed:

| model | 42 | 43 | 44 |
|---|---|---|---|
| attention | 98.9 | 98.9 | 98.9 |
| baseline | 79.9 | 70.6 | 72.3 |
