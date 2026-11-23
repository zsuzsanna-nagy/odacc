## Completion and certification

| Family | Cases | Updates | SMT done | SMT partial | SMT solved upd. | Exact done | Anytime done | Anytime cert. % |
|---|---|---|---|---|---|---|---|---|
| Comprehensive | 13 | 30 | 13 | 0 | 30 | 13 | 13 | 100.0 |
| Object attributes | 8 | 40 | 8 | 0 | 40 | 8 | 8 | 100.0 |
| Scalability | 62 | 1,087 | 30 | 30 | 594 | 62 | 62 | 100.0 |
| Order components | 34 | 904 | 16 | 18 | 587 | 34 | 34 | 100.0 |
| Controlled mutations | 216 | 6,049 | 87 | 129 | 3,855 | 182 | 216 | 92.4 |
| Multi-deviation | 76 | 2,180 | 18 | 58 | 1,377 | 40 | 76 | 79.2 |
| Overall | 409 | 10,290 | 172 | 235 | 6,483 | 339 | 409 | 91.1 |

## Running time

| Family | SMT med. s | Exact med. s | Anytime med. s | Exact p95 s | Anytime p95 s | Exact speed-up | n | Anytime speed-up | n |
|---|---|---|---|---|---|---|---|---|---|
| Comprehensive | 0.23 | 0.10 | 0.10 | 1.21 | 0.70 | 2.5x | 13 | 2.9x | 13 |
| Object attributes | 0.46 | 0.20 | 0.20 | 1.25 | 1.17 | 2.1x | 8 | 2.2x | 8 |
| Scalability | 0.56 | 0.08 | 0.08 | 0.85 | 0.85 | 2.1x | 30 | 2.2x | 30 |
| Order components | 1.37 | 0.03 | 0.03 | 0.13 | 0.13 | 76.4x | 16 | 78.9x | 16 |
| Controlled mutations | 1.37 | 0.03 | 0.03 | 0.17 | 0.88 | 24.3x | 87 | 73.2x | 82 |
| Multi-deviation | 1.24 | 0.03 | 0.05 | 1.27 | 4.19 | 7.5x | 18 | 49.0x | 16 |
| Overall | 1.26 | 0.03 | 0.04 | 0.45 | 1.12 | 12.4x | 172 | 26.3x | 165 |

## Correctness and gaps

| Family | Exact checked | Anytime checked | Uncertified | Median gap | Max gap | Optimum known | Cost optimal |
|---|---|---|---|---|---|---|---|
| Comprehensive | 30 | 30 | 0 | -- | -- | 0 | 0 |
| Object attributes | 40 | 40 | 0 | -- | -- | 0 | 0 |
| Scalability | 594 | 594 | 0 | -- | -- | 0 | 0 |
| Order components | 587 | 587 | 0 | -- | -- | 0 | 0 |
| Controlled mutations | 3,539 | 3,847 | 461 | 10 | 373 | 8 | 1 |
| Multi-deviation | 1,090 | 1,370 | 453 | 8 | 373 | 7 | 4 |
| Overall | 5,880 | 6,468 | 914 | 8 | 373 | 15 | 5 |

## Certification routes (overall)

| Route | Exact % | Anytime % |
|---|---|---|
| Direct continuation | 95.9 | 87.6 |
| Local repair | 2.4 | 2.1 |
| Joint lower bound | 1.3 | 1.3 |
| Checkpoint-relative search | 0.0 | 0.0 |
| Full-prefix search | 0.4 | 0.0 |
| SMT check | 0.0 | 0.1 |
| Uncertified (deadline) | 0.0 | 8.9 |
