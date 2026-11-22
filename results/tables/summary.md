## Completion and certification

| Family | Cases | Updates | SMT done | SMT partial | SMT solved upd. | Exact done | Anytime done | Anytime cert. % |
|---|---|---|---|---|---|---|---|---|
| Comprehensive | 13 | 30 | 13 | 0 | 30 | 13 | 13 | 100.0 |
| Object attributes | 8 | 40 | 8 | 0 | 40 | 8 | 8 | 100.0 |
| Scalability | 62 | 1,087 | 30 | 30 | 594 | 61 | 62 | 100.0 |
| Order components | 34 | 904 | 16 | 18 | 587 | 34 | 34 | 100.0 |
| Controlled mutations | 216 | 6,049 | 87 | 129 | 3,855 | 182 | 216 | 92.5 |
| Multi-deviation | 76 | 2,180 | 18 | 58 | 1,377 | 41 | 76 | 80.0 |
| Overall | 409 | 10,290 | 172 | 235 | 6,483 | 339 | 409 | 91.3 |

## Running time

| Family | SMT med. s | Exact med. s | Anytime med. s | Exact p95 s | Anytime p95 s | Exact speed-up | n | Anytime speed-up | n |
|---|---|---|---|---|---|---|---|---|---|
| Comprehensive | 0.23 | 0.10 | 0.10 | 1.24 | 0.76 | 2.6x | 13 | 2.7x | 13 |
| Object attributes | 0.46 | 0.19 | 0.19 | 1.22 | 1.21 | 2.2x | 8 | 2.1x | 8 |
| Scalability | 0.56 | 0.08 | 0.08 | 0.85 | 0.85 | 2.2x | 30 | 2.2x | 30 |
| Order components | 1.37 | 0.03 | 0.03 | 0.13 | 0.12 | 80.6x | 16 | 80.9x | 16 |
| Controlled mutations | 1.37 | 0.03 | 0.03 | 0.15 | 2.09 | 78.8x | 87 | 85.8x | 84 |
| Multi-deviation | 1.24 | 0.03 | 0.04 | 0.33 | 100.0 | 36.3x | 18 | 62.3x | 17 |
| Overall | 1.26 | 0.03 | 0.04 | 0.36 | 2.94 | 36.3x | 172 | 41.5x | 168 |

## Correctness and gaps

| Family | Exact checked | Anytime checked | Uncertified | Median gap | Max gap | Optimum known | Cost optimal |
|---|---|---|---|---|---|---|---|
| Comprehensive | 30 | 30 | 0 | -- | -- | 0 | 0 |
| Object attributes | 40 | 40 | 0 | -- | -- | 0 | 0 |
| Scalability | 594 | 594 | 0 | -- | -- | 0 | 0 |
| Order components | 587 | 587 | 0 | -- | -- | 0 | 0 |
| Controlled mutations | 3,539 | 3,850 | 456 | 10 | 303 | 5 | 0 |
| Multi-deviation | 1,090 | 1,375 | 436 | 8 | 282 | 2 | 0 |
| Overall | 5,880 | 6,476 | 892 | 10 | 303 | 7 | 0 |

## Certification routes (overall)

| Route | Exact % | Anytime % |
|---|---|---|
| Direct continuation | 95.9 | 87.7 |
| Local repair | 3.2 | 2.7 |
| Joint lower bound | 0.6 | 0.6 |
| Checkpoint-relative search | 0.0 | 0.0 |
| Full-prefix search | 0.4 | 0.1 |
| SMT check | 0.0 | 0.1 |
| Uncertified (deadline) | 0.0 | 8.7 |
