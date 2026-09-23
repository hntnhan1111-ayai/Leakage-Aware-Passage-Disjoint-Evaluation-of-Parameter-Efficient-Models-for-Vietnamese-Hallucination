# Dataset audit and passage-disjoint split

## P0 findings

- Source labeled rows: **7000**.
- Source labeled unique contexts: **3865**.
- Quarantined legacy test rows: **14000**.
- Quarantined legacy test unique IDs: **7000**.
- Quarantined legacy test unique triplets: **7000**.
- Every quarantined row exactly matches a labeled training triplet: **True**.
- The quarantined file must not be used for model selection or final evaluation.
- The unlabeled private file shares **1292** unique contexts with the labeled source and is retained only for optional blind inference; it is not used for metrics.

## P1 clean split

- Split objective: `0.0000069194`.
- Base seed: `42`; selected trial: `11031`; effective seed: `11073`.
- Group key: SHA-256 of normalized context.
- Exact ID, context, and triplet overlap across train/dev/test: zero.

| split | rows | unique contexts | no | intrinsic | extrinsic |
|---|---:|---:|---:|---:|---:|
| train | 4900 | 2693 | 1573 | 1714 | 1613 |
| dev | 1050 | 589 | 337 | 367 | 346 |
| test | 1050 | 583 | 335 | 367 | 348 |

## Interpretation

The clean test split is a ViHallu-derived, passage-disjoint evaluation split constructed from the released labeled data. It is not the official hidden challenge test set and must not be described as an official ViHallu leaderboard result.
