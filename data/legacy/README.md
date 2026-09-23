# Legacy Data

This directory documents the quarantined historical 14,000-row evaluation artifact.

## Finding

The legacy 14,000-row file labeled as a test set was found to contain:

- 14,000 rows
- 7,000 unique IDs
- every ID appearing exactly twice
- rows exactly reproducing the 7,000 released labeled examples

This artifact is **not an independent test set** and must not be used for evaluation.

## Why quarantined

The artifact does not represent held-out data. Every row duplicates a released training triplet, so using it for evaluation would measure memorization rather than generalization.
