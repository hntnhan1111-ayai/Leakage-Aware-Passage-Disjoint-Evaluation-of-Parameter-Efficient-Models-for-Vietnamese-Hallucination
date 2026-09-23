# Data

## Redistributed files

- \data/processed/train.csv\ - 4,900 rows (passage-disjoint training split)
- \data/processed/dev.csv\ - 1,050 rows (passage-disjoint development split)
- \data/processed/test.csv\ - 1,050 rows (passage-disjoint held-out test split)

These are derived from the 7,000 released labeled ViHallu examples using the passage-disjoint partition described in the paper.

## Not redistributed

- \data/raw/vihallu-train.csv\ - original released labeled source (7,000 rows)
- \data/raw/vihallu-test-duplicated-quarantine.csv\ - quarantined legacy 14,000-row file
- \data/raw/vihallu-private-test-unlabeled.csv\ - unlabeled challenge test set

Raw ViHallu data is not redistributed in this repository due to licensing uncertainty. Please download from the DSC2025 challenge page and follow the original terms.

## How to reconstruct

1. Download the ViHallu labeled data from the DSC2025 challenge.
2. Place it as \data/raw/vihallu-train.csv\.
3. Run \ash scripts/run_p0_data.sh\ to perform the audit and passage-disjoint split construction.
4. Verify the resulting split counts against eports/dataset/clean_split_validation.json\.
