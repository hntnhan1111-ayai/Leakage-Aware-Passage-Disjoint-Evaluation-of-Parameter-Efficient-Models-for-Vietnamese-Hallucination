# Qualitative error review queue

Model: `qwen35_peft`
Seed: `42`
Total errors: **185**
Sampled for manual review: **50**

The `suggested_category` column is heuristic. Before reporting categories in the paper, manually fill `human_category`, `review_notes`, and `label_quality_flag` for every selected row.

| error_pair          |   count |
|:--------------------|--------:|
| extrinsic→intrinsic |      52 |
| extrinsic→no        |      17 |
| intrinsic→extrinsic |      49 |
| intrinsic→no        |      19 |
| no→extrinsic        |      26 |
| no→intrinsic        |      22 |
