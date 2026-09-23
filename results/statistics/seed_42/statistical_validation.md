# Statistical validation

Rows: **1050**

## Bootstrap Macro-F1

- `qwen35_peft`: 0.825107 [0.801774, 0.847264]
- `gemma4_peft`: 0.822152 [0.798705, 0.844273]

## Paired differences

- `qwen35_peft__minus__gemma4_peft`: 0.002954 [-0.014315, 0.020128], bootstrap p=0.7448

## Exact McNemar tests

- `qwen35_peft__vs__gemma4_peft`: b=45, c=43, p=0.915187
