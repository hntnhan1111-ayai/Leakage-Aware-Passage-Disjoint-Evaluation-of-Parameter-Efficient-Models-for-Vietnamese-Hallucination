# Research Direction

## Must-Have

* Keep the rescue workflow evidence-first and reproducible.
* Run one target-machine baseline path cleanly before adding anything new.
* Add calibration reporting for the existing prediction outputs.
* Add bootstrap confidence intervals for the main metrics.
* Add runtime benchmarking for latency and memory.
* Avoid overclaiming novelty in the paper and README.

## Optional

* Add one modern prompt-only baseline such as Qwen3-4B-Instruct-2507 on the target machine.
* Compare the current pipeline against encoder baselines already listed in the repo.
* Add verifier-assisted hallucination detection as a small post-processing stage, not a full rewrite.
* Expand calibration plots and error slicing if target evidence is stable.

## Specific Upgrade Areas

* Modern baselines: keep to a small number of additional models and only after the core evidence path is stable.
* Verifier-assisted hallucination detection: use a lightweight verifier for suspicious cases, not a broad architecture rewrite.
* Calibration: report reliability, confidence distribution, and threshold behavior.
* Bootstrap confidence intervals: estimate uncertainty for macro-F1, weighted-F1, and accuracy.
* Runtime benchmarking: measure end-to-end latency, per-sample latency, and memory use.

## Constraints

* Do not propose large-scale refactors or new training loops unless the target evidence path is already stable.
* Do not expand the scope faster than the ICIT 2026 timeline can absorb.
* Keep claims grounded in generated artifacts only.
