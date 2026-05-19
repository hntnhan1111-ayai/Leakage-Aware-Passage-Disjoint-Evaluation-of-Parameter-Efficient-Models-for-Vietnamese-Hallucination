# Open Tasks

## P0

* [ ] Push `submit/icit2026-evidence-revision` after GitHub authentication. Impact: high.
* [ ] Resolve or replace the leaking public train/test evaluation split. Impact: critical.
* [ ] Run `PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh` on the RTX4090 uv environment after the dataset issue is resolved. Impact: high.
* [ ] Run the target RTX4090 pipeline and generate predictions plus reviewer-ready evidence. Impact: high.
* [ ] Verify generated outputs on the target machine. Impact: high.

## P1

* [ ] Update `materials/paper.tex` from generated evidence only. Impact: high.
* [ ] Add or refresh the results table, confusion matrix reference, error analysis, and reproducibility appendix. Impact: medium.
* [ ] Decide whether to archive legacy root-level scripts after target validation. Impact: medium.

## P2

* [ ] Add modern baselines beyond the minimum rescue set. Impact: low.
* [ ] Add verifier-assisted hallucination detection variants if the pipeline is stable. Impact: low.
* [ ] Expand calibration and runtime reporting beyond the current evidence pack. Impact: low.
