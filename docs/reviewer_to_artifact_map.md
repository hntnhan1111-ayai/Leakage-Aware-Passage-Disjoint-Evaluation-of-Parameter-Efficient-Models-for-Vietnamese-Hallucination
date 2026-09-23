# Reviewer Feedback to Artifact Map

This document maps each reviewer comment to the specific artifact and action taken during the camera-ready pass.

## Reviewer #1

### Comment: Explain train/dev/test construction.

**Action**: Strengthened split rationale and exact construction description.

**Artifact**: \docs/CAMERA_READY_CONSISTENCY_AUDIT.md\, eports/dataset/clean_split_metadata.json\, eports/dataset/clean_split_validation.json\, paper camera-ready manuscript (Data and Evaluation Protocol section).

### Comment: Report validation performance.

**Action**: Existing development metrics are available in esults/models/*/dev/classification_report.csv\. These are documented in the reproducibility section. No new evaluation was performed.

### Comment: Figure/Table redundancy.

**Action**: Removed the redundant ablation Figure 2; kept the numerical ablation table (Table 9 in the camera-ready manuscript). Cross-references updated.

### Comment: Define abbreviations.

**Action**: Completed manuscript-wide abbreviation audit. All abbreviations defined at first use in the camera-ready manuscript: PEFT, LoRA, QLoRA, NF4, BF16, NFKC, TF-IDF, CI, NLI, GPU, LLM, NLL, SHA-256.

## Reviewer #2

### Comment: No architecture/loss/training-paradigm novelty.

**Action**: Reframed as methodological evaluation study throughout Abstract, Introduction, Contributions, Related Work, Discussion, Limitations, and Conclusion.

### Comment: Why repartition 7,000 examples?

**Action**: Explained passage-group independence objective and clarified this is not an official test split. Wording: \The purpose was not to define a new official ViHallu test set, but to construct train, development, and held-out partitions in which examples derived from the same normalized source passage cannot cross partition boundaries.
### Comment: Other hallucination paradigms not evaluated.

**Action**: Strengthened Related Work and Limitations sections. Discussed NLI-based factual consistency, claim-level verification, chain-of-verification, and black-box/self-consistency hallucination detection as outside the controlled scope.

### Comment: Need qualitative linguistic/error analysis.

**Action**: Expanded using saved examples from esults/error_analysis/qwen35_peft/seed_42/error_review_queue.csv\ and \suggested_error_category_counts.csv\. Table 10 (qualitative error categories) is now explicitly referenced.

## Reviewer #3

### Comment: Abstract reduce numerical clutter.

**Action**: Rewrote Abstract to focus on problem, leakage concern, protocol, evaluated families, one concise main finding, and reproducibility contribution.

### Comment: Novelty reframing.

**Action**: Reframed methodology throughout. No new experiments.

### Comment: Split selection explanation.

**Action**: Added group construction, 20,000 deterministic assignments, class-balance criterion, timing before model training, and frozen test isolation. Wording from frozen config: \Base seed: 42; selected trial: 11031; effective seed: 11073.
### Comment: Baseline comparison remove broad LLM-vs-encoder claims.

**Action**: Narrowed claims to configuration-specific comparisons. Listed confounds: architecture, scale, tokenizer, pretraining, context length, objective, adaptation strategy. Acknowledged seed asymmetry.

### Comment: Table 10 reference.

**Action**: Added explicit reference to the qualitative error table in the Discussion section.

### Comment: Code link.

**Action**: Repository URL \https://github.com/hntnhan1111-ayai/Leakage-Aware-Passage-Disjoint-Evaluation-of-Parameter-Efficient-Models-for-Vietnamese-Hallucination.git\ added to reproducibility section.

## Reviewer #4

### Comment: Random-row-split comparison.

**Action**: Added limitation/future-work language. Changed wording from \passage leakage inflates scores\ to ow-level partitioning permits passage overlap and therefore can compromise the interpretation of held-out generalization.\ No new experiment performed.

### Comment: Response-only distribution-shift caveat.

**Action**: Added explicit caveat: \The response-only intervention is diagnostic rather than a separately trained response-only baseline. Its degradation can reflect both removal of contextual evidence and a shift away from the input structure observed during fine-tuning.
### Comment: 0.8251 vs 0.8280.

**Action**: Resolved via artifact audit. Main result 0.8251 used as canonical. Ablation full 0.8280 presented as separate pipeline condition. See \docs/CAMERA_READY_CONSISTENCY_AUDIT.md\.

### Comment: Qwen parameter count clarification.

**Action**: Renamed column to \Loaded text-path parameters\ in resource tables. Distinguished model-family naming, full model parameters, loaded text pathway, trainable LoRA parameters, and quantized base weights.

### Comment: References cleanup.

**Action**: Improved bibliographic metadata for PhoBERT, XLM-R, LoRA, QLoRA, VnCoreNLP, Qwen3.5, and Gemma 4 citations.

### Comment: Zero-shot/NLI baseline.

**Action**: Mentioned as future work. No new model added.

## Reviewer #5

### Comment: Context-cluster bootstrap.

**Action**: Acknowledged that row-level bootstrap treats rows as resampling units although 1,050 test rows correspond to 583 source contexts. Context-cluster resampling noted as a stronger future analysis.

### Comment: Shuffled-context reinterpretation.

**Action**: Reinterpreted as context sensitivity intervention, not conventional classification: \The shuffled-context condition is treated as a context-sensitivity intervention rather than a conventional classification benchmark, because replacing the source passage may alter whether the original label remains semantically valid.
### Comment: 0.8251 vs 0.8280.

**Action**: Same resolution as Reviewer #4. See \docs/CAMERA_READY_CONSISTENCY_AUDIT.md\.

### Comment: Persistent repository.

**Action**: Public GitHub repository URL provided. No Zenodo DOI claimed.
