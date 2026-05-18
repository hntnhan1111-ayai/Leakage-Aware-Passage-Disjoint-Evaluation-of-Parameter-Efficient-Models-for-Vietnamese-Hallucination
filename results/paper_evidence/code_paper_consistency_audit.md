# Code-Paper Consistency Audit

All file references use exact repository paths and line numbers.

## Hardcoded Hugging Face Token Patterns

* None found.

## LoRA Rank Claims and Config Values

* `docs/codex_memory_bank.md:18: * Do not claim `r=64` unless reproduced.`
* `docs/codex_session_memory.md:40: * /mnt/d/uit-paper/uit_r64.py is named as an r=64 variant but its LORA_CONFIG sets r to 128.`
* `materials/paper.tex:49: As Large Language Models (LLMs) are increasingly deployed in low-resource environments like Vietnam, ensuring their reliability against hallucinations remains a critical challenge. Leveraging the rigorous \textbf{DSC2025-ViHallu benchmark}, which establishes a standardized framework for detecting intrinsic and extrinsic inconsistencies, we propose \textbf{JAMP} (\textbf{J}accard-\textbf{A}ugmented \textbf{M}ulti-\textbf{P}rompting), a robust detection framework built upon the Vistral-7B-Chat backbone. Rather than pursuing complexity for marginal gains, our study focuses on identifying the optimal configuration for Vietnamese hallucination detection through a rigorous ablation analysis. We introduce a synergistic approach that combines \textit{generative reasoning} with \textit{explicit lexical anchoring} via Jaccard similarity scores, fine-tuned efficiently using QLoRA ($r=64$) with label smoothing. Our experimental breakdown demonstrates that injecting lexical signals serves as a crucial grounding mechanism, boosting performance stability compared to purely generative baselines. Furthermore, we provide empirical evidence that a multi-prompt ensemble strategy—synthesizing analytical, questioning, and classification perspectives—effectively mitigates the variance inherent in single-prompt reasoning. Achieving a Macro-F1 of \textbf{0.8320} on the official ViHallu test set, our work offers a transparent, explainable, and highly effective recipe for building safer LLM systems in Vietnamese.`
* `materials/paper.tex:69: \item We provide a detailed \textbf{ablation study} specific to the Vietnamese context, demonstrating that parameter-efficient fine-tuning via QLoRA \cite{dettmers2024qlora} with lower ranks ($r=64$) and label smoothing outperforms higher-capacity configurations ($r=128$), challenging the assumption that "bigger is better" for specific downstream tasks.`
* `materials/paper.tex:116: All experiments were conducted on a single \textbf{NVIDIA RTX 4090 (24GB)} GPU. The Vistral-7B-Chat model was loaded in 4-bit precision (NF4). For QLoRA fine-tuning, we used a rank $r=64$, alpha $\alpha=128$, and a learning rate of $1e-4$. The training process utilized a label smoothing factor of $0.1$ to mitigate overconfidence.`
* `materials/paper.tex:143: \item \textbf{Rank Efficiency:} Interestingly, increasing the LoRA rank to $r=128$ slightly degraded performance (0.8318) compared to $r=64$. This suggests that a lower-rank adaptation acts as a regularizer, preventing the model from overfitting to the training noise.`
* `materials/paper.tex:159: With LoRA Rank $r=128$ & 0.8318 & -0.0002 \\`
* `materials/paper.tex:190: \subsection{Efficiency vs. Capacity: The $r=64$ Paradox}`
* `materials/paper.tex:191: A key finding in our ablation study is that a lower LoRA rank ($r=64$) outperformed a higher rank ($r=128$). In traditional deep learning, increasing capacity typically leads to better performance; however, for the specific task of Vietnamese hallucination detection, we argue that lower-rank adapters act as a form of "bottleneck regularization." Since the ViHallu dataset contains high-quality but finite samples, a larger rank may allow the model to memorize specific linguistic patterns (overfitting) rather than learning the general logic of contradiction. Our results suggest that for specialized Vietnamese safety tasks, "slimmer" adapters provide better generalization on unseen private test data.`
* `materials/paper.tex:201: In this paper, we introduced \textbf{JAMP}, a robust framework for detecting hallucinations in Vietnamese LLMs, evaluated on the rigorous DSC2025-ViHallu benchmark. Our methodology moves beyond simple discriminative classification by synergizing the generative reasoning of Vistral-7B with explicit lexical anchoring via Jaccard similarity scores. Through a comprehensive ablation study, we identified an optimal configuration using QLoRA ($r=64$) and a multi-perspective prompt ensemble, achieving a state-of-the-art \textbf{Macro-F1 of 0.8320} on the private test set. Our findings suggest that for low-resource languages, grounding LLM reasoning in statistical lexical signals is a critical requirement for ensuring faithfulness.`
* `scripts/uit_nojaccard.py:70: 'r': 128,`
* `scripts/uit_nolabelsmo.py:141: 'r': 128,`
* `scripts/uit_nomulti-prompts.py:69: 'r': 128,`
* `scripts/uit_our_method.py:71: 'r': 128,`
* `scripts/uit_r64.py:71: 'r': 128,`
* `scripts/update_paper_from_evidence.py:24: rank_note = "Audited implementation evidence indicates LoRA rank r=128 in the current scripts; r=64 claims must not be used unless reproduced by a separate run."`
* `scripts/update_paper_from_evidence.py:25: if "r=64" in audit and "'r': 128" not in audit:`
* `scripts/update_paper_from_evidence.py:26: rank_note = "The audited evidence did not find an r=128 implementation conflict."`
* `scripts/update_paper_from_evidence.py:62: text = re.sub(r"QLoRA \(\$r=64\$\)", "QLoRA (audited current scripts use $r=128$)", text)`
* `uit_nojaccard.py:70: 'r': 128,`
* `uit_nolabelsmo.py:141: 'r': 128,`
* `uit_nomulti-prompts.py:69: 'r': 128,`
* `uit_our_method.py:71: 'r': 128,`
* `uit_r64.py:71: 'r': 128,`

## Evaluation Sampling Fractions

* `docs/codex_session_memory.md:41: * Multiple half-evaluation scripts use sample(frac=0.5).`
* `scripts/generate_predictions_current_best.py:143: df = df.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)`
* `scripts/uit_RoBERTa_best_half.py:128: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `scripts/uit_RoBERTa_best_half_aggressive.py:136: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `scripts/uit_RoBERTa_best_half_nostop.py:88: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `scripts/uit_RoBERTa_best_half_partial.py:124: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `scripts/uit_nolabelsmo.py:481: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `scripts/uit_our_method.py:236: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `scripts/uit_phobert_best_dotproduct.py:141: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `scripts/uit_phobert_best_half.py:215: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `scripts/uit_phobert_best_notopk.py:108: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `scripts/uit_phobert_best_top1.py:136: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `uit_RoBERTa_best_half.py:128: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `uit_RoBERTa_best_half_aggressive.py:136: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `uit_RoBERTa_best_half_nostop.py:88: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `uit_RoBERTa_best_half_partial.py:124: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `uit_nolabelsmo.py:481: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `uit_our_method.py:236: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `uit_phobert_best_dotproduct.py:141: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `uit_phobert_best_half.py:215: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `uit_phobert_best_notopk.py:108: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`
* `uit_phobert_best_top1.py:136: test_df = test_df.sample(frac=0.5, random_state=RANDOM_SEED).reset_index(drop=True)`

## Macro-F1 Without Classification Report and Confusion Matrix

* `macro1_f1.py:26: macro_f1 = f1_score(y_true, y_pred, average='macro')`
* `scripts/macro1_f1.py:26: macro_f1 = f1_score(y_true, y_pred, average='macro')`
* `scripts/uit_RoBERTa_best.py:148: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_RoBERTa_best_half.py:149: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_RoBERTa_best_half_aggressive.py:157: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_RoBERTa_best_half_base.py:249: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_RoBERTa_best_half_nostop.py:95: print(f"\nMACRO-F1 (No Removal): {f1_score(test_df['label'], preds, average='macro'):.4f}")`
* `scripts/uit_RoBERTa_best_half_partial.py:145: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_nojaccard.py:228: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_nolabelsmo.py:497: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_nomulti-prompts.py:208: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_our_method.py:244: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_phobert_best.py:126: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_phobert_best_dotproduct.py:160: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_phobert_best_half.py:251: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_phobert_best_notopk.py:126: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_phobert_best_top1.py:155: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `scripts/uit_r64.py:243: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_RoBERTa_best.py:148: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_RoBERTa_best_half.py:149: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_RoBERTa_best_half_aggressive.py:157: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_RoBERTa_best_half_base.py:249: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_RoBERTa_best_half_nostop.py:95: print(f"\nMACRO-F1 (No Removal): {f1_score(test_df['label'], preds, average='macro'):.4f}")`
* `uit_RoBERTa_best_half_partial.py:145: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_nojaccard.py:228: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_nolabelsmo.py:497: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_nomulti-prompts.py:208: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_our_method.py:244: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_phobert_best.py:126: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_phobert_best_dotproduct.py:160: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_phobert_best_half.py:251: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_phobert_best_notopk.py:126: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_phobert_best_top1.py:155: macro_f1 = f1_score(true_labels, preds, average='macro')`
* `uit_r64.py:243: macro_f1 = f1_score(true_labels, preds, average='macro')`

## Generation Settings

* `scripts/generate_predictions_current_best.py:115: outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)`
* `scripts/run_optional_qwen3_prompt_baseline.py:86: output = model.generate(**inputs, max_new_tokens=10, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)`
* `scripts/uit_nojaccard.py:186: outputs = model.generate(**inputs, max_new_tokens=10, output_scores=True, return_dict_in_generate=True, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)`
* `scripts/uit_nolabelsmo.py:395: outputs = model.generate(**inputs, max_new_tokens=10, output_scores=True, return_dict_in_generate=True, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)`
* `scripts/uit_nomulti-prompts.py:179: outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, do_sample=False)`
* `scripts/uit_our_method.py:192: outputs = model.generate(**inputs, max_new_tokens=10, output_scores=True, return_dict_in_generate=True, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)`
* `scripts/uit_phobert_best_dotproduct.py:70: top_k = min(5, len(sentences))`
* `scripts/uit_r64.py:193: outputs = model.generate(**inputs, max_new_tokens=10, output_scores=True, return_dict_in_generate=True, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)`
* `uit_nojaccard.py:186: outputs = model.generate(**inputs, max_new_tokens=10, output_scores=True, return_dict_in_generate=True, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)`
* `uit_nolabelsmo.py:395: outputs = model.generate(**inputs, max_new_tokens=10, output_scores=True, return_dict_in_generate=True, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)`
* `uit_nomulti-prompts.py:179: outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, do_sample=False)`
* `uit_our_method.py:192: outputs = model.generate(**inputs, max_new_tokens=10, output_scores=True, return_dict_in_generate=True, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)`
* `uit_phobert_best_dotproduct.py:70: top_k = min(5, len(sentences))`
* `uit_r64.py:193: outputs = model.generate(**inputs, max_new_tokens=10, output_scores=True, return_dict_in_generate=True, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)`

## Label Smoothing Settings

* `scripts/audit_code_paper_consistency.py:13: LABEL_SMOOTH_RE = re.compile(r"label_smoothing_factor")`
* `scripts/uit_nojaccard.py:66: 'label_smoothing_factor': 0.1`
* `scripts/uit_nojaccard.py:146: label_smoothing_factor=TRAINING_CONFIG['label_smoothing_factor'],`
* `scripts/uit_nolabelsmo.py:133: 'label_smoothing_factor': 0.0 # Label Smoothing removed`
* `scripts/uit_nolabelsmo.py:319: label_smoothing_factor=0.0, # Removed for this configuration`
* `scripts/uit_nomulti-prompts.py:65: 'label_smoothing_factor': 0.1`
* `scripts/uit_nomulti-prompts.py:155: label_smoothing_factor=TRAINING_CONFIG['label_smoothing_factor'],`
* `scripts/uit_our_method.py:67: 'label_smoothing_factor': 0.1`
* `scripts/uit_our_method.py:155: label_smoothing_factor=TRAINING_CONFIG['label_smoothing_factor'],`
* `scripts/uit_r64.py:67: 'label_smoothing_factor': 0.1`
* `scripts/uit_r64.py:156: label_smoothing_factor=TRAINING_CONFIG['label_smoothing_factor'],`
* `uit_nojaccard.py:66: 'label_smoothing_factor': 0.1`
* `uit_nojaccard.py:146: label_smoothing_factor=TRAINING_CONFIG['label_smoothing_factor'],`
* `uit_nolabelsmo.py:133: 'label_smoothing_factor': 0.0 # Label Smoothing removed`
* `uit_nolabelsmo.py:319: label_smoothing_factor=0.0, # Removed for this configuration`
* `uit_nomulti-prompts.py:65: 'label_smoothing_factor': 0.1`
* `uit_nomulti-prompts.py:155: label_smoothing_factor=TRAINING_CONFIG['label_smoothing_factor'],`
* `uit_our_method.py:67: 'label_smoothing_factor': 0.1`
* `uit_our_method.py:155: label_smoothing_factor=TRAINING_CONFIG['label_smoothing_factor'],`
* `uit_r64.py:67: 'label_smoothing_factor': 0.1`
* `uit_r64.py:156: label_smoothing_factor=TRAINING_CONFIG['label_smoothing_factor'],`

## Prediction Output Names and Columns

* `README.md:43: PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh`
* `README.md:65: * `predict_label``
* `configs/vihallu_evidence.yaml:21: - final_submission_scratch.csv`
* `configs/vihallu_evidence.yaml:28: pred_col: predict_label`
* `docs/codex_session_memory.md:42: * No prediction CSV was present at final_submission_scratch.csv, results/predictions.csv, or results/paper_evidence/predictions.csv during initial inspection.`
* `docs/setup_uv_rtx4090.md:27: PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh`
* `docs/target_runbook.md:22: If a prediction CSV already exists and contains `id,predict_label`:`
* `docs/target_runbook.md:25: PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh`
* `macro1_f1.py:10: # 'label' comes from vihallu-test.csv and 'predict_label' from your submission`
* `macro1_f1.py:13: df_pred[['id', 'predict_label']],`
* `macro1_f1.py:24: y_pred = merged['predict_label']`
* `macro1_f1.py:38: evaluate_performance('vihallu-test.csv', 'final_submission_scratch.csv')`
* `scripts/audit_code_paper_consistency.py:14: SUBMISSION_RE = re.compile(r"final_submission_scratch\.csv|predict_label")`
* `scripts/build_paper_evidence.py:24: return first_existing(["final_submission_scratch.csv", "results/predictions.csv", "results/paper_evidence/predictions.csv"])`
* `scripts/build_paper_evidence.py:40: wrong["confusion_type"] = wrong["label"].astype(str) + "_to_" + wrong["predict_label"].astype(str)`
* `scripts/build_paper_evidence.py:50: lines.append(f"## Case {i}: {row.get('label')} -> {row.get('predict_label')}")`
* `scripts/build_paper_evidence.py:105: parser.add_argument("--pred_col", default="predict_label")`
* `scripts/build_paper_evidence.py:142: merged = merged.rename(columns={args.label_col: "label", args.pred_col: "predict_label"})`
* `scripts/build_paper_evidence.py:144: wrong = merged[merged["label"] != merged["predict_label"]].copy()`
* `scripts/build_paper_evidence.py:147: summary = compute_and_save(merged["label"], merged["predict_label"], out)`
* `scripts/generate_predictions_current_best.py:153: "predict_label": pred,`
* `scripts/generate_predictions_current_best.py:162: bad = sorted(set(out_df["predict_label"].astype(str)) - set(LABELS))`
* `scripts/generate_predictions_current_best.py:164: raise ValueError(f"Invalid predict_label values generated: {bad}")`
* `scripts/macro1_f1.py:10: # 'label' comes from vihallu-test.csv and 'predict_label' from your submission`
* `scripts/macro1_f1.py:13: df_pred[['id', 'predict_label']],`
* `scripts/macro1_f1.py:24: y_pred = merged['predict_label']`
* `scripts/macro1_f1.py:38: evaluate_performance('vihallu-test.csv', 'final_submission_scratch.csv')`
* `scripts/run_optional_qwen3_prompt_baseline.py:93: "predict_label": pred,`
* `scripts/run_optional_qwen3_prompt_baseline.py:114: "predict_label",`
* `scripts/run_target_rtx4090_full_pipeline.sh:48: if [ -z "$PRED" ] && [ -f "final_submission_scratch.csv" ]; then`
* `scripts/run_target_rtx4090_full_pipeline.sh:49: PRED="final_submission_scratch.csv"`
* `scripts/run_target_rtx4090_full_pipeline.sh:64: echo "  PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh"`
* `scripts/run_target_rtx4090_full_pipeline.sh:69: python3 scripts/build_paper_evidence.py --gold_csv vihallu-test.csv --pred_csv "$PRED" --out_dir results/paper_evidence --label_col label --pred_col predict_label --seed 42`
* `scripts/uit_nomulti-prompts.py:172: def predict_label(model, tokenizer, context, response):`
* `scripts/uit_nomulti-prompts.py:205: preds.append(predict_label(model, tokenizer, row['context'], row['response']))`
* `uit_nomulti-prompts.py:172: def predict_label(model, tokenizer, context, response):`
* `uit_nomulti-prompts.py:205: preds.append(predict_label(model, tokenizer, row['context'], row['response']))`
