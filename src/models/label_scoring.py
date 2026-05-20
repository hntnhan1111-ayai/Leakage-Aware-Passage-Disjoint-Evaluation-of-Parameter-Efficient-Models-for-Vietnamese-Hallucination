from __future__ import annotations

import torch
import torch.nn.functional as functional


LABELS = ("no", "intrinsic", "extrinsic")


def encode_prompt(tokenizer, prompt, max_prompt_tokens=1024):
    original_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        return tokenizer(prompt, return_tensors="pt", add_special_tokens=True, max_length=max_prompt_tokens, truncation=True)
    finally:
        tokenizer.truncation_side = original_side


def score_labels_causal_lm(model, tokenizer, prompt, labels=LABELS, device=None, max_prompt_tokens=1024):
    model.eval()
    device = device or next(model.parameters()).device
    results = {}
    with torch.inference_mode():
        for label in labels:
            label_ids = tokenizer(" " + label, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            max_length = max(1, int(max_prompt_tokens) - int(label_ids.shape[1]))
            prompt_ids = encode_prompt(tokenizer, prompt, max_length).input_ids.to(device)
            input_ids = torch.cat([prompt_ids, label_ids], dim=1)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            start = prompt_ids.shape[1] - 1
            end = start + label_ids.shape[1]
            label_logits = logits[:, start:end, :]
            label_targets = targets[:, start:end]
            loss = functional.cross_entropy(
                label_logits.reshape(-1, label_logits.shape[-1]),
                label_targets.reshape(-1),
                reduction="mean",
            )
            results[label] = float(loss.detach().cpu())
    return min(results, key=results.get), results
