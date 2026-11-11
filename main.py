"""
1. Profile which parameters of the model are activated by --prune_dataset and by --keep_dataset.
2. Answer the --eval_datasets with the unpruned model.
3. Zero out the parameters activated by --prune_dataset but not by --keep_dataset.
4. Answer the --eval_datasets again with the pruned model.

Run from the repository root: python main.py --help
"""

import os

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.config import parse_args
from src.dataset import EfficientDataset
from src.make_datasets import resolve
from src.pruning import generate_answers_batched, profile_activations, prune_by_masks


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset names are built on first use; csv paths are used as-is.
    prune_label, prune_path = resolve(args.prune_dataset)
    keep_label, keep_path = resolve(args.keep_dataset)
    eval_sets = [resolve(d) for d in args.eval_datasets]

    print(f"Loading model: {args.model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    # Only the language model's linear layers are profiled and pruned (not the vision encoder).
    linear_modules = {
        name: module for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and "language_model" in name
    }
    print(f"Target linear layers: {len(linear_modules)}")

    activations = {}
    for label, path in [(prune_label, prune_path), (keep_label, keep_path)]:
        print(f"Profiling activations on {label} ({path})...")
        ds = EfficientDataset(pd.read_csv(path))
        activations[label] = profile_activations(ds, model, processor, linear_modules, args.r, args.batch_size)

    pruning_masks = {k: activations[prune_label][k] & ~activations[keep_label][k] for k in linear_modules}

    def evaluate(stage):
        print(f"\nAnswering evaluation datasets ({stage})...")
        return {
            label: generate_answers_batched(model, processor, path, os.path.join(args.output_dir, f"{stage}_{label}.csv"),
                                            args.batch_size, args.max_new_tokens)
            for label, path in eval_sets
        }

    scores_before = evaluate("pre_pruning")

    print(f"\nPruning parameters activated by {prune_label} but not by {keep_label}...")
    total_pruned = prune_by_masks(model, pruning_masks)
    print(f"Zeroed {total_pruned:,} parameters across {len(pruning_masks)} layers.")

    scores_after = evaluate("post_pruning")

    print("\nSummary (heuristic scores; check the csvs in the output directory):")
    for label, _ in eval_sets:
        before, after = scores_before[label], scores_after[label]
        if before is not None:
            print(f"  {label:15} before: {before:.0%}   after: {after:.0%}")
        else:
            print(f"  {label:15} answers saved (no automatic scoring)")


if __name__ == "__main__":
    main()
