import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.dataset import EfficientDataset
from src.pruning import profile_activations, prune_by_masks, generate_answers_batched
from src.config import parse_args
from src.make_datasets import get_gsm8k, get_RACE, get_LA


def main():
    args = parse_args()

    get_gsm8k()
    get_RACE()
    get_LA()

    print(f"Loading model: {args.model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

    linear_modules = {
        name: module for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and "language_model" in name
    }
    print(f"Target linear layers extracted: {len(linear_modules)}")

    processor = AutoProcessor.from_pretrained(args.model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    # Profile activation masks sequentially to optimize memory usage
    datasets = {
        "ds1": args.dataset_1,
        "ds2": args.dataset_2,
        "ds3": args.dataset_3,
#        "ds4": args.dataset_4,
    }
    
    activations = {}
    for key, path in datasets.items():
        print(f"Profiling {key} ({path})...")
        ds = EfficientDataset(pd.read_csv(path))
        activations[key] = profile_activations(ds, model, processor, linear_modules, args.r, args.batch_size)

    act1, act2 = activations["ds1"], activations["ds2"]

    print("\nCalculating disjoint parameter sets...")
    dict12 = {k: int((act1[k] & ~act2[k]).sum().item()) for k in act1}
    print(f"Expected parameters to prune (DS1 - DS2): {sum(dict12.values()):,}")

    # Baseline Generations
    print("\nRunning pre-pruning generation baseline...")
    generate_answers_batched(model, processor, args.dataset_1, f"{args.output_dir}/pre_pruning_baseline.csv", args.batch_size)
    generate_answers_batched(model, processor, args.dataset_3, f"{args.output_dir}/pre_pruning_{args.dataset_3}.csv", args.batch_size)
    #generate_answers_batched(model, processor, args.dataset_4, f"{args.output_dir}/pre_pruning_{args.dataset_4}.csv", args.batch_size)

    # Pruning Execution
    pruning_masks = {k: (act1[k] & ~act2[k]) for k in act1}
    print("\nExecuting weight surgery...")
    total_pruned = prune_by_masks(model, pruning_masks)
    print(f"Successfully zeroed out {total_pruned:,} parameters across {len(pruning_masks)} layers.")

    # Post-pruning Generations
    print("\nRunning post-pruning generation evaluation...")
    generate_answers_batched(model, processor, args.dataset_1, f"{args.output_dir}/post_pruning_results.csv", args.batch_size)
    generate_answers_batched(model, processor, args.dataset_3, f"{args.output_dir}/post_pruning_{args.dataset_3}.csv", args.batch_size)
    #generate_answers_batched(model, processor, args.dataset_4, f"{args.output_dir}/post_pruning_{args.dataset_4}.csv", args.batch_size)

    print("Pipeline execution complete.")


if __name__ == "__main__":
    main()