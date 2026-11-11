"""
Activation-based parameter importance, pruning, and generation.

profile_activations follows the MathNeuro scoring rule: the importance of weight W[i, j] in a linear layer is
|W[i, j]| times the average (over samples) L2 norm, across tokens, of input feature j. The top r fraction of
weights of each layer by this score are marked as "activated" for the dataset.
"""

import pandas as pd
import torch
import torch.nn as nn

from src.dataset import EfficientDataset, pre_processor
from src.polygons import extract_position


def _model_inputs(processor, dataset, start, batch_size, device):
    batch_text, batch_context, batch_images = dataset.get_batch(start, batch_size)
    formatted = pre_processor(processor, batch_text, batch_context, batch_images)

    kwargs = {"text": formatted, "return_tensors": "pt", "padding": True}
    if batch_images and any(img is not None for img in batch_images):
        kwargs["images"] = batch_images
    return processor(**kwargs).to(device)


def profile_activations(dataset: EfficientDataset, model, processor, linear_modules: dict, r_ratio: float, batch_size: int = 8):
    """Return {layer name: boolean mask of the top r_ratio fraction of weights by importance on `dataset`}."""
    l2_sum_accum = {
        name: torch.zeros(module.weight.shape[1], device=module.weight.device, dtype=torch.float32)
        for name, module in linear_modules.items()
    }

    def define_hook(name):
        def hook(module, input_data, output):
            with torch.no_grad():
                # input: (batch, tokens, features) -> L2 norm over tokens, summed over the batch
                l2_sum_accum[name] += input_data[0].float().norm(p=2, dim=1).sum(dim=0)
        return hook

    hooks = [module.register_forward_hook(define_hook(name)) for name, module in linear_modules.items()]

    total_samples = 0
    try:
        for i in range(0, len(dataset), batch_size):
            inputs = _model_inputs(processor, dataset, i, batch_size, model.device)
            total_samples += inputs["input_ids"].shape[0]
            with torch.no_grad():
                model(**inputs)
    finally:
        for handle in hooks:
            handle.remove()

    activation_masks = {}
    with torch.no_grad():
        for name, module in linear_modules.items():
            weights = module.weight.data
            mean_input_norm = l2_sum_accum[name] / total_samples
            scores = (weights.abs().float() * mean_input_norm.unsqueeze(0)).view(-1)

            cutoff = int(r_ratio * scores.numel())
            mask = torch.zeros_like(scores, dtype=torch.bool)
            if cutoff > 0:
                mask[torch.topk(scores, cutoff).indices] = True
            activation_masks[name] = mask.view(weights.shape)

    del l2_sum_accum
    torch.cuda.empty_cache()
    return activation_masks


def prune_by_masks(model: nn.Module, pruning_masks: dict) -> int:
    """Set the masked weights to zero. Returns the number of weights zeroed."""
    total_pruned = 0
    modules_dict = dict(model.named_modules())

    with torch.no_grad():
        for layer_name, mask in pruning_masks.items():
            if layer_name in modules_dict:
                modules_dict[layer_name].weight.data[mask] = 0.0
                total_pruned += int(mask.sum().item())

    return total_pruned


def generate_answers_batched(model, processor, dataset_path: str, output_csv: str, batch_size: int = 8, max_new_tokens: int = 1024):
    """
    Generate an answer for every row of the csv at dataset_path and save the rows plus a `model_answer` column
    to output_csv. For "spot the intruder" datasets, also add `predicted_position` and `correct` columns and
    return the accuracy; otherwise return None.
    """
    df = pd.read_csv(dataset_path)
    ds = EfficientDataset(df)
    answers = []

    print(f"Generating responses for {len(df)} samples from {dataset_path}...")
    for i in range(0, len(ds), batch_size):
        inputs = _model_inputs(processor, ds, i, batch_size, model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        answers += [a.strip() for a in processor.batch_decode(generated_ids, skip_special_tokens=True)]

    df["model_answer"] = answers
    accuracy = None
    if "intruder_position" in df.columns:
        df["predicted_position"] = df["model_answer"].map(extract_position)
        df["correct"] = df["predicted_position"] == df["intruder_position"]
        accuracy = df["correct"].mean()
        print(f"Score: {df['correct'].sum()}/{len(df)} (heuristic answer extraction; see {output_csv})")

    df.to_csv(output_csv, index=False)
    print(f"Saved outputs to {output_csv}")
    return accuracy
