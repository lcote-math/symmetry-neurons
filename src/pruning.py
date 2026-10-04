import torch
import torch.nn as nn
import pandas as pd
from src.dataset import EfficientDataset, pre_processor

def profile_activations(dataset: EfficientDataset, model, processor, linear_modules: dict, r_ratio: float, batch_size: int = 8):
    l2_sum_accum = {
        name: torch.zeros(module.weight.shape[1], device=module.weight.device, dtype=torch.float32)
        for name, module in linear_modules.items()
    }

    hooks = []
    def define_hook(name):
        def hook(module, input_data, output):
            with torch.no_grad():
                tensor_input = input_data[0].float()
                seq_l2 = tensor_input.norm(p=2, dim=1)
                l2_sum_accum[name] += seq_l2.sum(dim=0)
        return hook

    for name, module in linear_modules.items():
        hooks.append(module.register_forward_hook(define_hook(name)))

    total_samples = 0
    for i in range(0, len(dataset), batch_size):
        batch_text, batch_context, batch_images = dataset.get_batch(i, batch_size)
        formatted = pre_processor(processor, batch_text, batch_context, batch_images)

        kwargs = {"text": formatted, "return_tensors": "pt", "padding": True}
        if batch_images and any(img is not None for img in batch_images):
            kwargs["images"] = batch_images

        inputs = processor(**kwargs).to(model.device)
        total_samples += inputs["input_ids"].shape[0]

        with torch.no_grad():
            _ = model(**inputs)

    for handle in hooks:
        handle.remove()

    activation_masks = {}
    with torch.no_grad():
        for name, module in linear_modules.items():
            tens = module.weight.data
            normalized_tensor_input = l2_sum_accum[name] / total_samples
            weighted_tens = tens.abs().float() * normalized_tensor_input.unsqueeze(0)

            score_flat = weighted_tens.view(-1)
            cutoff = int(r_ratio * score_flat.numel())
            mask = torch.zeros_like(score_flat, dtype=torch.bool)
            if cutoff > 0:
                _, ind = torch.topk(score_flat, cutoff)
                mask[ind] = True

            activation_masks[name] = mask.view(tens.shape)

    del l2_sum_accum
    torch.cuda.empty_cache()
    return activation_masks


def prune_by_masks(model: nn.Module, pruning_masks: dict) -> int:
    total_pruned = 0
    modules_dict = dict(model.named_modules())

    with torch.no_grad():
        for layer_name, mask in pruning_masks.items():
            if layer_name in modules_dict:
                module = modules_dict[layer_name]
                module.weight.data[mask] = 0.0
                total_pruned += int(mask.sum().item())

    return total_pruned


def generate_answers_batched(model, processor, dataset_path: str, output_csv: str, batch_size: int = 8):
    df = pd.read_csv(dataset_path)
    ds = EfficientDataset(df)
    rows = []

    print(f"Generating responses for {len(df)} samples from {dataset_path}...")
    for i in range(0, len(ds), batch_size):
        batch_text, batch_context, batch_images = ds.get_batch(i, batch_size)
        formatted_prompts = pre_processor(processor, batch_text, batch_context, batch_images)

        kwargs = {"text": formatted_prompts, "return_tensors": "pt", "padding": True}
        if batch_images and any(img is not None for img in batch_images):
            kwargs["images"] = batch_images

        inputs = processor(**kwargs).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        answers = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for txt, ans in zip(batch_text, answers):
            rows.append({"input": txt, "answer": ans.strip()})

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved outputs to {output_csv}")