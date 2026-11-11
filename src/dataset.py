"""
EfficientDataset wraps a dataframe and serves batches of (prompts, contexts, images).
pre_processor turns such a batch into chat-formatted prompts for a multimodal model.

Column detection (case-insensitive):
  - prompt:  first of question / text / prompt / input
  - context: article (optional; prepended to the prompt, as in RACE)
  - image:   first of image / image_path / img / images (optional; a local path or http(s) URL)
"""

import os

import pandas as pd
from transformers.image_utils import load_images


def _find_column(df, candidates):
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


class EfficientDataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.text_col = _find_column(df, ["question", "text", "prompt", "input"])
        if self.text_col is None:
            raise ValueError(f"no prompt column found among {list(df.columns)}")
        self.context_col = _find_column(df, ["article"])
        self.image_col = _find_column(df, ["image", "image_path", "img", "images"])

    def __len__(self):
        return len(self.df)

    def get_batch(self, start_idx: int, batch_size: int):
        sub_df = self.df.iloc[start_idx:start_idx + batch_size]

        prompts = sub_df[self.text_col].astype(str).tolist()
        contexts = sub_df[self.context_col].astype(str).tolist() if self.context_col else None

        images = None
        if self.image_col:
            images = []
            for src in sub_df[self.image_col].tolist():
                if not isinstance(src, str):
                    images.append(None)
                    continue
                if not (src.startswith("http") or os.path.exists(src)):
                    raise FileNotFoundError(f"image {src!r} not found (run from the repository root)")
                images.append(load_images([src]))

        return prompts, contexts, images


def pre_processor(processor, text_prompts, context_prompts=None, images_list=None):
    formatted_prompts = []

    for idx, question in enumerate(text_prompts):
        content = []
        if images_list and images_list[idx] is not None:
            content.append({"type": "image"})

        context = context_prompts[idx] if context_prompts else None
        if context and context.strip() != "nan":
            full_text = f"{context}\n\nQuestion:\n{question}"
        else:
            full_text = str(question)

        content.append({"type": "text", "text": full_text})
        messages = [{"role": "user", "content": content}]
        formatted_prompts.append(processor.apply_chat_template(messages, add_generation_prompt=True))

    return formatted_prompts
