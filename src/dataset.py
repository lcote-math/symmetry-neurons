"""
    This script defines the EfficientDataset class. This comes with a constructor method, which creates an object from a pandas dataframe object.
    It also comes with a get_batch() method, which (for a specified start and end id) associate to an EfficientDataset object three lists called prompts, contexts, images.
    Finally, we have a pre_processor class which takes these lists and formats them in the format that can be fed to an LLM.

"""
from transformers.image_utils import load_images



class EfficientDataset:
    def __init__(self, df: pd.DataFrame, default_prompt: str = "Which image is the intruder?..."):
        col_names = [c.lower() for c in df.columns]

        self.context_col = df.columns[col_names.index("article")] if "article" in col_names else None

        self.text_col = None
        for candidate in ["question", "text", "prompt", "input"]:
            if candidate in col_names:
                self.text_col = df.columns[col_names.index(candidate)]
                break
        if self.text_col is None:
            self.text_col = df.columns[0]

        self.image_col = None
        for candidate in ["image", "image_path", "img", "images"]:
            if candidate in col_names:
                self.image_col = df.columns[col_names.index(candidate)]
                break

        self.df = df
        self.default_prompt = default_prompt

    def __len__(self):
        return len(self.df)

    def get_batch(self, start_idx: int, batch_size: int):
        sub_df = self.df.iloc[start_idx:start_idx + batch_size]

        contexts = sub_df[self.context_col].astype(str).tolist() if self.context_col else None
        prompts = sub_df[self.text_col].astype(str).tolist()

        images = None
        if self.image_col:
            raw_sources = sub_df[self.image_col].tolist()
            images = [
                load_images([src]) if isinstance(src, str) and (src.startswith("http") or os.path.exists(src)) else None 
                for src in raw_sources
            ]

        return prompts, contexts, images

def pre_processor(processor, text_prompts, context_prompts=None, images_list=None):
    formatted_prompts = []
    has_images = images_list is not None and any(img is not None for img in images_list)

    for idx, question in enumerate(text_prompts):
        content = []
        if has_images and images_list and images_list[idx] is not None:
            content.append({"type": "image"})

        if context_prompts and idx < len(context_prompts) and context_prompts[idx] and str(context_prompts[idx]).strip() != "nan":
            full_text = f"{context_prompts[idx]}\n\nQuestion:\n{question}"
        else:
            full_text = str(question)

        content.append({"type": "text", "text": full_text})
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        formatted_prompts.append(prompt)

    return formatted_prompts