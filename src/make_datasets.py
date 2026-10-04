"""
This file downloads the required datasets and selects n-many entries, randomized according to seed. Then it saves this in a csv file.
"""

import pandas as pd

def get_gsm8k(n=30, seed = 42):
    url = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"
    df = pd.read_parquet(url)
    sampled_df = df.sample(n, random_state=seed).reset_index(drop=True)
    sampled_df.to_csv("/datasets/gsm8k.csv", index=False)
    print(f"Saved {n} samples to gsm8k_sample.csv!")


def get_RACE(n=30, seed = 42):
    url = "https://huggingface.co/datasets/race/resolve/refs%2Fconvert%2Fparquet/high/train/0000.parquet"
    df = pd.read_parquet(url)
    df = df.iloc[:, 1:]

    sampled_df = df.sample(n, random_state=seed).reset_index(drop=True)
    sampled_df.to_csv("/datasets/race.csv", index=False)
    print(f"Saved {n} samples to race_sample.csv!")


def get_LA(n=30, seed = 42):

    df = pd.read_csv("/data/custom_datasets/LA_2026.csv")
    sampled_df = df.sample(n, random_state=seed).reset_index(drop=True)
    sampled_df.to_csv("/datasets/sampled_uni_math_dataset.csv", index=False)
    print(f"Saved {n} samples to uni_math_dataset.csv!")


