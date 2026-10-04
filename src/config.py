import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="MathNeuro Activation Profiling and Pruning")
    parser.add_argument("--model_name", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--dataset_1", type=str, default="datasets/gsm8k.csv")
    parser.add_argument("--dataset_2", type=str, default="datasets/race.csv")
    parser.add_argument("--dataset_3", type=str, default="datasets/sampled_uni_math_dataset.csv")
#    parser.add_argument("--dataset_4", type=str, default="datasets/c4_polygons.csv")
    parser.add_argument("--r", type=float, default=0.05, help="Top-r parameter ratio threshold")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for forward/generation")
    return parser.parse_args()
