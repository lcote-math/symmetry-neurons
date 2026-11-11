import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune the parameters activated by one dataset but not another, "
                    "and compare a model's answers on evaluation datasets before and after."
    )
    parser.add_argument("--model_name", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--prune_dataset", type=str, default="gsm8k",
                        help="dataset whose activated parameters are pruned (name or csv path)")
    parser.add_argument("--keep_dataset", type=str, default="race",
                        help="parameters also activated by this dataset are kept (name or csv path)")
    parser.add_argument("--eval_datasets", type=str, nargs="+", default=["regular_c4", "irregular_c4"],
                        help="datasets to answer before and after pruning (names or csv paths)")
    parser.add_argument("--r", type=float, default=0.05,
                        help="fraction of each layer's weights counted as activated by a dataset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for forward passes and generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    return parser.parse_args()
