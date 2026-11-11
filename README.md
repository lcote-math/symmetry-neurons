# symmetry-neurons

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lcote-math/symmetry-neurons/blob/main/colab.ipynb)

## Overview 
This repository contains the code accompanying my two blog posts: https://lcotenotesonai.wpcomstaging.com/2026/09/17/common-neurons-for-symmetry-and-mathematics-experiment-1-with-gemma12b/
and
https://lcotenotesonai.wpcomstaging.com/2026/09/17/a-neural-pathway-for-symmetry-and-mathematics/

The code contained in this respository was originally written by hand in a Colab notebook, and subsequently completely refactored by a coding agent (Opus 5.5). Gemini was used to make polygons.py.

The rest of the Readme is written by Opus 5.5. It contains a summary of the aforementioned blogposts and an overview of the codebase. 

LC.

## The research problem

Do the parameters a multimodal LLM uses for elementary mathematics also help it perceive geometric regularity?

Dehaene and collaborators found that the brain areas involved in higher mathematics are the same ones that respond to symmetry in humans. They also found that non-human animals and vision-only neural networks don't seem to respond to symmetry. This repository tests the analogous question for a multimodal LLM:

1. Find the parameters that are strongly activated by grade-school math problems (GSM8K) but not by reading comprehension (RACE).
2. Zero them out, using the pruning method of *Math Neurosurgery* (Christ et al., 2024).
3. Compare the model's ability to spot the odd shape out in a grid of polygons, before and after pruning.

## The "spot the intruder" tasks

Each image shows four white polygons of equal area on a black background, in a 2x2 grid. Three are the same shape up to rotation and scaling, and one, the intruder, is different. The model is asked which position (top-left, top-right, bottom-left, bottom-right) holds the intruder.

| Dataset | Three identical shapes | Intruder |
|---|---|---|
| `regular_c4` | squares | an irregular quadrilateral |
| `irregular_c4` | an irregular quadrilateral | a different irregular quadrilateral |

The generator in [src/polygons.py](src/polygons.py) also supports C4-symmetric shapes with 8, 12, … sides. These are symmetric shapes without a common name.

## Results (Gemma 3 12B, 20 questions per dataset, r = 5%)

| Task | Before pruning | After pruning |
|---|---|---|
| `regular_c4` (squares) | 13/20 | 2/20 |
| `irregular_c4` | 6/20 | 7/20 |

Random guessing scores 5/20. The model finds the intruder among squares much more easily, and pruning the GSM8K-specific parameters destroys that ability. On the harder task, performance stays near chance. The sample size is small.

## Usage

### On Google Colab

Click the **Open in Colab** badge above. The notebook walks through setup, builds the datasets, shows an example image, runs the experiment and displays the answers. The free T4 GPU can run Gemma 3 4B. Gemma 3 12B needs an A100 or L4 runtime.

### Locally

```bash
pip install -r requirements.txt
python main.py                       # build missing datasets, profile, prune, evaluate
python main.py --help                # all options
```

Run everything from the repository root. Gemma models are gated on Hugging Face: accept the license on the model page and log in with `huggingface-cli login` first. Gemma 3 12B in bfloat16 needs about 24 GB of GPU memory. For a quick test, use a smaller model such as `--model_name google/gemma-3-4b-it`.

`main.py` writes one csv per evaluation dataset and stage to `outputs/` (for example `outputs/post_pruning_regular_c4.csv`). Each csv contains the model's full answer and the correct position. The `correct` column comes from a heuristic that takes the last position mentioned in the answer, so check the answers by hand before relying on it.

### Datasets

Datasets are built automatically the first time `main.py` needs them. You can also build them yourself:

```bash
python -m src.make_datasets                         # list datasets and whether they are built
python -m src.make_datasets regular_c4 irregular_c4 # build some
python -m src.make_datasets all --force --n 50      # rebuild all with 50 samples each
```

| Name | Contents |
|---|---|
| `gsm8k` | grade-school math word problems ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)) |
| `race` | high-school reading comprehension ([race](https://huggingface.co/datasets/race)) |
| `linear_algebra` | true/false linear algebra questions ([data/custom_datasets/LA_2026.csv](data/custom_datasets/LA_2026.csv)) |
| `regular_c4`, `irregular_c4` | the "spot the intruder" images described above |

Sampling is seeded (default 20 samples, seed 42), so the same command produces the same datasets. The `--prune_dataset`, `--keep_dataset` and `--eval_datasets` options of `main.py` accept any of these names, or a path to your own csv. A csv needs a `question` column, and can have optional `article` (context) and `image_path` columns. To add a named dataset, write a builder function and register it in `DATASETS` in [src/make_datasets.py](src/make_datasets.py).

## Code layout

| File | Purpose |
|---|---|
| [main.py](main.py) | the experiment: profile, evaluate, prune, evaluate |
| [colab.ipynb](colab.ipynb) | the same, as a Google Colab notebook |
| [src/config.py](src/config.py) | command-line options |
| [src/make_datasets.py](src/make_datasets.py) | dataset registry and command-line builder |
| [src/polygons.py](src/polygons.py) | polygon image generation and answer scoring |
| [src/dataset.py](src/dataset.py) | batching and chat-template formatting |
| [src/pruning.py](src/pruning.py) | activation profiling, pruning, generation |

## Method

For each linear layer of the language model, weight `W[i, j]` gets the score `|W[i, j]| · mean_samples ‖X_j‖₂`. Here `X_j` is input feature `j` across the tokens of a sample. For each dataset, the top `r` fraction of weights in each layer are marked as activated. The pruned set is the weights activated by GSM8K but not by RACE. The vision encoder is not touched.

## License

Public domain ([the Unlicense](LICENSE)): use it for anything, with no conditions.
