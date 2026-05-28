# CS336 Assignment 1: Basics

This repository is my independent, self-study walkthrough of the CS336
Assignment 1 "Basics" project. I am working through it separately from the
class, so this repo is not aimed at Gradescope submission or the course
leaderboard.

For the original assignment description, see
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf).

## Status

All core code implementations are complete:

- Byte-level BPE tokenizer training and encoding/decoding
- Transformer language model components
- Cross-entropy, gradient clipping, AdamW, LR schedule, checkpointing
- Tokenized data loading
- Decoding/generation utilities
- Training script scaffold for experiments

The local test suite currently passes:

```sh
uv run pytest tests/
```

The repo is ready to run TinyStories/OpenWebText experiments and ablations on a
GPU instance using the tokenized datasets and training script.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

The adapters in [./tests/adapters.py](./tests/adapters.py) are wired to the
local implementations.

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Running Experiments

After downloading the raw datasets, train/tokenize with the BPE utilities in
`cs336_basics/bpe.py` and `scripts/tokenizer_experiments.py`, then train a
language model with:

```sh
uv run python scripts/train_lm.py \
  --train-tokens-path out/tinystories_train_tokens.npy \
  --valid-tokens-path out/tinystories_valid_tokens.npy \
  --checkpoint-path out/checkpoints/tinystories.pt \
  --vocab-size 10000 \
  --device cuda
```

Use `--help` to see model, optimizer, logging, evaluation, checkpoint, and
resume options:

```sh
uv run python scripts/train_lm.py --help
```
