#!/bin/bash

nvidia-smi

SINGULARITYENV_CUDA_VISIBLE_DEVICE=0
CUDA_VISIBLE_DEVICE=0

. .venv/bin/activate
uv run python3 train.py