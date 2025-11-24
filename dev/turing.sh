#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# Run as:
# bash dev/turing.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# This is also why I hide this script away in dev/

# 简化后的 setup (仅保留运行时必须的环境配置，非第一次运行)
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
source .venv/bin/activate # 关键：激活已存在的虚拟环境
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# wipe the report
python -m nanochat.report reset

# train tokenizer on ~1B characters
python -m nanochat.dataset -n 4

python -m scripts.tok_train --max_chars=1000000000

exit

python -m scripts.tok_eval

# train a very small 4 layer model on the CPU
# each optimization step processes a single sequence of 1024 tokens
# we only run 50 steps of optimization (bump this to get better results)
python -m scripts.base_train \
    --depth=2 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --core_metric_every=-1 \
    --core_metric_max_per_task=12 \
    --sample_every=30 \
    --num_iterations=30
python -m scripts.base_loss --device_batch_size=1 --split_tokens=4096
python -m scripts.base_eval --max-per-task=16

exit

# midtraining
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --total_batch_size=1024 \
    --num_iterations=100
# eval results will be terrible, this is just to execute the code paths.
# note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# SFT
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --num_iterations=20 \
    --eval_steps=4 \
    --eval_metrics_max_problems=4

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate
