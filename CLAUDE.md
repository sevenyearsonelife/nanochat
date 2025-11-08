# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

nanochat 是一个完整的类 ChatGPT 大语言模型实现，设计为在单个 8XH100 节点上运行完整训练流程。项目包含从分词、预训练、微调、评估到推理和 Web 服务的端到端实现。

## 常用命令

### 环境设置和依赖安装
```bash
# 安装 uv 并创建虚拟环境
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu  # GPU 环境
# uv sync --extra cpu  # CPU 环境
source .venv/bin/activate

# 构建 Rust BPE 分词器
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### 训练流程
```bash
# 快速训练 $100 级别的 d20 模型 (约4小时)
bash speedrun.sh

# 训练 $1000 级别的 d32 模型 (约42小时)
bash run1000.sh

# 在 screen 会话中运行
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

### 单独训练阶段
```bash
# 分词器训练
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# 基础模型预训练
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# 中期训练
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# 监督微调 (SFT)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# 强化学习 (可选)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN
```

### 推理和交互
```bash
# Web UI 聊天界面
python -m scripts.chat_web

# 命令行聊天
python -m scripts.chat_cli

# 带提示词的聊天
python -m scripts.chat_cli -p "你好，请介绍一下自己"
```

### 测试
```bash
# 运行所有测试
python -m pytest

# 运行特定测试
python -m pytest tests/test_rustbpe.py -v -s
python -m pytest tests/test_engine.py -v -s

# 跳过慢速测试
python -m pytest -m "not slow"
```

### 数据和报告
```bash
# 下载预训练数据分片
python -m nanochat.dataset -n 8  # 下载8个分片 (~2B 字符)

# 生成训练报告
python -m nanochat.report reset   # 重置报告
python -m nanochat.report generate  # 生成最终报告
```

## 项目架构

### 核心模块 (`nanochat/`)
- **`gpt.py`** - Transformer 模型实现，包含现代优化技术：rotary embeddings、QK norm、MQA 等
- **`tokenizer.py`** - BPE 分词器封装，兼容 GPT-4 风格
- **`engine.py`** - 高效推理引擎，支持 KV Cache
- **`dataloader.py`** - 分布式数据加载器，集成分词功能
- **`checkpoint_manager.py`** - 模型检查点保存/加载管理
- **`configurator.py`** - 配置管理，替代 argparse
- **`common.py`** - 通用工具函数

### 训练脚本 (`scripts/`)
- **`base_train.py`** - 基础模型预训练
- **`mid_train.py`** - 中期训练（教授对话特殊标记、工具使用等）
- **`chat_sft.py`** - 监督微调训练
- **`chat_rl.py`** - 强化学习训练
- **`chat_eval.py`** - 对话模型评估
- **`chat_web.py`** - Web UI 推理服务
- **`chat_cli.py`** - 命令行推理

### 评估任务 (`tasks/`)
- **`gsm8k.py`** - 小学数学问题
- **`humaneval.py`** - Python 编程任务
- **`mmlu.py`** - 多领域选择题
- **`arc.py`** - 科学推理问题
- **`smoltalk.py`** - 对话数据集

### 优化器
- **`adamw.py`** - 分布式 AdamW 优化器（用于嵌入层参数）
- **`muon.py`** - 分布式 Muon 优化器（用于矩阵参数）

## 训练配置要点

### 模型规模调整
- 使用 `--depth` 参数控制模型大小
- 调整 `--device_batch_size` 避免 OOM 错误
- 脚本会根据 Chinchilla 缩放定律自动计算训练步数

### 数据需求
- 每个数据分片约 250M 字符
- 预训练数据量 = 模型参数量 × 20 × 4.8 字符/标记
- 使用 `python -m nanochat.dataset -n N` 下载适当数量的分片

### 分布式训练
- 默认使用 8 个 GPU (`NPROC_PER_NODE=8`)
- 单 GPU 运行时移除 `torchrun` 前缀
- 自动调整梯度累积步数以保持总批次大小

## 设备兼容性

### GPU 训练
```bash
uv sync --extra gpu  # 安装 CUDA 版本 PyTorch
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### CPU/MPS 训练 (Mac)
```bash
uv sync --extra cpu  # 安装 CPU 版本 PyTorch
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1
```

参考 `dev/runcpu.sh` 获取完整的 CPU 训练配置示例。

## 日志和监控

### WandB 集成
```bash
wandb login  # 首次使用
WANDB_RUN=experiment_name bash speedrun.sh  # 带日志记录的训练
```

### 报告生成
- 训练过程中自动生成中间报告到 `~/.cache/nanochat/report/`
- 最终报告生成到项目根目录的 `report.md`
- 包含模型性能、损失曲线和评估指标

## 自定义和扩展

### 身份定制
- 参考 `dev/gen_synthetic_data.py` 生成个性化对话数据
- 在中期训练阶段混合自定义身份对话数据

### 能力扩展
- 参考讨论区的指南添加新能力
- 通过修改 `tasks/` 目录添加新的评估任务
- 调整 `scripts/` 中的训练脚本以支持新功能

## 开发注意事项

- 代码设计为最小化、可黑客攻击的单一代码库
- 避免过度复杂的配置系统
- 所有训练脚本都是端到端可运行的
- 使用 `uv` 进行依赖管理，支持 Python 3.10+
- Rust 分词器通过 `maturin` 构建并集成到 Python 中