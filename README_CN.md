# nanochat

![nanochat logo](dev/nanochat.png)

> 花费 100 美元能买到的最好的 ChatGPT。

这个仓库是一个完整的大语言模型实现，类似于 ChatGPT，采用单一、简洁、最小化、可黑客攻击、依赖轻量的代码库。nanochat 设计为通过像 [speedrun.sh](speedrun.sh) 这样的脚本在单个 8XH100 节点上运行，这些脚本从头到尾运行整个流程。这包括分词、预训练、微调、评估、推理，以及通过简单的 UI 进行网络服务，这样你就可以像与 ChatGPT 交谈一样与你自己的大语言模型对话。nanochat 将成为 Eureka Labs 正在开发的课程 LLM101n 的顶点项目。

## 与它对话

要了解这个仓库的最终效果，你目前可以在 [nanochat.karpathy.ai](https://nanochat.karpathy.ai/) 上找到 [nanochat d32](https://github.com/karpathy/nanochat/discussions/8)。"d32" 意味着这个模型在 Transformer 神经网络中有 32 层。这个模型有 19 亿参数，它通过简单运行单个脚本 [run1000.sh](run1000.sh) 在 380 亿个标记上训练，总训练成本约为 800 美元（在 8XH100 GPU 节点上约 33 小时训练时间）。虽然今天的水平足以超越 2019 年的 GPT-2，但它与现代大型语言模型如 GPT-5 相比仍有很大差距。当与这些微型模型交谈时，你会发现它们会犯很多错误，它们有点天真和愚蠢，而且会产生大量幻觉，有点像儿童。这挺有趣的。但 nanochat 的独特之处在于它完全属于你 - 完全可配置、可调整、可黑客攻击，并且从头到尾由你训练。要训练并与你自己的模型对话，我们转向...

## 快速开始

体验魔力的最快方法是运行 speedrun 脚本 [speedrun.sh](speedrun.sh)，它训练和推理 100 美元级别的 nanochat。在 8XH100 节点上每小时 24 美元，这大约需要 4 小时的总运行时间。从你喜欢的提供商启动一个新的 8XH100 GPU 盒子（例如，我使用并喜欢 [Lambda](https://lambda.ai/service/gpu-cloud)），然后启动训练脚本：

```bash
bash speedrun.sh
```

或者，由于脚本运行 4 小时，我喜欢在新的 screen 会话 `speedrun` 中像这样启动它（同时也将输出记录到 `speedrun.log`）：

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

如果你不太熟悉，请查看 [screen 速查表](https://gist.github.com/jctosta/af918e1618682638aa82)。你可以在 screen 会话中观看它运行，或者用 `Ctrl-a d` 分离并 `tail speedrun.log` 查看进度。现在等待 4 小时。完成后，你可以通过类似 ChatGPT 的 Web UI 与你的大语言模型对话。确保你的本地 uv 虚拟环境再次激活（运行 `source .venv/bin/activate`），然后提供服务：

```bash
python -m scripts.chat_web
```

然后访问显示的 URL。确保正确访问它，例如在 Lambda 上使用你所在节点的公共 IP，后跟端口，例如 [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/)，等等。然后像平时与 ChatGPT 交谈一样与你的大语言模型对话！让它写故事或诗歌。询问它你是谁来看看幻觉。问它为什么天空是蓝色的。或者为什么它是绿色的。speedrun 是一个 4e19 FLOPs 能力模型，所以有点像和幼儿园小朋友交谈 :)。

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

你也可以 `cat report.md` 文件，它出现在项目目录中，包含运行的"成绩单"，即一系列评估和指标。最后，你会看到一个汇总表，例如：

---

- 字符数：333,989
- 行数：8,304
- 文件数：44
- 标记数（约）：83,497
- 依赖（uv.lock 行数）：2,004

| 指标             | BASE     | MID      | SFT      | RL       |
|------------------|----------|----------|----------|----------|
| CORE             | 0.2219   | -        | -        | -        |
| ARC-Challenge    | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy         | -        | 0.3561   | 0.3876   | -        |
| GSM8K            | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval        | -        | 0.0671   | 0.0854   | -        |
| MMLU             | -        | 0.3111   | 0.3151   | -        |
| ChatCORE         | -        | 0.0730   | 0.0884   | -        |

总墙上时钟时间：3小时51分钟

---

（你的表可能默认缺少 RL 数字）。关于 speedrun 脚本以及要注意什么和期待什么，请参考我在仓库讨论区发布的 walkthrough：["介绍 nanochat：花费 100 美元能买到的最好的 ChatGPT"](https://github.com/karpathy/nanochat/discussions/1)。

## 更大的模型

不出所料，100 美元不足以训练出高性能的 ChatGPT 克隆。事实上，大语言模型以其数百万美元的资本支出而闻名。就我们的目的而言，我认为还有两个更大规模的兴趣点。首先是约 300 美元的 d26 模型（即 depth=26），在大约 12 小时内训练，略优于 GPT-2 CORE 分数。第二个是 1000 美元级别（约 41.6 小时），只是因为它是一个不错的整数。但这两个都尚未完全支持，因此也未附加在 master 分支中。

也就是说，为了举例说明，[speedrun.sh](speedrun.sh) 文件训练 GPT-2 级别模型 d26 所需的示例更改只涉及三个更改：

```bash
...
# 你需要下载更多的预训练数据分片
# 获取参数数量，乘以 20 得到标记数，乘以 4.8 得到字符数，
# 除以 2.5 亿得到分片数量。todo 需要改进这个...
python -m nanochat.dataset -n 450 &
...
# 使用 --depth 增加模型大小。为了避免 oom，将设备批次大小 32 -> 16 减半：
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# 确保之后在中期训练时也使用相同的设置：
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

就是这样！需要注意的最大事情是确保你有足够的数据分片进行训练（否则代码会循环并在同一训练集上做更多 epoch，稍微降低学习速度），以及管理你的内存/VRAM，主要通过减少 `device_batch_size` 直到合适（脚本会自动通过增加梯度累积循环次数来补偿，简单地将并行计算转换为顺序计算）。

关于将运行 nanochat 的计算环境的更多信息：

- 代码在 Ampere 8XA100 GPU 节点上也能正常运行，但会稍慢一些。
- 所有代码即使在单个 GPU 上也能正常运行，只需省略 `torchrun`，并且会产生几乎相同的结果（代码会自动切换到梯度累积），但你必须等待 8 倍的时间。
- 如果你的 GPU VRAM 少于 80GB，你必须调整一些超参数，否则会 OOM / 耗尽 VRAM。在脚本中寻找 `--device_batch_size` 并减少它直到合适。例如从 32（默认）到 16、8、4、2，甚至 1。少于这个你就需要更了解你在做什么并更有创意。

## 在 CPU / MPS 上运行

nanochat 可以在 CPU 或 MPS 上运行（如果你在 Macbook 上），并且会自动尝试检测最佳运行设备。没有 GPU 你不会走得太远，但至少你能够运行代码路径，也许可以耐心训练一个小型大语言模型。关于如何使所有运行命令更小的示例（请随意调整！），你可以参考 [dev/runcpu.sh](dev/runcpu.sh) 文件。你会看到我本质上限制了所有脚本训练更小的模型，以运行更短的迭代次数等。这个功能是新的，有点复杂（触及了很多代码），并且是在 2025 年 10 月 21 日的这个 [CPU|MPS PR](https://github.com/karpathy/nanochat/pull/88) 中合并的。

## 定制化

要定制你的 nanochat，请参阅讨论区中的[指南：为你的 nanochat 注入身份](https://github.com/karpathy/nanochat/discussions/139)，它描述了如何通过合成数据生成和将该数据混合到中期训练和 SFT 阶段来调整你的 nanochat 的个性。

此外，要为 nanochat 添加新能力，请参阅[指南：在 strawberry 中数 r（以及如何一般地添加能力）](https://github.com/karpathy/nanochat/discussions/164)。

## 问题

nanochat 被设计为简短而精炼。一个很大的优势是我们可以将所有文件打包在一起，并将它们复制粘贴到你喜欢的大语言模型来询问任意问题。例如，我喜欢使用 [files-to-prompt](https://github.com/simonw/files-to-prompt) 工具打包仓库，如下所示：

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

这包括所有 py、rs、html、toml、sh 文件，排除 `rustbpe/target` 文件夹，并选择 cxml 输出格式。所有内容都写入 `packaged.txt` 文件，目前约为 330KB（即远低于最先进大语言模型的约 100K 标记），以及 45 个文件中的约 8K 行代码。

或者，我推荐使用来自 Devin/Cognition 的 [DeepWiki](https://deepwiki.com/karpathy/nanochat) 来询问这个仓库的问题。在这个仓库的 URL 中，简单地将 github.com 改为 deepwiki.com，你就可以开始了。

## 测试

我在这里投入不多，但存在一些测试，特别是对于分词器。运行例如：

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## 文件结构

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # 身份的示例合成数据
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # 预训练数据分片生成
│   └── runcpu.sh                   # 在 CPU/MPS 上运行的小示例
├── nanochat
│   ├── __init__.py                 # 空
│   ├── adamw.py                    # 分布式 AdamW 优化器
│   ├── checkpoint_manager.py       # 保存/加载模型检查点
│   ├── common.py                   # 杂项小工具，生活质量
│   ├── configurator.py             # argparse 的更优替代
│   ├── core_eval.py                # 评估基础模型 CORE 分数（DCLM 论文）
│   ├── dataloader.py               # 分词分布式数据加载器
│   ├── dataset.py                  # 预训练数据的下载/读取工具
│   ├── engine.py                   # 高效模型推理与 KV Cache
│   ├── execution.py                # 允许大语言模型执行 Python 代码作为工具
│   ├── gpt.py                      # GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # 评估每字节位数（而不是损失）
│   ├── muon.py                     # 分布式 Muon 优化器
│   ├── report.py                   # 编写 nanochat 报告的工具
│   ├── tokenizer.py                # GPT-4 风格的 BPE 分词器包装器
│   └── ui.html                     # nanochat 前端的 HTML/CSS/JS
├── pyproject.toml
├── run1000.sh                      # 训练约 800 美元的 nanochat d32
├── rustbpe                         # 自定义 Rust BPE 分词器训练器
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── README.md                   # 参见这里了解为什么存在这个
│   └── src
│       └── lib.rs
├── scripts
│   ├── base_eval.py                # 基础模型：计算 CORE 分数
│   ├── base_loss.py                # 基础模型：计算每字节位数，采样
│   ├── base_train.py               # 基础模型：训练
│   ├── chat_cli.py                 # 聊天模型（SFT/Mid）：通过 CLI 对话
│   ├── chat_eval.py                # 聊天模型（SFT/Mid）：评估任务
│   ├── chat_rl.py                  # 聊天模型（SFT/Mid）：强化学习
│   ├── chat_sft.py                 # 聊天模型：训练 SFT
│   ├── chat_web.py                 # 聊天模型（SFT/Mid）：通过 WebUI 对话
│   ├── mid_train.py                # 聊天模型：中期训练
│   ├── tok_eval.py                 # 分词器：评估压缩率
│   └── tok_train.py                # 分词器：训练它
├── speedrun.sh                     # 训练约 100 美元的 nanochat d20
├── tasks
│   ├── arc.py                      # 多项科学选择问题
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 从任意 jsonl 对话创建任务
│   ├── gsm8k.py                    # 8K 小学数学问题
│   ├── humaneval.py                # 误称；简单 Python 编程任务
│   ├── mmlu.py                     # 多项选择问题，广泛主题
│   ├── smoltalk.py                 来自 HF 的 SmolTalk 综合数据集
│   └── spellingbee.py              # 教模型拼写/计算字母的任务
├── tests
│   └── test_rustbpe.py
└── uv.lock
```

## 贡献

nanochat 远未完成。目标是在可在 1000 美元以下预算端到端运行的微模型方面提高最先进水平。可访问性既关乎总成本，也关乎认知复杂性 - nanochat 不是一个详尽可配置的大语言模型"框架"；代码库中不会有巨大的配置对象、模型工厂或 if-then-else 怪物。它是一个单一、连贯、最小化、可读、可黑客攻击、最大可派生的"强基线"代码库，设计为从头到尾运行并产生一个具体的 ChatGPT 克隆及其成绩单。

当前大语言模型政策：披露。提交 PR 时，请声明任何有重大大语言模型贡献且你未编写或不完全理解的部分。

## 致谢

- 名称（nanochat）源自我早期的项目 [nanoGPT](https://github.com/karpathy/nanoGPT)，它只涵盖预训练。
- nanochat 也受到 [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) 的启发，该项目的 nanogPT 仓库通过清晰的指标和排行榜进行了游戏化，并借用了很多想法和一些预训练实现。
- 感谢 [HuggingFace](https://huggingface.co/) 提供 fineweb 和 smoltalk。
- 感谢 [Lambda](https://lambda.ai/service/gpu-cloud) 为开发此项目提供的计算资源。
- 感谢首席大语言模型低语者 🧙‍♂️ Alec Radford 的建议/指导。

## 引用

如果你发现 nanochat 对你的研究有帮助，请简单地引用为：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 许可证

MIT