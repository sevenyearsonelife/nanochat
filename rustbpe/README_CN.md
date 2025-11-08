# rustbpe

> 缺失的 tiktoken 训练代码

一个非常轻量级的 Rust 库，用于训练 GPT 分词器。问题是推理库 [tiktoken](https://github.com/openai/tiktoken) 很棒，但只做推理。另外，huggingface 的 [tokenizers](https://github.com/huggingface/tokenizers) 库做训练，但它相当臃肿，真的很难导航，因为它必须支持多年来人们处理分词器的所有不同历史包袱。最近，我还写了 [minbpe](https://github.com/karpathy/minbpe) 库，它既做训练又做推理，但只在低效的 Python 中。基本上，我真正想要的是一个非常简单、超级简单但仍然相对高效的 GPT 分词器训练代码（比 minbpe 更高效，比 tokenizers 更干净/简单），然后导出训练好的词汇表用于 tiktoken 推理。这说得通吗？所以我们就在这里。这里还有更多优化机会，我只是早点停止了，因为不像之前的 minbpe，rustbpe 现在足够简单和快速，并且不是 nanochat 的显著瓶颈。