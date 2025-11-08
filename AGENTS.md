# Repository Guidelines

## 项目结构与模块组织
- `nanochat/`：核心 Python 模块（如 `gpt.py`, `engine.py`, `tokenizer.py`）集中于此，负责模型结构、训练循环与推理执行；`ui.html` 为基础 Web UI。
- `scripts/`：包含 `base_train.py`, `chat_web.py`, `chat_rl.py` 等可执行入口，按照训练阶段（base/mid/sft/rl）命名，方便以 `python -m scripts.xxx` 调用。
- `tests/`：Pytest 套件覆盖数据管线与推理，所有新模块需在此添加 `test_<module>.py`；`tasks/` 集中封装 ARC/MMLU 等评测基准；`rustbpe/` 存放 BPE 内核，使用 Maturin 构建。

## 构建、测试与开发命令
- 依赖管理：`uv sync --all-extras` 安装 CPU/GPU 变体，`uv venv && source .venv/bin/activate` 启用环境。
- 训练/推理：`bash speedrun.sh` 运行端到端流水线，`python -m scripts.base_train --config configs/base.json` 触发单阶段训练，`python -m scripts.chat_web` 启动本地 UI。
- Rust 扩展：修改 BPE 后执行 `uv run maturin develop --release -m rustbpe/Cargo.toml` 重新编译。

## 编码风格与命名约定
- Python 采用 PEP8 与 4 空格缩进，函数名使用 snake_case，模型/配置类保持 PascalCase；常量全部大写并集中于模块顶部。
- 仅在 `__init__.py` 中暴露公共 API，内部工具放入 `_common.py` 风格文件；配置键建议沿用 `configurator.py` 里的现有命名。
- keep imports ordered：标准库→第三方→本地，相邻分组间空一行，确保与现有文件一致。

## 测试准则
- 运行 `uv run pytest` 作为默认回归；重训练相关慢测使用 `uv run pytest -m slow` 并记录耗时。
- 新算子/算例需提供最小可复现实例（例如伪 batch 的张量形状）并断言数值或日志输出，保持 deterministic。
- 若改动 `scripts/` 入口，附带一次集成演练（可执行 `python -m scripts.chat_cli --prompt "test"`）并在 PR 描述中贴日志摘要。

## 提交与 PR 规范
- Commit 信息遵循 Git 历史的动词首句格式，例如 `Refine tok_eval caching`，控制在 65 字符内并聚焦单一变更。
- PR 描述模板：背景→修改点→验证→风险，必要时附 `speedrun.log` 或 Web UI 截图；跨模块改动需链接 issue/讨论串。
- 在合并前确认 `report.md`、大模型 checkpoint 等衍生文件未被追踪；如需共享结果，改为上传到外部存储并贴链接。

## 安全与配置提示
- 每次切换 GPU/CPU 目标前检查 `pyproject.toml` 中的 `tool.uv.sources`，避免同时激活 `cpu` 与 `gpu` extra；生产集群需显式导出 `TORCH_CUDA_ARCH_LIST`。
- `.env` 或凭证仅用于本地测试，部署前改写为环境变量；`scripts/chat_web.py` 默认绑定 0.0.0.0:8000，面向公网时务必置于反向代理之后并启用 HTTPS。
