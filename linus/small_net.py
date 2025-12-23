import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb  # ✅ 1) 引入 wandb

# 一个简单模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = Net()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

criterion = nn.CrossEntropyLoss()

# ✅ 2) 初始化一个 wandb run（建议放在训练前）
wandb.init(
    project="toy-net",          # 你的项目名（随便起）
    name="adamw-baseline",      # 这次实验的名字（可选）
)

# ✅ 3) 记录超参（强烈建议）
wandb.config.update({
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 0.01,
    "batch_size": 32,
    "epochs": 10,
    "model": "Net(784-256-10)",
})

# ✅ 4) 可选：自动跟踪梯度/参数的分布（对调试很有用）
wandb.watch(model, log="gradients", log_freq=1)

global_step = 0

# 模拟训练
for epoch in range(10):
    optimizer.zero_grad()
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))

    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    # ✅ 5) 记录指标（至少把 loss 上报）
    wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)
    global_step += 1

    print(f"epoch {epoch}, loss={loss.item():.4f}")

# ✅ 6) 可选：结束 run
wandb.finish()
