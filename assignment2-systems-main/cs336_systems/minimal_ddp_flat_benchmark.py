#Naive DDP 慢是因为它对每个参数都进行all_reduce。Flattening的策略是：先把所有梯度装进一个大箱子（Flatten 成一个巨大的 1D Tensor），然后只进行一次all-reduce。
import torch
import torch.nn as nn
import torch.distributed as dist
import time
import os
import torch.optim as optim
from cs336_basics.model import BasicsTransformerLM


class MinimalDDPFlat(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # 初始化广播权重 (和朴素实现一样)
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def sync_gradients(self):
        """
        Batching 策略：
        1. 收集所有梯度
        2. 压扁成一个大 Tensor (Flatten)
        3. 通信一次 (All-Reduce)
        4. 解压回各个参数 (Unflatten)
        """
        world_size = dist.get_world_size()
        # 收集所有有效的梯度
        grads = [p.grad.data for p in self.module.parameters() if p.grad is not None]

        if not grads:
            return

        # 压扁 (Flatten)
        # 这会创建一个新的大张量，包含所有梯度的拷贝
        flat_grad = torch._utils._flatten_dense_tensors(grads)

        # 一次性通信 (All-Reduce)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)

        # 求平均
        flat_grad /= world_size

        # 解压还原
        # _unflatten_dense_tensors 返回的是切片后的新张量列表
        # 我们需要把这些值 copy_ 回原来的 p.grad 中
        synced_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)  #第二个参数是模板，按照模板的形状来还原

        for grad, synced in zip(grads, synced_grads):
            grad.copy_(synced) #原地操作，把grad的数据修改为同步后的 synced_grads数据

#仍然准备两个规格，我的有一个机器是无法实现XL规格的
XL_CONFIG = {
    "vocab_size": 10000,
    "d_model": 1600,
    "d_ff": 6400,
    "num_layers": 48,
    "num_heads": 25,
    "context_length": 128,
    "rope_theta": 10000.0
}
MEDIUM_CONFIG = {
    "vocab_size": 10000,
    "d_model": 1024,
    "d_ff": 4096,
    "num_layers": 24,
    "num_heads": 16,
    "context_length": 128,
    "rope_theta": 10000.0
}

ACTIVE_CONFIG = MEDIUM_CONFIG
BATCH_SIZE = 4


def run_benchmark():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print(f"=== Starting Flat DDP Benchmark ===")
        print(f"Config: {ACTIVE_CONFIG}")

    # 初始化模型
    raw_model = BasicsTransformerLM(
        vocab_size=ACTIVE_CONFIG["vocab_size"],
        context_length=ACTIVE_CONFIG["context_length"],
        d_model=ACTIVE_CONFIG["d_model"],
        num_layers=ACTIVE_CONFIG["num_layers"],
        num_heads=ACTIVE_CONFIG["num_heads"],
        d_ff=ACTIVE_CONFIG["d_ff"],
        rope_theta=ACTIVE_CONFIG["rope_theta"]
    ).to(device)

    # 使用 Flat DDP 封装
    ddp_model = MinimalDDPFlat(raw_model)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randint(0, ACTIVE_CONFIG["vocab_size"], (BATCH_SIZE, ACTIVE_CONFIG["context_length"]), device=device)
    targets = torch.randint(0, ACTIVE_CONFIG["vocab_size"], (BATCH_SIZE, ACTIVE_CONFIG["context_length"]),
                            device=device)

    # Warm-up
    if rank == 0: print("Warming up...")
    for _ in range(3):
        optimizer.zero_grad()
        logits = ddp_model(inputs)
        loss = criterion(logits.view(-1, ACTIVE_CONFIG["vocab_size"]), targets.view(-1))
        loss.backward()
        ddp_model.sync_gradients()
        optimizer.step()

    torch.cuda.synchronize()

    if rank == 0: print("Benchmarking...")
    num_steps = 5
    total_step = 0.0
    total_comm = 0.0

    for step in range(num_steps):
        t0 = time.time()

        optimizer.zero_grad()
        logits = ddp_model(inputs)
        loss = criterion(logits.view(-1, ACTIVE_CONFIG["vocab_size"]), targets.view(-1))
        loss.backward()
        torch.cuda.synchronize()

        # 测量通信时间
        comm_start = time.time()
        ddp_model.sync_gradients()  # 这里只发生一次 All-Reduce
        torch.cuda.synchronize()
        comm_end = time.time()

        optimizer.step()
        torch.cuda.synchronize()

        total_step += (time.time() - t0)
        total_comm += (comm_end - comm_start)

    if rank == 0:
        avg_step = total_step / num_steps
        avg_comm = total_comm / num_steps
        overhead = (avg_comm / avg_step) * 100
        print(f"\n=== Flat DDP Results ===")
        print(f"Avg Step: {avg_step:.4f}s")
        print(f"Avg Comm: {avg_comm:.4f}s")
        print(f"Overhead: {overhead:.2f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    run_benchmark()