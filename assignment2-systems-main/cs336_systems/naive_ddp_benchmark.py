import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

# 引用刚才写的 NaiveDDP 类
from naive_ddp import NaiveDDP
from cs336_basics.model import BasicsTransformerLM

# --- 配置部分 ---
XL_CONFIG = {
    "vocab_size": 10000,
    "d_model": 1600,
    "d_ff": 6400,
    "num_layers": 48,
    "num_heads": 25,  # 1600 / 25 = 64 (head_dim)
    "context_length": 128,
    "rope_theta": 10000.0
}

# Config (备用)
MEDIUM_CONFIG = {
    "vocab_size": 10000,
    "d_model": 1024,
    "d_ff": 4096,
    "num_layers": 24,
    "num_heads": 16,
    "context_length": 128,
    "rope_theta": 10000.0
}

# 根据显存情况切换。XL 可能会 OOM，如果报错改用 MEDIUM_CONFIG
ACTIVE_CONFIG =MEDIUM_CONFIG #XL_CONFIG
BATCH_SIZE = 4


def run_benchmark():
    # 环境变量与初始化
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 双机单卡设置 (每台机器只有一个可见的 GPU 0)
    local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:                                                           #只在主节点打印日志
        print(f"=== Starting Naive DDP Benchmark ===")
        print(f"Model Config: {ACTIVE_CONFIG}")
        print(f"World Size: {world_size}")

    # 初始化 Assignment 1 的模型
    raw_model = BasicsTransformerLM(
        vocab_size=ACTIVE_CONFIG["vocab_size"],
        context_length=ACTIVE_CONFIG["context_length"],
        d_model=ACTIVE_CONFIG["d_model"],
        num_layers=ACTIVE_CONFIG["num_layers"],
        num_heads=ACTIVE_CONFIG["num_heads"],
        d_ff=ACTIVE_CONFIG["d_ff"],
        rope_theta=ACTIVE_CONFIG["rope_theta"]
    ).to(device)

    # 封装 NaiveDDP (触发 Broadcast)
    ddp_model = NaiveDDP(raw_model)

    # 简单的优化器
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 生成随机数据 (模拟 Batch)
    vocab_size = ACTIVE_CONFIG["vocab_size"]
    context_length = ACTIVE_CONFIG["context_length"]

    inputs = torch.randint(0, vocab_size, (BATCH_SIZE, context_length), device=device)
    targets = torch.randint(0, vocab_size, (BATCH_SIZE, context_length), device=device)

    # 热身
    if rank == 0:
        print("Warming up...")
    for _ in range(3):
        optimizer.zero_grad()
        logits = ddp_model(inputs)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        ddp_model.sync_gradients()
        optimizer.step()

    torch.cuda.synchronize()
    if rank == 0:
        print("Benchmark started...")

    # 正式计时循环
    num_steps = 5
    total_step_time = 0.0
    total_comm_time = 0.0

    for step in range(num_steps):
        t0 = time.time()

        # 计算部分 Forward + Backward
        optimizer.zero_grad()
        logits = ddp_model(inputs)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()

        torch.cuda.synchronize()

        # 通信部分
        comm_start = time.time()
        ddp_model.sync_gradients()  # 这里会触发几百次 All-Reduce
        torch.cuda.synchronize()  # 必须等待通信完成
        comm_end = time.time()

        # 更新部分
        optimizer.step()

        torch.cuda.synchronize()
        t_end = time.time()

        step_time = t_end - t0
        comm_time = comm_end - comm_start

        total_step_time += step_time
        total_comm_time += comm_time

        if rank == 0:
            print(f"Step {step + 1}: Total={step_time:.4f}s, Comm={comm_time:.4f}s")

    # 结果报告
    avg_step = total_step_time / num_steps
    avg_comm = total_comm_time / num_steps
    overhead_pct = (avg_comm / avg_step) * 100

    if rank == 0:
        print(f"\n=== Results for Naive DDP (XL Config) ===")
        print(f"Avg Training Step Time: {avg_step:.4f} s")
        print(f"Avg Communication Time: {avg_comm:.4f} s")
        print(f"Communication Overhead: {overhead_pct:.2f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    run_benchmark()
#通信开销特别大。