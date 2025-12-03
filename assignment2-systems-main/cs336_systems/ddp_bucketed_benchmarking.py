import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from ddp_overlap_bucketed import DDPBucketed
from cs336_basics.model import BasicsTransformerLM

XL_CONFIG = {
    "vocab_size": 10000,
    "d_model": 1600,
    "d_ff": 6400,
    "num_layers": 48,
    "num_heads": 25,  # 1600 / 25 = 64 (head_dim)
    "context_length": 128,
    "rope_theta": 10000.0
}
# 使用 Medium Config
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

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 我们测试 5MB, 25MB
    bucket_sizes = [5, 25]

    if rank == 0:
        print(f"=== Starting Bucketed DDP Benchmark ===")
        print(f"Config: {ACTIVE_CONFIG}")

    for bucket_size in bucket_sizes:
        if rank == 0:
            print(f"\n--- Testing Bucket Size: {bucket_size} MB ---")

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

        # 封装 DDP (传入 bucket_size)
        ddp_model = DDPBucketed(raw_model, bucket_size_mb=bucket_size)

        optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        inputs = torch.randint(0, ACTIVE_CONFIG["vocab_size"], (BATCH_SIZE, ACTIVE_CONFIG["context_length"]), device=device)
        targets = torch.randint(0, ACTIVE_CONFIG["vocab_size"], (BATCH_SIZE, ACTIVE_CONFIG["context_length"]),device=device)

        # Warm-up
        for _ in range(3):
            optimizer.zero_grad()
            logits = ddp_model(inputs)
            loss = criterion(logits.view(-1, ACTIVE_CONFIG["vocab_size"]), targets.view(-1))
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()

        torch.cuda.synchronize()

        # Benchmark
        num_steps = 5
        total_time = 0.0

        for step in range(num_steps):
            t0 = time.time()
            optimizer.zero_grad()
            logits = ddp_model(inputs)
            loss = criterion(logits.view(-1, ACTIVE_CONFIG["vocab_size"]), targets.view(-1))

            # Backward (自动触发分桶通信)
            loss.backward()

            # 等待并解包
            ddp_model.finish_gradient_synchronization()

            optimizer.step()
            torch.cuda.synchronize()
            total_time += (time.time() - t0)

        if rank == 0:
            avg_time = total_time / num_steps
            print(f"Bucket Size {bucket_size}MB -> Avg Time: {avg_time:.4f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    run_benchmark()