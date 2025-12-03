import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_benchmark(rank, world_size, data_sizes_mb):
    """
    运行基准测试的 Worker 函数
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    results = []

    for size_mb in data_sizes_mb:
        # 计算元素数量 (float32 = 4 bytes)
        # 1 MB = 1024 * 1024 bytes
        num_elements = int(size_mb * 1024 * 1024 / 4)
        # 创建随机张量 (CPU)
        data = torch.rand(num_elements)
        # Warm-up (热身)
        for _ in range(5):
            dist.all_reduce(data, op=dist.ReduceOp.SUM)

        # 测量 10 次取平均
        repeat = 10
        start_time = time.time()
        for _ in range(repeat):
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            # CPU 不需要 torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / repeat

        # 只在 Rank 0 收集结果
        if rank == 0:
            print(f"[Rank {rank}] Processes: {world_size}, Data: {size_mb}MB, Time: {avg_time:.4f}s")
            results.append({
                "Backend": "Gloo (CPU)",
                "Processes": world_size,
                "Data Size (MB)": size_mb,
                "Time (s)": avg_time
            })

    dist.destroy_process_group()
    return results


def main():
    # 作业要求的配置
    process_counts = [2, 4, 6]
    data_sizes_mb = [1, 10, 100, 1000]  # 1MB, 10MB, 100MB, 1GB

    all_results = []

    print("=== 开始单节点 Gloo (CPU) 基准测试 ===")

    for world_size in process_counts:
        print(f"\n--- Testing with {world_size} processes ---")
        # mp.spawn 启动多进程
        # 注意: spawn 返回 None，我们无法直接拿回返回值，这里简化逻辑，
        # 实际做表手动记录 Rank 0 的输出，或者用 Queue 传递数据。
        mp.spawn(
            run_benchmark,
            args=(world_size, data_sizes_mb),
            nprocs=world_size,
            join=True
        )

if __name__ == "__main__":
    main()