import os
import time
import torch
import torch.distributed as dist

def run_nccl_benchmark():
    # 从环境变量获取配置
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 针对单机单卡，设置当前设备
    # 每台机器只有一个可见的 GPU device 0
    local_device_id = 0
    torch.cuda.set_device(local_device_id)
    device = torch.device(f"cuda:{local_device_id}")

    # 初始化进程组 (NCCL)
    print(f"[Rank {rank}] 初始化nccl进程")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] 初始化完成")

    # 定义测试的数据大小 (MB)
    data_sizes_mb = [1, 10, 100, 1000]

    for size_mb in data_sizes_mb:
        # 计算元素数量 (float32)
        num_elements = int(size_mb * 1024 * 1024 / 4)

        # 创建 GPU 张量
        data = torch.rand(num_elements).to(device)

        # 热身
        for _ in range(5):
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()  # 确保 warm-up 完成

        # 测试
        repeat = 10
        start_time = time.time()
        for _ in range(repeat):
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            # 在 NCCL/GPU 计时时，必须手动同步等待 GPU 完成
        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / repeat
        # 计算带宽
        # All-reduce 传输量约为 2 * size (对于 Ring 算法是 2(N-1)/N * size)
        # 这里简单估算有效吞吐量 = Data Size / Time
        throughput = size_mb / avg_time

        print(f"[Rank {rank}] Data: {size_mb}MB | Time: {avg_time:.5f}s | Speed: {throughput:.2f} MB/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    run_nccl_benchmark()