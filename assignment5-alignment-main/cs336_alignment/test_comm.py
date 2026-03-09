import os
import torch
import torch.distributed as dist


def main():
    # 1. 初始化分布式环境
    print("正在初始化 NCCL 分布式环境...")
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] 初始化成功! World Size: {world_size}")

    # 确保 GPU 可用并设置当前设备
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA 设备！")
    torch.cuda.set_device(0)  # 每台机器上只有 1 张卡，直接用 0 号
    device = torch.device("cuda:0")

    # ==========================================
    # 测试一：跨机器传输 Python 对象 (列表/字符串)
    # ==========================================
    print(f"\n[Rank {rank}] --- 测试一：传输控制指令与元数据 ---")
    if rank == 0:
        # 主节点准备发送数据
        metadata_to_send = ["EVAL", ["问题1: 1+1=?", "问题2: 2+2=?"]]
        print(f"[Rank 0] 准备发送指令: {metadata_to_send}")
        dist.broadcast_object_list(metadata_to_send, src=0)
    else:
        # 从节点准备接收数据
        received_metadata = [None, None]  # 提前准备好对应长度的占位符
        print(f"[Rank 1] 等待接收指令...")
        dist.broadcast_object_list(received_metadata, src=0)
        print(f"[Rank 1] 成功接收指令: {received_metadata}")

    # ==========================================
    # 测试二：跨机器传输 GPU Tensor (模拟权重同步)
    # ==========================================
    print(f"\n[Rank {rank}] --- 测试二：传输 GPU Tensor (模拟模型权重) ---")
    tensor_shape = (5000, 5000)  # 约 100MB 的 bfloat16 Tensor

    if rank == 0:
        # 主节点在 GPU 上生成一个随机 Tensor
        weight_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16, device=device)
        print(f"[Rank 0] 生成 Tensor，形状: {weight_tensor.shape}, 均值: {weight_tensor.float().mean().item():.4f}")
        print("[Rank 0] 正在通过 NCCL 发送 Tensor...")
        dist.broadcast(weight_tensor, src=0)
        print("[Rank 0] 发送完成！")
    else:
        # 从节点在 GPU 上分配一块同等大小的空内存来接收
        weight_tensor = torch.empty(tensor_shape, dtype=torch.bfloat16, device=device)
        print(f"[Rank 1] 已分配空 Tensor，准备接收数据...")
        dist.broadcast(weight_tensor, src=0)
        print(f"[Rank 1] 成功接收 Tensor！形状: {weight_tensor.shape}, 均值: {weight_tensor.float().mean().item():.4f}")

    print(f"\n[Rank {rank}] 通信测试圆满结束！")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()