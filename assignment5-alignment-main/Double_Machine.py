import torch
import torch.distributed as dist


def sync_weights_to_vllm_across_network(policy_model=None, llm_model=None, rank=0):
    """
    双机权重同步魔法：
    Rank 0 (4080S): 提取 state_dict，将元数据和权重张量 Broadcast 给 Rank 1。
    Rank 1 (3070): 接收元数据，动态在 GPU 上分配空间接收张量，并加载进 vLLM。
    """
    # 1. 同步字典的 Keys、Shapes 和 Dtypes (使用 broadcast_object_list 传 Python 对象)
    if rank == 0:
        sd = policy_model.state_dict()
        metadata = [(k, v.shape, v.dtype) for k, v in sd.items()]
        dist.broadcast_object_list([metadata], src=0)
    else:
        metadata_container = [None]
        dist.broadcast_object_list(metadata_container, src=0)
        metadata = metadata_container[0]

    # 2. 跨网传输 Tensor (使用 NCCL，极快)
    if rank == 0:
        for k, v in sd.items():
            tensor = v.to("cuda")  # 确保在 GPU 上以使用 NCCL
            dist.broadcast(tensor, src=0)
    else:
        received_sd = {}
        for k, shape, dtype in metadata:
            # 在 3070 上预先分配空 Tensor 准备接收
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            dist.broadcast(tensor, src=0)
            received_sd[k] = tensor

        # 3. 将接收到的权重加载进 vLLM [cite: 1, 399]
        llm_engine_model = llm_model.llm_engine.model_executor.driver_worker.model_runner.model
        llm_engine_model.load_weights(received_sd.items())

        # 释放显存，防止 8GB 被撑爆
        del received_sd
        torch.cuda.empty_cache()