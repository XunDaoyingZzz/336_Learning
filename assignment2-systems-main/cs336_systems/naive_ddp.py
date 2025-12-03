import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import copy

class NaiveDDP(nn.Module):
    def __init__(self,module: nn.Module):
        super().__init__()
        self.module=module
        self.rank=dist.get_rank()
        self.world_size=dist.get_world_size()

        #初始化的权重同步，保证各rank的起跑线一致，把rank0的weights传给其他rank
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0) #param的数据，src=0代表是以rank0的权重为准

    def forward(self, *input,**kwargs):
        return self.module(*input,**kwargs)

    def sync_gradients(self):
        """
        朴素的梯度同步函数，遍历每个参数，单独进行All-Reduce；通信次数=参数张量的数量
        """
        for param in self.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

#验证代码
def run_verification():
    rank=int(os.environ["RANK"])
    world_size=int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    if not dist.is_initialized():
        dist.init_process_group("nccl",rank=rank,world_size=world_size)
    torch.manual_seed(42)
    model=nn.Sequential(
        nn.Linear(10,10,bias=False),
        nn.Linear(10,1,bias=False),
    ).to(device)

    torch.manual_seed(100)
    inputs = torch.randn(2, 10).to(device) * 2
    labels = torch.randn(2, 1).to(device)

    #单卡
    if rank==0:
        print("作为参考的单卡训练")
        ref_model=copy.deepcopy(model)
        optimizer=optim.SGD(ref_model.parameters(),lr=0.01)

        optimizer.zero_grad()
        output=ref_model(inputs)
        loss=(output-labels).pow(2).sum()
        loss.backward()
        optimizer.step()
        print(f"参考的损失{loss.item()}")
        print(f"参考的第一层梯度 : {ref_model[0].weight.grad[0][0].item()}")
    #ddp
    dist.barrier()
    ddp_model=NaiveDDP(model)
    optimizer=optim.SGD(ddp_model.parameters(),lr=0.01)

    optimizer.zero_grad()
    output=ddp_model(inputs)
    loss=(output-labels).pow(2).sum()
    loss.backward()

    ddp_model.sync_gradients()
    optimizer.step()
    print(f"Rank {rank} ddp loss :{loss.item()}")

    if rank==0:
        print("朴素 ddp 步骤完成")
    dist.destroy_process_group()

if __name__=="__main__":
    run_verification()