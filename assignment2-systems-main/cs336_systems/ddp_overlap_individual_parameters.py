import torch
import torch.nn as nn
import torch.distributed as dist

class DDPOverlapIndividual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # 注册 Hook
        # 我们需要追踪所有的异步通信句柄 (handles)
        self.handles = []

        for param in self.module.parameters():
            if param.requires_grad:
                # 使用闭包来捕获当前的 param
                # 当反向传播算完这个 param 的梯度时，hook 会被触发
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param):
        def hook_fn(p):            #这里的p就是param的引用，当 PyTorch 触发 register_post_accumulate_grad_hook 时，它会自动传入触发该 hook 的参数本身作为参数
            if p.grad is not None:
                # 异步通信
                # async_op=True允许 CPU 立即返回去计算下一层梯度

                # 先除以 world_size (做平均)
                p.grad.data /= dist.get_world_size()

                # 发起异步通信
                handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append(handle)

        return hook_fn

    def forward(self, *inputs, **kwargs):
        # 每次 Forward 前，清空上一轮的 handles
        self.handles = []
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        在 optimizer.step() 之前调用。
        等待所有在后台运行的梯度数据包传输完毕。
        """
        for handle in self.handles:
            handle.wait()
        self.handles = []

