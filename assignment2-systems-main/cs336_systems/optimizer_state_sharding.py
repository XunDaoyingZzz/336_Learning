#在之前的DDP中每个RANK都持有一份完整的优化器状态，这里的思想是每个RANK的优化器只负责更新一小部分参数，然后广播更新参数。
#以我的设备为例，RANK 0 和 RANK 1，我们把参数分为 A B 两组，RANK 0负责更新 A组、RANK 1负责更新 B组，step()被调用的时候两个RANK调用 AdamW更新自己的参数，此时 RANK 0的 B组参数仍然没动，同时 RANK 1的 A组参数也没动。
#更新完后再广播同步
import torch
import torch.distributed as dist
from typing import Type, Any

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        """
        params: 待优化的参数列表或参数组
        optimizer_cls: 底层优化器类 (例如 torch.optim.AdamW)
        kwargs: 传递给 optimizer_cls 的参数 (例如 lr, weight_decay)
        """
        # 先调用父类的 __init__，自动处理 params 将其统一转化为 self.param_groups 格式。
        # 我们把 kwargs 作为 defaults 传给父类。
        super().__init__(params, defaults=kwargs)

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # 建立参数到 Rank 的映射
        # Param 0 -> Rank 0, Param 1 -> Rank 1, Param 2 -> Rank 0 ...
        self.param_to_rank = {}
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 把所有 group 里的参数展平成一个列表，用于确定顺序
        all_params = []
        for group in self.param_groups:
            for p in group['params']:
                all_params.append(p)

        # 分配归属权
        for i, p in enumerate(all_params):
            owner = i % self.world_size
            self.param_to_rank[p] = owner

        # 创建本地优化器，只处理归为当前rank的那部分参数
        local_param_groups = []

        for group in self.param_groups:
            # 复制 group 的超参数 (lr, betas 等)，去掉 params
            local_group = {k: v for k, v in group.items() if k != 'params'}

            # 筛选出属于当前 Rank 的参数
            my_params = [p for p in group['params'] if self.param_to_rank[p] == self.rank]

            if my_params:
                local_group['params'] = my_params
                local_param_groups.append(local_group)

        # 实例化底层优化器 (只包含当前rank的参数 shard)
        self.local_optimizer = self.optimizer_cls(local_param_groups, **self.optimizer_kwargs)

    def step(self, closure=None, **kwargs):
        """
        执行一步优化，并同步参数。
        """
        # 本地更新
        # 只更新属于当前rank的那部分参数
        # 这一步会创建/更新
        loss = None
        if self.local_optimizer.param_groups:
            loss = self.local_optimizer.step(closure=closure, **kwargs)
        # 全局同步，我们遍历所有参数，让 Owner 把最新值广播给大家。
        for group in self.param_groups:
            for p in group['params']:
                owner = self.param_to_rank[p]
                dist.broadcast(p.data, src=owner)
        return loss

    def add_param_group(self, param_group):
        """
        支持动态添加参数组
        """
        # 添加到全局记录
        super().add_param_group(param_group)
        # 为新参数分配归属权，这里简单的 append 逻辑可能需要重新计算所有映射，我们可以直接往 local_optimizer 里 add_param_group。
        local_group = {k: v for k, v in param_group.items() if k != 'params'}
        my_new_params = []
        pass

#由于之前发现自己的有一个卡跑不了XL规格，所以就不写最后的一个optimizer_state_sharding_accounting了