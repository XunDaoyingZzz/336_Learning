import torch
import torch.nn as nn
import torch.distributed as dist


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float = 5):
        """
        param bucket_size_mb: 每个桶的目标大小 (MB)
        """
        super().__init__()
        self.module = module
        # 将 MB 转换为 字节
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # 构建 Buckets (按参数逆序)
        params_reversed = list(reversed(list(self.module.parameters())))

        self.buckets = []
        current_bucket_params = []
        current_bucket_size = 0

        for param in params_reversed:
            if not param.requires_grad:
                continue

            # 计算参数占用的字节数
            param_bytes = param.numel() * param.element_size()  #.numel()是元素数量，element_size()是单个元素的字节数

            current_bucket_params.append(param)                 #往桶里装参数
            current_bucket_size += param_bytes                  #累计容量

            # 如果当前桶满了，就封桶
            if current_bucket_size >= self.bucket_size_bytes:
                self._create_bucket(current_bucket_params)
                current_bucket_params = []                      #重置current_bucket
                current_bucket_size = 0                         #重置容量

        # 处理剩余的参数
        if current_bucket_params:
            self._create_bucket(current_bucket_params)

        #  注册 Hook
        for bucket_idx, bucket in enumerate(self.buckets):
            for param in bucket['params']:
                param.register_post_accumulate_grad_hook(self._make_hook(bucket_idx, param)) #make_hook把bucket_idx与hook绑定，hook触发的时候知道是哪个桶的哪个参数

    def _create_bucket(self, params):
        total_numel = sum(p.numel() for p in params) #传入一个params列表，计算params里面共有多少个数字元素

        ref_param = params[0]          #拿一个参考的数据来记录类型和设备
        # 预分配一个连续的、扁平的大 Buffer
        # 必须设为不需要梯度，因为我们要手动操作它
        flat_buffer = torch.zeros(
            total_numel,
            device=ref_param.device,
            dtype=ref_param.dtype
        )

        self.buckets.append({
            'params': params,               # 桶里的参数列表
            'buffer': flat_buffer,          # 通信用的连续显存
            'handle': None,                 # 异步通信句柄
            'ready_count': 0,               # 计数器：桶里几个参数算好了？
            'total_param_count': len(params)# 桶里总共有几个参数
        })

    def _make_hook(self, bucket_idx, param):  #返回一个hook_fn，bucket_idx会被闭包捕获
        def hook_fn(p):
            bucket = self.buckets[bucket_idx] #通过bucket_idx找到对应的桶
            bucket['ready_count'] += 1        #然后又有一个参数算好了

            # 检查桶是否满了
            if bucket['ready_count'] == bucket['total_param_count']:
                self._all_reduce_bucket(bucket)             #如果算好就开始执行all_reduce

        return hook_fn

    def _all_reduce_bucket(self, bucket):
        # 将分散的 p.grad 拷贝到连续的 buffer 中
        offset = 0
        tensor_list = bucket['params']
        buffer = bucket['buffer']

        for param in tensor_list:
            if param.grad is not None:
                grad_data = param.grad.data.view(-1)  # 展平
                numel = grad_data.numel()             # 计数
                # 拷贝到 buffer 的对应切片
                buffer[offset: offset + numel].copy_(grad_data)
                offset += numel
            else:
                offset += param.numel()
        # 循环结束，buffer装满了这个桶的梯度
        buffer.div_(dist.get_world_size())  # buffer 里所是有卡的梯度加起来除以卡数

        # 发起异步通信
        handle = dist.all_reduce(buffer, op=dist.ReduceOp.SUM, async_op=True)
        bucket['handle'] = handle   #保存好当前handle，方便在finish_gradient_synchronization里检查

    def forward(self, *inputs, **kwargs):
        # 每次 Forward 前重置桶状态
        for bucket in self.buckets:
            bucket['ready_count'] = 0
            bucket['handle'] = None
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        等待通信，并手动解包；这个函数在 optimizer.step()前调用，等待所有桶都算完并把数据放回p.grad才进行更新步骤
        """
        for bucket in self.buckets:
            # 等待通信完成
            if bucket['handle'] is not None:
                bucket['handle'].wait()

            # 从 buffer 切片拷贝回 p.grad
            offset = 0
            buffer = bucket['buffer']

            for param in bucket['params']:
                if param.grad is not None:
                    numel = param.numel()
                    # 从 buffer 取出切片
                    grad_slice = buffer[offset: offset + numel]
                    # 变形并拷贝回原梯度张量
                    param.grad.data.copy_(grad_slice.view(param.grad.shape))
                    offset += numel
                else:
                    offset += param.numel()