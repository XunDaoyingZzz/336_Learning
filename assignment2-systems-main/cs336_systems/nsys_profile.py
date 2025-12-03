from __future__ import annotations

import argparse
import timeit
import statistics
from typing import Tuple

import torch
import torch.nn as nn
from einops import einsum
import math

import cs336_basics.model
from cs336_basics.nn_utils import cross_entropy,softmax
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

from jaxtyping import Float, Bool, Int
from collections.abc import Callable

from torch import Tensor
import torch.cuda.nvtx as nvtx

def create_random_batch(batch_size:int,context_length:int,vocab_size:int,device:str)->torch.Tensor:
    return torch.randint(0,vocab_size,(batch_size,context_length),device=device)

@nvtx.range("step")
def new_step(self, closure: Callable | None = None):
    with nvtx.range("step"):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]
                alpha = group["lr"]
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                t = state.get("t", 1)
                prev_m_t = state.get("m", torch.zeros_like(grad))
                prev_v_t = state.get("v", torch.zeros_like(grad))

                m_t = beta_1 * prev_m_t + ((1 - beta_1) * grad)
                v_t = beta_2 * prev_v_t + ((1 - beta_2) * torch.square(grad))

                alpha_t = alpha * (math.sqrt(1 - (beta_2**t)) / (1 - (beta_1**t)))
                p.data -= alpha_t * m_t / (torch.sqrt(v_t) + eps)

                p.data -= alpha * group["weight_decay"] * p.data

                state["m"] = m_t
                state["v"] = v_t
                state["t"] = t + 1

        return loss

AdamW.step=new_step


@nvtx.range("benchmarking")
def benchmark_model(
        model:nn.Module,
        batch:torch.Tensor,
        warmup_steps:int,
        benchmark_steps:int,
        measure_backward:bool=True,
        device:str="cuda"
)->Tuple[float,float]:
    model.train() if measure_backward else model.eval() #如果计算反向传播，则切换为训练模式，反之为评估模式
    print(f"正在运行{warmup_steps}个预热步骤...")
    for _ in range(warmup_steps):
        if measure_backward:
            model.zero_grad()

        logits=model(batch) #前向传播得到模型结果, (batch size, sequence_length, vocab_size)
        if measure_backward:#如果计算反向传播
            targets=torch.randint(0,model.vocab_size,(batch.size(0),batch.size(1)),device=device) #造一个同形的数据
            loss=cross_entropy(
                logits.view(-1,logits.size(-1)), #把(b s v)展成 (b*s v)
                targets.view(-1)  #这个是b*s的，v是各个词汇的概率，这里的每个元素都会提供一个索引，这个索引是v中的某一个，由此求熵
            )
            loss.backward() #反向传播

        if device=="cuda":
            torch.cuda.synchronize() #确保cuda操作完成

    #以上我们进行完了预热

    print(f"正在运行{benchmark_steps}个测试步骤")
    times=[]

    for i in range(benchmark_steps):
        if measure_backward:
            model.zero_grad()

        start_time=timeit.default_timer()
        with nvtx.range("forward pass"):
            logits=model(batch)

        if measure_backward:

            targets=torch.randint(0,model.vocab_size,(batch.size(0),batch.size(1)),device=device)
            loss=cross_entropy(
                logits.view(-1,logits.size(-1)),
                targets.view(-1)
            )
            with nvtx.range("backward pass"):
                loss.backward()

        if device=="cuda":
            torch.cuda.synchronize()
        end_time=timeit.default_timer()
        step_time=end_time-start_time
        times.append(step_time)    #把某次循环cuda上面用时添加到times

        print(f"步骤{i+1}:{step_time:.4f}s")


    mean_time=statistics.mean(times) #用时均值
    std_time=statistics.stdev(times) if len(times)>1 else 0.0  #用时标准差
    print(f"用时均值和标准差是{mean_time},{std_time}")
    return mean_time,std_time


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    with nvtx.range("computing attention scores"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        output=einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return output

cs336_basics.model.scaled_dot_product_attention=annotated_scaled_dot_product_attention

def main():
    model_configs = {
        "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }


    parser=argparse.ArgumentParser(description="对模型的基准测试")
    parser.add_argument("--benchmark_steps",type=int,default=10,help="测试次数")
    parser.add_argument("--vocab_size",type=int,default=10_000,help="词汇表大小")
    parser.add_argument("--batch_size",type=int,default=4,help="一个批大小")
    parser.add_argument("--rope_theta",type=int,default=10_000,help="ROPE的theta")
    parser.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu",help="设备")
    parser.add_argument("--warmup_steps",type=int,default=5,help="热身次数")
    parser.add_argument("--context_length",type=int,nargs="+",default=[128,256,512,1024],help="截断长度")#nargs是参数的数量规范，+表示接受一个或多个值
    parser.add_argument("--models",type=str,nargs="+",default=list(model_configs.keys()),help="几个规模的模型")
    parser.add_argument("--forward_only",action="store_true",help="是否仅测量前向传播")

    args=parser.parse_args()

    if args.device=="cuda" and not torch.cuda.is_available():
        #再次检查一遍cuda是否可用
        print("检查发现cuda不可用,换回cpu")
        args.device='cpu'

    for model_name in args.models:
        config_params=model_configs[model_name] #取出模型超参数
        print(f"\n当前测试的模型规模{model_name}")
        #print("此规模的各个参数")
        max_context_length=max(args.context_length)
        print("初始化模型ing...")
        try:
            model=BasicsTransformerLM(
                vocab_size=args.vocab_size,
                context_length=max_context_length,
                rope_theta=args.rope_theta,
                **config_params
            ).to(args.device)
            optimizer=AdamW(model.parameters())

            for context_length_item in args.context_length:
                print(f"\n当前截断长度{context_length_item}")
                forward_times=[]
                backward_times=[]
                print("创建随机的样本")
                batch=create_random_batch(args.batch_size,context_length_item,args.vocab_size,args.device)
                print("开始当次测试")
                mean_time,std_time=benchmark_model(
                    model=model,
                    batch=batch,
                    warmup_steps=args.warmup_steps,
                    benchmark_steps=args.benchmark_steps,
                    measure_backward=not args.forward_only,
                    device=args.device
                )
                pass_type="前向+反向传播" if not args.forward_only else "仅前向传播"
                print(f"{pass_type}\n")
        except torch.OutOfMemoryError:
            print("内存消耗过大，跳过代码块")
            pass

if __name__=="__main__":
    main()


