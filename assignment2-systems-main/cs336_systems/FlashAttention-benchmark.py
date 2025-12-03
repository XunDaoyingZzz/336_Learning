from __future__ import annotations

import FlashAttention_triton_fb

import pandas as pd
import triton

import torch

attention=FlashAttention_triton_fb.FlashAttention2


def benchmark_flash_attention2():
    BATCH_SIZE = 8
    D_MODELS = [16, 32, 64, 128,256]
    SEQ_LENS = [256, 1024, 2048, 4096, 8192,16384]
    IS_CAUSAL = [False, True]
    DTYPE = torch.float16 #准备混合精度
    N_REPEATS = 100
    N_WARMUP = 4

    if not torch.cuda.is_available():
        print("CUDA 不可用，无法进行 基准测试。")
        return

    device = torch.device("cuda")
    print(f"正在 {torch.cuda.get_device_name(device)} 中运行")

    torch.set_float32_matmul_precision('high')   #启用float32运算获得精度（混合精度）
    results=[]

    for d_model in D_MODELS:
        for seq_len in SEQ_LENS:

            for is_causal in IS_CAUSAL:
                causal_str = "True" if is_causal else "False"
                print(f"当前嵌入维度{d_model}||当前序列长度{seq_len}||当前是否掩码{causal_str}")
                try:
                    q = torch.randn((BATCH_SIZE, seq_len, d_model), dtype=DTYPE, device=device, requires_grad=True)
                    k = torch.randn((BATCH_SIZE, seq_len, d_model), dtype=DTYPE, device=device, requires_grad=True)
                    v = torch.randn((BATCH_SIZE, seq_len, d_model), dtype=DTYPE, device=device, requires_grad=True)
                    grad_output = torch.randn((BATCH_SIZE, seq_len, d_model), dtype=DTYPE, device=device)
                    #同步+峰值内存清除
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)

                    #warmup是预热次数，rep是重复计算次数,quantiles是返回时间分位数的p值，fast_flush是保持独立性的指标
                    fwd_latency = triton.testing.do_bench(lambda: attention.apply(q, k, v, is_causal),warmup=N_WARMUP,rep=100,quantiles=[0.5],fast_flush=True)
                    # 我们通常关注中位数 (0.5 quantile)
                    peak_mem_fwd_GB=torch.cuda.max_memory_allocated()/(1024**3)
                    median_fwd_ms = fwd_latency * 1000



                    # 反向传播前需要先执行一次前向
                    output = attention.apply(q, k, v, is_causal)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)

                    bwd_latency = triton.testing.do_bench(lambda: output.backward(grad_output, retain_graph=True),warmup=N_WARMUP,rep=100,quantiles=[0.5],fast_flush=True)
                    median_bwd_ms = bwd_latency * 1000
                    peak_mem_bwd_GB=torch.cuda.max_memory_allocated()/(1024**3)


                    print(
                        f"前向中位用时{median_fwd_ms:.2f}ms,反向中位用时{median_bwd_ms:.2f}ms,前后向峰值内存分别为{peak_mem_fwd_GB:.2f}GB,{peak_mem_bwd_GB:.2f}GB")
                    results.append({
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "is_causal": causal_str,
                        "forward time":f"{median_fwd_ms:.2f}ms",
                        "backward time":f"{median_bwd_ms:.2f}ms",
                        "forward peak Mem":f"{peak_mem_fwd_GB:.2f}GB",
                        "backward peak Mem":f"{peak_mem_bwd_GB:.2f}GB",
                    })

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("gpu内存不足")
                        results.append({
                            "d_model": d_model,
                            "seq_len": seq_len,
                            "is_causal": causal_str,
                            "forward time":"BOOM",
                            "backward time":"BOOM",
                            "forward peak Mem":"BOOM",
                            "backward peak Mem":"BOOM",
                        })
                    else:
                        print(f"发生未知错误:{e}")
                        results.append({
                            "d_model": d_model,
                            "seq_len": seq_len,
                            "is_causal": causal_str,
                            "forward time": "BOOM",
                            "backward time": "BOOM",
                            "forward peak Mem": "BOOM",
                            "backward peak Mem": "BOOM",
                        })
                finally:
                    del q,k,v,grad_output
                    if "output" in locals():del output
                    torch.cuda.empty_cache()

    dataframe=pd.DataFrame(results)
    print("\n\n---Benchmark FlashAttention2---")
    print(dataframe.to_markdown(index=False))


if __name__ == "__main__":
    benchmark_flash_attention2()

# def benchmark():
#     BATCH_SIZE = 8
#     D_MODELS = [16, 32, 64, 128]
#     SEQ_LENS = [256, 1024, 2048, 4096, 8192]  # 移除了8192和16384
#     N_REPEATS = 100
#     N_WARMUP = 4
#
#     if not torch.cuda.is_available():
#         print("cuda 不可用")
#         return 0
#
#     device = torch.device("cuda")
#     print(f"正在{torch.cuda.get_device_name(device)}中运行")
#
#     results = []
#
#
#     for d_model in D_MODELS:
#         for seq_len in SEQ_LENS:
#             print(f"当前嵌入维度{d_model}\n当前序列长度{seq_len}", flush=True)
#
#             try:
#                 q=torch.randn((BATCH_SIZE, seq_len, d_model),device=device,dtype=torch.float32,requires_grad=True)
#                 k=torch.randn((BATCH_SIZE, seq_len, d_model),device=device,dtype=torch.float32,requires_grad=True)
#                 v=torch.randn((BATCH_SIZE, seq_len, d_model),device=device,dtype=torch.float32,requires_grad=True)
#                 #热身阶段
#                 for _ in range(N_WARMUP):
#                     output = attention.apply(q, k, v)
#                     output.backward(torch.ones_like(q))
#
#                 total_fw_time = 0
#                 total_bw_time = 0
#
#                 torch.cuda.reset_peak_memory_stats(device)
#
#                 output = attention.apply(q, k, v)
#                 peak_mem_GB = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
#                 output.backward(torch.ones_like(q))
#
#
#                 #正式的前后向传播
#                 for _ in range(N_REPEATS):
#                     torch.cuda.synchronize()
#                     start_time_fw = timeit.default_timer()
#
#                     output = attention.apply(q, k, v)
#
#                     torch.cuda.synchronize()
#                     end_time_fw = timeit.default_timer()
#
#                     grad_output = torch.randn_like(q)
#
#                     torch.cuda.synchronize()
#                     start_time_bw = timeit.default_timer()
#
#                     output.backward(grad_output)
#
#                     torch.cuda.synchronize()
#                     end_time_bw = timeit.default_timer()
#
#                     total_fw_time += (end_time_fw - start_time_fw)
#                     total_bw_time += (end_time_bw - start_time_bw)
#
#                     q.grad, k.grad, v.grad = None, None, None
#
#                 mean_fw_time = total_fw_time / N_REPEATS
#                 mean_bw_time = total_bw_time / N_REPEATS
#
#                 print(
#                     f"前向平均用时{mean_fw_time * 1000:.2f}ms, 反向平均用时{mean_bw_time * 1000:.2f}ms, 峰值内存{peak_mem_GB:.2f}GB")
#                 results.append({
#                     "d_model": d_model,
#                     "seq_len": seq_len,
#                     "forward time": f"{mean_fw_time * 1000:.2f} ms",
#                     "backward time": f"{mean_bw_time * 1000:.2f} ms",
#                     "Peak Mem": f"{peak_mem_GB:.2f} GB"
#                 })
#
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print("gpu内存不足")
#                     results.append({
#                         "d_model": d_model,
#                         "seq_len": seq_len,
#                         "forward time": "BOOM",
#                         "backward time": "BOOM",
#                         "Peak Mem": "BOOM"
#                     })
#                 else:
#                     print(f"发生未知错误: {e}")
#
#             finally:
#                 del q, k, v
#                 if "output" in locals(): del output
#                 torch.cuda.empty_cache()
#
#     dataframe = pd.DataFrame(results)
#     print("\n\n---Benchmark Flash-attention---")
#     print(dataframe.to_markdown(index=False))



