import pandas as pd

import torch
import timeit

#ok triton无法直接在windows上使用

from cs336_basics.model import scaled_dot_product_attention

# def benchmark_attention():
#     BATCH_SIZE=8
#     D_MODELS=[16,32,64,128]
#     SEQ_LENS=[256,1024,4096,8192,16384]
#     N_REPEATS=100
#     N_WARMUP=4
#
#     if not torch.cuda.is_available():
#         print("cuda 不可用")
#         return 0
#
#     device=torch.device("cuda")
#     print(f"正在{torch.cuda.get_device_name(device)}中运行")
#
#     results=[]
#
#     attention=scaled_dot_product_attention
#
#     for d_model in D_MODELS:
#         for seq_len in SEQ_LENS:
#             print(f"当前嵌入维度{d_model}\n当前序列长度{seq_len}")
#
#             try:
#
#                 q=torch.randn((BATCH_SIZE,seq_len,d_model),device=device,dtype=torch.float16)
#                 k=torch.randn((BATCH_SIZE,seq_len,d_model),device=device,dtype=torch.float16)
#                 v=torch.randn((BATCH_SIZE,seq_len,d_model),device=device,dtype=torch.float16)
#
#                 for _ in range(N_WARMUP):
#                     output=attention(q,k,v)
#                     output.backward(torch.ones_like(output),retain_graph=True) #保留梯度
#
#                 torch.cuda.synchronize()
#                 start_time_fw=timeit.default_timer()
#                 for _ in range(N_REPEATS):
#                     output=attention(q,k,v)
#                 torch.cuda.synchronize()
#                 end_time_fw=timeit.default_timer()
#                 mean_fw_time=(end_time_fw-start_time_fw)/N_REPEATS
#                 torch.cuda.reset_peak_memory_stats(device) #重置峰值来记录反向传播
#                 output=attention(q,k,v)
#                 peak_mem_GB=torch.cuda.max_memory_allocated(device)/(1024**3)
#
#                 grad_output=torch.randn_like(output)
#                 torch.cuda.synchronize()
#                 start_time_bw=timeit.default_timer()
#                 for _ in range(N_REPEATS):
#                     output.backward(grad_output,retain_graph=True)
#                 torch.cuda.synchronize()
#                 end_time_bw=timeit.default_timer()
#                 mean_bw_time=(end_time_bw-start_time_bw)/N_REPEATS
#
#                 print(f"前向平均用时{mean_fw_time:.2f}s,反向平均用时{mean_bw_time:.2f}s")
#                 results.append({
#                     "d_model":d_model,
#                     "seq_len":seq_len,
#                     "forward time":f"{mean_fw_time:.2f}",
#                     "backward time":f"{mean_bw_time:.2f}",
#                     "Peak Mem":f"{peak_mem_GB:.2f}"
#                 })
#
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print("gpu内存不足")
#                     results.append(
#                         {
#                             "d_model":d_model,
#                     "seq_len":seq_len,
#                     "forward time":"BOOM",
#                     "backward time":"BOOM",
#                     "Peak Mem":"BOOM"
#                         }
#                     )
#             finally:
#                 del q,k,v
#                 if "output" in locals():del output
#                 torch.cuda.empty_cache()
#
#     dataframe=pd.DataFrame(results)
#     print("\n\n---Benchmark attention---")
#     print(dataframe.to_markdown(index=False))

#有点问题 迭代100次
#还是不太会 搞不明白
#下面搞compile版本的
def benchmark_attention():
    BATCH_SIZE = 8
    D_MODELS = [ 16,32,64,128]
    SEQ_LENS = [256, 1024, 2048, 4096,8192,16384]  # 移除了8192和16384
    N_REPEATS = 100
    N_WARMUP = 6

    if not torch.cuda.is_available():
        print("cuda 不可用")
        return 0

    device = torch.device("cuda")
    print(f"正在{torch.cuda.get_device_name(device)}中运行")

    results = []
    compiled_attention=torch.compile(scaled_dot_product_attention)

    for d_model in D_MODELS:
        for seq_len in SEQ_LENS:
            print(f"当前嵌入维度{d_model}\n当前序列长度{seq_len}", flush=True)

            try:
                q = torch.randn((BATCH_SIZE, seq_len, d_model), device=device, dtype=torch.float16, requires_grad=True)
                k = torch.randn((BATCH_SIZE, seq_len, d_model), device=device, dtype=torch.float16, requires_grad=True)
                v = torch.randn((BATCH_SIZE, seq_len, d_model), device=device, dtype=torch.float16, requires_grad=True)

                for _ in range(N_WARMUP):
                    output = compiled_attention(q, k, v)
                    output.backward(torch.ones_like(output), retain_graph=False)
                    q.grad, k.grad, v.grad = None, None, None


                total_fw_time = 0
                total_bw_time = 0

                torch.cuda.reset_peak_memory_stats(device)
                output = compiled_attention(q, k, v)
                peak_mem_GB = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                output.backward(torch.ones_like(output))
                q.grad, k.grad, v.grad = None, None, None

                # 开始计时
                for _ in range(N_REPEATS):
                    torch.cuda.synchronize()
                    start_time_fw = timeit.default_timer()

                    output = compiled_attention(q, k, v)

                    torch.cuda.synchronize()
                    end_time_fw = timeit.default_timer()

                    grad_output = torch.randn_like(output)

                    torch.cuda.synchronize()
                    start_time_bw = timeit.default_timer()

                    output.backward(grad_output, retain_graph=False)

                    torch.cuda.synchronize()
                    end_time_bw = timeit.default_timer()

                    total_fw_time += (end_time_fw - start_time_fw)
                    total_bw_time += (end_time_bw - start_time_bw)

                    # *** 关键改动 3: (可选但推荐) 清空梯度 ***
                    q.grad, k.grad, v.grad = None, None, None

                mean_fw_time = total_fw_time / N_REPEATS
                mean_bw_time = total_bw_time / N_REPEATS

                print(
                    f"前向平均用时{mean_fw_time * 1000:.2f}ms, 反向平均用时{mean_bw_time * 1000:.2f}ms, 峰值内存{peak_mem_GB:.2f}GB")
                results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "forward time": f"{mean_fw_time * 1000:.2f} ms",
                    "backward time": f"{mean_bw_time * 1000:.2f} ms",
                    "Peak Mem": f"{peak_mem_GB:.2f} GB"
                })

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("gpu内存不足")
                    results.append({
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "forward time": "BOOM",
                        "backward time": "BOOM",
                        "Peak Mem": "BOOM"
                    })
                else:
                    print(f"发生未知错误: {e}")
            finally:
                del q, k, v
                if "output" in locals(): del output
                torch.cuda.empty_cache()

    dataframe = pd.DataFrame(results)
    print("\n\n---Benchmark attention---")
    print(dataframe.to_markdown(index=False))
    # headers = {
    #     "d_model": "d_model",
    #     "seq_len": "seq_len",
    #     "forward time": "forward time",
    #     "backward time": "backward time",
    #     "Peak Mem": "Peak Mem"
    # }
    # col_widths = {key: len(value) + 4 for key, value in headers.items()}
    #
    # # 2. 打印表头
    # header_line = (
    #     f"{headers['d_model']:<{col_widths['d_model']}}"
    #     f"{headers['seq_len']:<{col_widths['seq_len']}}"
    #     f"{headers['forward time']:<{col_widths['forward time']}}"
    #     f"{headers['backward time']:<{col_widths['backward time']}}"
    #     f"{headers['Peak Mem']:<{col_widths['Peak Mem']}}"
    # )
    # print("\n\n---Benchmark attention---")
    # print(header_line)
    # print("-" * len(header_line))  # 打印一条分隔线
    #
    # # 3. 遍历结果并格式化打印每一行
    # for r in results:
    #     row_line = (
    #         f"{r['d_model']:<{col_widths['d_model']}}"
    #         f"{r['seq_len']:<{col_widths['seq_len']}}"
    #         f"{str(r['forward time']):<{col_widths['forward time']}}"
    #         f"{str(r['backward time']):<{col_widths['backward time']}}"
    #         f"{str(r['Peak Mem']):<{col_widths['Peak Mem']}}"
    #     )
    #     print(row_line)


if __name__=="__main__":
    benchmark_attention()

#还是不太会 搞不明白