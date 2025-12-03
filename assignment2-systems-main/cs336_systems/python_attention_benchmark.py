import torch
import timeit
import pandas as pd

from cs336_basics.model import scaled_dot_product_attention


def benchmark_attention():
    BATCH_SIZE = 8
    D_MODELS = [ 16,32,64,128]
    SEQ_LENS = [256, 1024, 2048, 4096,8192]  # 移除了8192和16384
    N_REPEATS = 100
    N_WARMUP = 4

    if not torch.cuda.is_available():
        print("cuda 不可用")
        return 0

    device = torch.device("cuda")
    print(f"正在{torch.cuda.get_device_name(device)}中运行")

    results = []
    attention = scaled_dot_product_attention

    for d_model in D_MODELS:
        for seq_len in SEQ_LENS:
            print(f"当前嵌入维度{d_model}\n当前序列长度{seq_len}", flush=True)

            try:
                q = torch.randn((BATCH_SIZE, seq_len, d_model), device=device, dtype=torch.float16, requires_grad=True)
                k = torch.randn((BATCH_SIZE, seq_len, d_model), device=device, dtype=torch.float16, requires_grad=True)
                v = torch.randn((BATCH_SIZE, seq_len, d_model), device=device, dtype=torch.float16, requires_grad=True)

                for _ in range(N_WARMUP):
                    output = attention(q, k, v)
                    output.backward(torch.ones_like(output), retain_graph=False)
                    q.grad, k.grad, v.grad = None, None, None


                total_fw_time = 0
                total_bw_time = 0

                torch.cuda.reset_peak_memory_stats(device)
                output = attention(q, k, v)
                peak_mem_GB = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                output.backward(torch.ones_like(output))
                q.grad, k.grad, v.grad = None, None, None

                # 开始计时
                for _ in range(N_REPEATS):
                    torch.cuda.synchronize()
                    start_time_fw = timeit.default_timer()

                    output = attention(q, k, v)

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


if __name__ == "__main__":
    benchmark_attention()