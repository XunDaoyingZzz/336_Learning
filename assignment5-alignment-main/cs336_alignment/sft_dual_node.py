import os
import json
import torch
import torch.distributed as dist
import torch.nn.utils as nn_utils
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import random

from cs336_alignment.utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    compute_entropy
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


#vLLM初始化工具
#在调用前，调用方负责清除dist相关环境变量
def init_vllm_isolated(model_id, device="cuda:0", seed=42, gpu_memory_utilization=0.85):
    """
    以单机独立模式启动vLLM，不加入任何已有的ProcessGroup。
    调用本函数前，必须确保通信的设置已从环境变量中移除。
    vLLM在 tensor_parallel_size=1 时，默认会走 UniProcExecutor
    但若环境中存在 MASTER_ADDR 等变量（即使 tp=1），某些版本会退化到
    MultiprocessingExecutor，后者会调用dist.init_process_group
    我们的策略：清除环境变量+不传distributed_executor_backend，
    让 vLLM 自己选择 UniProcExecutor（单进程，完全不碰 dist）
    enforce_eager=True可以进一步保证不会触发任何并行初始化路径
    """
    llm = LLM(
        model=model_id,
        device=device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        enforce_eager=True,    # 禁用 CUDAGraph，避免触发并行 warmup 路径
        tensor_parallel_size=1,
        #不传distributed_executor_backend，让vLLM自动选UniProcExecutor
    )
    return llm


#跨网络权重同步
def sync_weights_to_vllm_across_network(policy_model=None, llm_model=None, rank=0):
    """
    通过NCCL跨机器同步模型权重。
    Rank 0：发送HuggingFace模型的state_dict。
    Rank 1：接收Tensor并注入vLLM实例。
    """
    if rank == 0:
        sd = policy_model.state_dict()
        metadata = [(k, tuple(v.shape), str(v.dtype)) for k, v in sd.items()]
        dist.broadcast_object_list([metadata], src=0)

        for k, v in sd.items():
            t = v.to("cuda").contiguous()
            dist.broadcast(t, src=0)

    else:  #rank==1
        metadata_container = [None]
        dist.broadcast_object_list(metadata_container, src=0)
        metadata = metadata_container[0]

        #dtype字符串->torch.dtype
        _dtype_map = {
            "torch.bfloat16": torch.bfloat16,
            "torch.float16":  torch.float16,
            "torch.float32":  torch.float32,
        }

        received_sd = {}
        for k, shape, dtype_str in metadata:
            dtype = _dtype_map.get(dtype_str, torch.bfloat16)
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            dist.broadcast(tensor, src=0)
            received_sd[k] = tensor

        # 注入 vLLM 底层模型
        llm_engine_model = (
            llm_model
            .llm_engine
            .model_executor
            .driver_worker
            .model_runner
            .model
        )
        llm_engine_model.load_weights(received_sd.items())

        del received_sd
        torch.cuda.empty_cache()                        #清理掉临时字典防止oom，我们现在副机上运行着vllm和接受的模型权重
        print("[Rank 1]权重注入完毕，显存已清理。")



#主节点训练
def run_master_trainer():
    print("[Rank 0]初始化训练环境...")
    device = torch.device("cuda:0")

    model_path = "models/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    learning_rate = 1e-5
    optimizer = AdamW(policy_model.parameters(), lr=learning_rate)

    train_batch_size = 32
    gradient_accumulation_steps = 8
    micro_batch_size = train_batch_size // gradient_accumulation_steps  # 4
    eval_every_n_steps = 50

    #加载数据
    print("[Rank 0]加载数据集...")
    with open("cs336_alignment/prompts/r1_zero.prompt", "r",encoding="utf-8") as f:
        prompt_template=f.read()

    def load_jsonl(path):
        with open(path, "r",encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_raw=load_jsonl("data/gsm8k/train.jsonl")
    val_raw=load_jsonl("data/gsm8k/test.jsonl")
    val_raw=val_raw[:50]
    val_questions=[ex["question"] for ex in val_raw]
    #val_ground_truths=[ex["answer"] for ex in val_raw]
    val_ground_truths=[]
    for ex in val_raw:
        raw_ans=ex["answer"]
        gt=raw_ans.split("####")[1].strip() if "####" in raw_ans else raw_ans.strip()
        val_ground_truths.append(gt)
    formatted_val_prompts=[
        prompt_template.replace("question",q) for q in val_questions
    ]

    train_data=[]
    for ex in train_raw:
        raw_ans=ex["answer"]
        if "####" in raw_ans:
            reasoning,final_ans=raw_ans.split("####")
            reasoning=reasoning.strip()
            final_ans=final_ans.strip()

            formatted_response=f"<think>{reasoning}</think> <answer>{final_ans}</answer>"
        else:
            formatted_response=f"<answer>{raw_ans.strip()}</answer>"

        train_data.append({"prompt": prompt_template.replace("{question}", ex["question"]), "response": formatted_response})
    random.seed(42)
    random.shuffle(train_data)

    #训练循环
    policy_model.train()
    global_step = 0
    micro_step = 0
    num_epochs=2

    print(f"[Rank 0]开始SFT训练循环,总epoch:{num_epochs}")
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        print(f"\n[RANK 0] 开始Epoch{epoch+1}/{num_epochs}")
        random.shuffle(train_data)

        for idx in range(0, len(train_data), micro_batch_size):
            batch = train_data[idx: idx + micro_batch_size]
            prompts   = [ex["prompt"]   for ex in batch]
            responses = [ex["response"] for ex in batch]

            batch_tensors = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids    = batch_tensors["input_ids"].to(device)
            labels       = batch_tensors["labels"].to(device)
            response_mask = batch_tensors["response_mask"].to(device)

            log_probs_dict = get_response_log_probs(policy_model, input_ids, labels)

            loss, meta = sft_microbatch_train_step(
                policy_log_probs=log_probs_dict["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

            micro_step += 1   #梯度累计步数

            if micro_step % gradient_accumulation_steps == 0:
                nn_utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1 #记录更新次数
                print(f"[Rank 0]Step {global_step}, Loss: {meta['loss'].item():.4f}")

                #触发验证
                if global_step % eval_every_n_steps == 0 or global_step == 1:
                    print(f"\n[Rank 0]触发验证(Step {global_step})")

                    #发送EVAL信号
                    dist.broadcast_object_list(["EVAL"], src=0)

                    #同步权重
                    print("[Rank 0]正在同步模型权重至副机...")
                    sync_weights_to_vllm_across_network(
                        policy_model=policy_model, rank=0
                    )

                    #发送验证 Prompt
                    dist.broadcast_object_list([formatted_val_prompts], src=0)

                    #等待生成结果
                    print("[Rank 0]等待副机生成回复...")
                    responses_container = [None]
                    dist.broadcast_object_list(responses_container, src=1)
                    generated_texts = responses_container[0]

                    print("[Rank 0]收到副机回复，开始计算准确率")
                    correct_count=0
                    for i, (q,gt,a) in enumerate(zip(val_questions,val_ground_truths, generated_texts)):
                        reward_dict=r1_zero_reward_fn(a,gt)
                        answer_reward=reward_dict.get("answer_reward",0.)

                        if answer_reward > 0.:
                            correct_count+=1
                        if i%100==0:
                            print(f"  [{i}] Q: {q[:]}...")
                            print(f"  [{i}] Ground Truth: {gt}")
                            #截断打印长文本
                            print(f"  [{i}] Generation: {a}...")
                            print(f"  [{i}] Answer Reward: {answer_reward}")
                            print("-" * 40)
                    accuracy=correct_count/len(val_questions)
                    print("[Rank 0]计算平均熵")
                    policy_model.eval()
                    total_entropy_sum=0.0
                    total_tokens=0
                    chunk_size=4
                    with torch.no_grad():
                        for i in range(0,len(formatted_val_prompts),chunk_size):
                            c_prompts=formatted_val_prompts[i:i+chunk_size]
                            c_gens=generated_texts[i:i+chunk_size]
                            c_tensors=tokenize_prompt_and_output(c_prompts, c_gens, tokenizer)
                            c_input_ids=c_tensors["input_ids"].to(device)
                            c_mask=c_tensors["response_mask"].to(device)

                            logits=policy_model(c_input_ids).logits
                            entropy=compute_entropy(logits)
                            total_entropy_sum+=(entropy*c_mask).sum().item()
                            total_tokens+=c_mask.sum().item()
                    policy_model.train()

                    avg_entropy=total_entropy_sum/total_tokens if total_tokens>0 else 0.
                    print(f"[Rank 0] Step{global_step}验证结束，准确率: {accuracy*100:.2f}%")
                    print(f"平均响应熵:{avg_entropy:.4f}")
    print("[Rank 0]训练完成，发送停止信号...")
    dist.broadcast_object_list(["STOP"], src=0)


#副节点vLLM 推理

def run_worker_generator(vllm_model):
    print("[Rank 1]进入监听循环，等待主机指令...")

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        #stop=["</answer>"],
    )

    while True:
        cmd_container = [None]
        dist.broadcast_object_list(cmd_container, src=0)
        cmd = cmd_container[0]

        if cmd == "EVAL":
            print("\n[Rank 1]收到EVAL，准备接收最新权重...")
            sync_weights_to_vllm_across_network(llm_model=vllm_model, rank=1)

            prompts_container = [None]
            dist.broadcast_object_list(prompts_container, src=0)
            prompts = prompts_container[0]

            print(f"[Rank 1]使用vLLM生成{len(prompts)}条回复...")
            outputs = vllm_model.generate(prompts, sampling_params, use_tqdm=False)
            generated_texts = [out.outputs[0].text for out in outputs]

            print("[Rank 1]生成完成，回传结果...")
            dist.broadcast_object_list([generated_texts], src=1)

        elif cmd == "STOP":
            print("[Rank 1]收到STOP，退出。")
            break

        else:
            print(f"[Rank 1]未知指令:{cmd}，忽略。")



#主函数
def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(0)

    vllm_model = None

    if rank == 1:
        #临时移除分布式环境变量，让 vLLM 以完全独立模式启动
        #此时dist尚未初始化
        _dist_env_keys = [
            "RANK", "LOCAL_RANK", "WORLD_SIZE",
            "MASTER_ADDR", "MASTER_PORT",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
        ]
        _saved_env = {k: os.environ.pop(k, None) for k in _dist_env_keys}
        print("[Rank 1]环境变量已临时清除，开始独立初始化 vLLM...")

        vllm_model = init_vllm_isolated(
            model_id="models/Qwen2.5-Math-1.5B",
            device="cuda:0",
            seed=42,
            gpu_memory_utilization=0.70,  #留余量给NCCL
        )
        print("[Rank 1]vLLM初始化完成！")

        #还原环境变量，准备加入NCCL通信组
        for k, v in _saved_env.items():
            if v is not None:
                os.environ[k] = v
        print("[Rank 1]环境变量已还原，准备加入NCCL通信组...")

        #如果 vLLM 内部偷偷初始化了 dist，先强制销毁它
        #enforce_eager=True + 清除环境变量后，正常不会走到这里
        if dist.is_initialized():
            print("[Rank 1]检测到vLLM残留的process group，正在销毁...")
            dist.destroy_process_group()
            print("[Rank 1]已销毁vLLM残留process group。")

    #两机在此处同步：Rank 0在此等待，直到Rank 1的vLLM初始化完成
    print(f"[Rank {rank}] 正在初始化 dist.init_process_group (backend=nccl)...")
    dist.init_process_group(backend="nccl")
    print(f"[Rank {rank}] NCCL 通信组初始化成功！")

    if rank == 0:
        run_master_trainer()
    else:
        run_worker_generator(vllm_model)

    dist.destroy_process_group()
    print(f"[Rank {rank}] 进程正常退出。")


if __name__ == "__main__":
    main()