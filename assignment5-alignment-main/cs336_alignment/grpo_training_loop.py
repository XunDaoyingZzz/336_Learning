import os
import json
import torch
import torch.distributed as dist
import torch.nn.utils as nn_utils
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import random
from cs336_alignment.utils import grpo_microbatch_train_step,compute_grpo_clip_loss,compute_group_normalized_rewards

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
    print("[Rank 0]初始化GRPO训练环境...")
    device = torch.device("cuda:0")

    model_path = "models/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    #config没按讲义来，按自己的机器做了调整
    n_grpo_steps = 200
    learning_rate = 1e-5
    advantage_eps = 1e-6
    rollout_batch_size = 32  #本地调整256->32
    group_size = 8  #讲义默认8
    epochs_per_rollout_batch = 1  #讲义默认1(On-policy)
    train_batch_size = 32  #必须等于rollout_batch_size
    gradient_accumulation_steps = 16  #32/16=2(Microbatch size)
    loss_type = "reinforce_with_baseline"
    use_std_normalization =True
    cliprange =0.2

    #断言检查，防止配置错位
    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size

    optimizer = AdamW(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95)
    )

    #加载数据
    print("[Rank 0]加载数据集...")
    with open("cs336_alignment/prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_raw = load_jsonl("data/gsm8k/train.jsonl")
    val_raw = load_jsonl("data/gsm8k/test.jsonl")[:50]  #验证集依然截断为50条

    val_questions = [ex["question"] for ex in val_raw]
    val_ground_truths = [
        ex["answer"].split("####")[1].strip() if "####" in ex["answer"] else ex["answer"].strip()
        for ex in val_raw
    ]
    formatted_val_prompts = [prompt_template.replace("{question}", q) for q in val_questions]

    #GRPO训练主循环
    for step in range(1, n_grpo_steps + 1):
        print(f"[Rank 0]开始GRPO Step{step}/{n_grpo_steps}")

        #抽题并复制Group Size次
        batch_raw = random.sample(train_raw, n_prompts_per_rollout_batch)
        flat_prompts, flat_gts = [], []
        for ex in batch_raw:
            q = prompt_template.replace("{question}", ex["question"])
            gt = ex["answer"].split("####")[1].strip() if "####" in ex["answer"] else ex["answer"].strip()
            flat_prompts.extend([q] * group_size)
            flat_gts.extend([gt] * group_size)

        #呼叫副机生成Rollouts
        print(f"[Rank 0]请求副机生成{rollout_batch_size}条回答...")
        dist.broadcast_object_list(["GRPO_GENERATE"], src=0)
        sync_weights_to_vllm_across_network(policy_model=policy_model, rank=0)
        dist.broadcast_object_list([flat_prompts], src=0)

        responses_container = [None]
        dist.broadcast_object_list(responses_container, src=1)
        flat_responses = responses_container[0]

        #为了避免大模型一直不输出 </answer>，这里必须安全截断
        clean_responses = [r.split("</answer>")[0] + "</answer>" if "</answer>" in r else r for r in flat_responses]

        #计算优势
        advantages, raw_rewards, group_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=clean_responses,
            repeated_ground_truths=flat_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization
        )
        print(f"[Rank 0]本轮Reward均值: {group_meta['reward_mean']:.4f}")

        advantages = advantages.to(device)
        raw_rewards = raw_rewards.to(device)

        #提取旧策略对数概率(用于Off-policy/GRPO-Clip)
        #现在是On-policy，统一提取，方便后续直接改参数就能跑Clip实验
        old_log_probs_list = []
        policy_model.eval()
        with torch.inference_mode():
            for idx in range(0, rollout_batch_size, micro_train_batch_size):
                batch_p = flat_prompts[idx: idx + micro_train_batch_size]
                batch_r = clean_responses[idx: idx + micro_train_batch_size]

                tensors = tokenize_prompt_and_output(batch_p, batch_r, tokenizer)
                input_ids = tensors["input_ids"].to(device)
                labels = tensors["labels"].to(device)

                log_probs_dict = get_response_log_probs(policy_model, input_ids, labels, return_token_entropy=False)
                old_log_probs_list.append(log_probs_dict["log_probs"])

        #这里的old_log_probs我们并不拼接，而是在微调循环里按切片取出来用

        #微调循环
        policy_model.train()
        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()
            micro_idx = 0

            #为了保持advantages和responses 对应，我们不再shuffle
            for idx in range(0, rollout_batch_size, micro_train_batch_size):
                batch_p = flat_prompts[idx: idx + micro_train_batch_size]
                batch_r = clean_responses[idx: idx + micro_train_batch_size]

                #切出对应的advantage 和raw_rewards
                batch_adv = advantages[idx: idx + micro_train_batch_size]
                batch_raw = raw_rewards[idx: idx + micro_train_batch_size]
                batch_old_log_probs = old_log_probs_list[micro_idx]

                tensors = tokenize_prompt_and_output(batch_p, batch_r, tokenizer)
                input_ids = tensors["input_ids"].to(device)
                labels = tensors["labels"].to(device)
                response_mask = tensors["response_mask"].to(device)

                log_probs_dict = get_response_log_probs(policy_model, input_ids, labels, return_token_entropy=True)
                policy_log_probs = log_probs_dict["log_probs"]

                loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=batch_raw,
                    advantages=batch_adv,
                    old_log_probs=batch_old_log_probs,
                    cliprange=cliprange
                )

                micro_idx += 1

                #梯度累加更新
                if micro_idx % gradient_accumulation_steps == 0:
                    nn_utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    print(f"[Epoch {epoch + 1}]Micro-step {micro_idx}, Loss: {meta['loss'].item():.4f}, Entropy: {log_probs_dict.get('token_entropy', torch.tensor(0)).mean().item():.4f}")

        #周期性评估(每5步验证一次)
        if step % 5 == 0 or step == 1:
            print(f"\n[Rank 0]触发验证(Step {step})")
            dist.broadcast_object_list(["EVAL"], src=0)
            sync_weights_to_vllm_across_network(policy_model=policy_model, rank=0)
            dist.broadcast_object_list([formatted_val_prompts], src=0)

            responses_container = [None]
            dist.broadcast_object_list(responses_container, src=1)
            generated_texts = responses_container[0]

            correct_count = 0
            for gt, a in zip(val_ground_truths, generated_texts):
                reward_dict = r1_zero_reward_fn(a, gt)
                if reward_dict.get("answer_reward", 0.0) > 0.0:
                    correct_count += 1

            accuracy = correct_count / len(val_questions)
            print(f"[Rank 0]Step {step} 验证准确率: {accuracy * 100:.2f}% ({correct_count}/{len(val_questions)})\n")

    print("[Rank 0]GRPO训练完成，发送STOP信号...")
    dist.broadcast_object_list(["STOP"], src=0)


#副节点vLLM推理

def run_worker_generator(vllm_model):
    print("[Rank 1]进入监听循环，等待主机指令 (GRPO 模式)...")

    #GRPO收集数据阶段的配置
    #temperature=1.0，并且按照讲义要求加上min_tokens=4 避免空字符串崩溃
    sampling_params_grpo = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        min_tokens=4
    )

    #Eval验证阶段的配置
    #评估测试集时也必须使用 temperature=1.0
    sampling_params_eval = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
    )

    while True:
        #监听主机的命令
        cmd_container = [None]
        dist.broadcast_object_list(cmd_container, src=0)
        cmd = cmd_container[0]

        if cmd == "GRPO_GENERATE":
            print("\n[Rank 1]收到GRPO_GENERATE，准备接收最新权重...")
            sync_weights_to_vllm_across_network(llm_model=vllm_model, rank=1)

            #接收Prompts
            prompts_container = [None]
            dist.broadcast_object_list(prompts_container, src=0)
            prompts = prompts_container[0]

            print(f"[Rank 1]使用vLLM高温(1.0)生成{len(prompts)}条GRPO回复...")
            #关闭tqdm进度条避免双机日志穿插污染
            outputs = vllm_model.generate(prompts, sampling_params_grpo, use_tqdm=False)
            generated_texts = [out.outputs[0].text for out in outputs]

            print("[Rank 1]生成完成，回传结果...")
            dist.broadcast_object_list([generated_texts], src=1)

        elif cmd == "EVAL":
            print("\n[Rank 1]收到EVAL，准备接收最新权重...")
            sync_weights_to_vllm_across_network(llm_model=vllm_model, rank=1)

            #接收验证集Prompts
            prompts_container = [None]
            dist.broadcast_object_list(prompts_container, src=0)
            prompts = prompts_container[0]

            print(f"[Rank 1]使用vLLM验证生成{len(prompts)}条回复...")
            outputs = vllm_model.generate(prompts, sampling_params_eval, use_tqdm=False)
            generated_texts = [out.outputs[0].text for out in outputs]

            print("[Rank 1]验证生成完成，回传结果...")
            dist.broadcast_object_list([generated_texts], src=1)

        elif cmd == "STOP":
            print("[Rank 1]收到STOP，任务圆满结束，退出监听循环。")
            break


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
    print(f"[Rank {rank}]正在初始化 dist.init_process_group (backend=nccl)...")
    dist.init_process_group(backend="nccl")
    print(f"[Rank {rank}]NCCL 通信组初始化成功！")

    if rank == 0:
        run_master_trainer()
    else:
        run_worker_generator(vllm_model)

    dist.destroy_process_group()
    print(f"[Rank {rank}] 进程正常退出。")


if __name__ == "__main__":
    main()