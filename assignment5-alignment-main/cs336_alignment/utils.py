import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Optional,Literal
import numpy as np
#import wandb

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer) -> dict[str, torch.Tensor]:
    """
    分别对prompt和output行分词，将它们拼接在一起，并构造一个只对response生效的mask
    """
    batch_size = len(prompt_strs)

    #分别对prompt和output分词
    prompt_encodings = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_encodings = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    #拼接并记录长度
    concatenated_ids = []
    response_masks = []

    for p_ids, o_ids in zip(prompt_encodings, output_encodings):
        #拼接prompt和output
        concat_id = p_ids + o_ids
        concatenated_ids.append(concat_id)

        #构造mask,prompt部分为0,output部分为1
        mask = [0] * len(p_ids) + [1] * len(o_ids)
        response_masks.append(mask)

    #计算batch内的最大长度以便进行Padding
    max_len = max(len(ids) for ids in concatenated_ids)

    #填充,使用tokenizer.pad_token_id)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input_ids = []
    padded_labels = []
    padded_masks = []

    #对齐并切片
    for ids, mask in zip(concatenated_ids, response_masks):
        #计算需要填充的长度
        pad_len = max_len - len(ids)

        #右侧填充
        padded_ids = ids + [pad_token_id] * pad_len
        padded_mask = mask + [0] * pad_len

        #按照讲义要求进行Shift操作：
        #input_ids去掉了最后一个token
        padded_input_ids.append(padded_ids[:-1])
        #labels是去掉了第一个token
        padded_labels.append(padded_ids[1:])
        #response_mask是针对labels的，所以也要去掉第一个 token
        padded_masks.append(padded_mask[1:])

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "response_mask": torch.tensor(padded_masks, dtype=torch.long)
    }

def compute_entropy(logits:torch.Tensor) -> torch.Tensor:
    log_probs=F.log_softmax(logits, dim=-1)
    probs=torch.exp(log_probs)
    entropy=-torch.sum(probs*torch.log(probs), dim=-1)
    return entropy

def get_response_log_probs(model:PreTrainedModel,
                           input_ids:torch.Tensor,
                           labels:torch.Tensor,
                           return_token_entropy:bool=False,) -> dict[str,torch.Tensor]:
    logits= model(input_ids).logits        #前向传播获取logits

    log_probs=F.log_softmax(logits, dim=-1)#获取log softmax值,这里log_probs是(batch_size,seq_len,vocab_size)
    target_log_probs=torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1) #索引真实标签对应的log概率，原本labels的形状是(batch_size,seq_len)，需要扩充一个维度来gather，最后再还原维度

    result={
        "log_probs":target_log_probs
    }
    if return_token_entropy:
        result["token_entropy"]=compute_entropy(logits)
    return result

def masked_normalize(
        tensor:torch.Tensor,
        mask:torch.Tensor,
        normalized_constant:float,
        dim:Optional[int]=None,
)->torch.Tensor:
    """在指定的维度上求和并除以一个数归一化，只考虑mask==1的元素"""
    masked_tensor=tensor*mask.to(tensor.dtype)

    if dim is None:
        summed=masked_tensor.sum()
    else:
        summed=masked_tensor.sum(dim=dim)

    normalized=summed/normalized_constant

    return normalized

def sft_microbatch_train_step(
        policy_log_probs:torch.Tensor,
        response_mask:torch.Tensor,
        gradient_accumulation_steps:int,
        normalize_constant:float=1.,
)->tuple[torch.Tensor,dict[str,torch.Tensor]]:
    per_token_loss=-policy_log_probs #负对数似然
    batch_size=policy_log_probs.shape[0]
    loss=masked_normalize(
        tensor=per_token_loss,
        mask=response_mask,
        normalized_constant=normalize_constant,
        dim=None
    )
    #梯度累计，我们已经在example里面写过
    scaled_loss=loss/gradient_accumulation_steps/batch_size #test的细节问题，在除以累积步数后还得除以一个batch_size
    scaled_loss.backward()
    metadata={
        "unscaled_loss": loss.detach(),
        "loss":scaled_loss.detach(),
    }
    return scaled_loss,metadata

def log_generations(
        vllm_model,
        prompts: list[str],
        ground_truths: list[str],
        reward_fn: callable,
        sampling_params,
        step: int,
        policy_model=None,  #用于精确计算熵
        tokenizer=None      #配合policy_model使用
):
    """
    在训练循环中生成回复并记录各类统计指标。
    """
    print(f"\nStep{step}验证阶段")

    #使用vLLM生成回复
    outputs = vllm_model.generate(prompts, sampling_params, use_tqdm=False)

    format_rewards = []
    answer_rewards = []
    total_rewards = []
    response_lengths = []
    correct_lengths = []
    incorrect_lengths = []

    #遍历结果进行统计
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]

        #计算Reward
        reward_dict = reward_fn(generated_text, ground_truth)
        format_reward = reward_dict.get("format_reward", 0.0)
        answer_reward = reward_dict.get("answer_reward", 0.0)
        total_reward = reward_dict.get("reward", 0.0)

        format_rewards.append(format_reward)
        answer_rewards.append(answer_reward)
        total_rewards.append(total_reward)

        #统计长度
        length = len(output.outputs[0].token_ids)
        response_lengths.append(length)
        if answer_reward > 0:
            correct_lengths.append(length)
        else:
            incorrect_lengths.append(length)

        #打印前5个生成的例子
        if i < 5:
            print(f"\n[Example{i + 1}]")
            print(f"Prompt:{prompts[i][:50]}...")  #截断打印避免刷屏
            print(f"Ground Truth:{ground_truth}")
            print(f"Generation:\n{generated_text}")
            print(f"Rewards->Format:{format_reward}, Answer: {answer_reward}")

    #计算聚合指标
    metrics = {
        "eval/format_reward": np.mean(format_rewards),
        "eval/answer_reward": np.mean(answer_rewards),
        "eval/total_reward": np.mean(total_rewards),
        "eval/response_length": np.mean(response_lengths) if response_lengths else 0,
        "eval/correct_length": np.mean(correct_lengths) if correct_lengths else 0,
        "eval/incorrect_length": np.mean(incorrect_lengths) if incorrect_lengths else 0,
    }

    #计算平均Token熵
    #如果传了policy_model，我们可以取小批量数据算一下平均熵
    if policy_model is not None and tokenizer is not None:
        total_entropy_sum = 0.0
        total_response_tokens = 0

        #提取所有的生成文本
        all_outputs_text = [out.outputs[0].text for out in outputs]

        #按chunk_size分批处理
        chunk_size = 4

        for i in range(0, len(prompts), chunk_size):
            chunk_prompts = prompts[i: i + chunk_size]
            chunk_outputs = all_outputs_text[i: i + chunk_size]

            batch_data = tokenize_prompt_and_output(chunk_prompts, chunk_outputs, tokenizer)
            input_ids = batch_data["input_ids"].to(policy_model.device)
            response_mask = batch_data["response_mask"].to(policy_model.device)

            with torch.no_grad():
                logits = policy_model(input_ids).logits
                entropy = compute_entropy(logits)

                #累加当前chunk的熵和有效token数量
                total_entropy_sum += (entropy * response_mask).sum().item()
                total_response_tokens += response_mask.sum().item()

        #最终除以所有的有效token数量
        metrics["eval/token_entropy"] = (total_entropy_sum / total_response_tokens) if total_response_tokens > 0 else 0.0

    #打印汇总
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("==========================================\n")

    #如果配置了wandb，可以直接调用wandb.log(metrics, step=step)

    return metrics

def compute_group_normalized_rewards(
        reward_fn,
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float = 1e-6,
        normalize_by_std: bool = True,
):
    """
    计算组内归一化的优势
    """
    raw_rewards_list = []

    #遍历计算所有生成的Reward
    #rollout_responses的长度是rollout_batch_size(即questions_num * group_size)
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(resp, gt)
        #根据讲义，reward_fn返回包含"reward", "format_reward", "answer_reward" 的字典
        #我们使用总的"reward"来计算优势
        raw_rewards_list.append(reward_dict.get("reward", 0.0))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    #按组(group_size)重塑
    #将形状从(rollout_batch_size,)变成(num_questions, group_size)
    reshaped_rewards = raw_rewards.view(-1, group_size)

    #计算每组的均值
    group_means = reshaped_rewards.mean(dim=1, keepdim=True)

    #根据normalize_by_std决定是否除以标准差
    if normalize_by_std:
        #unbiased=False计算总体标准差，防止组内奖励全一样时std报错或失真
        group_stds = reshaped_rewards.std(dim=1, unbiased=True, keepdim=True)
        #归一化: (R-mean)/(std+eps)
        adv = (reshaped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        adv = reshaped_rewards - group_means

    #展平回1D张量，形状恢复为(rollout_batch_size,)
    advantages = adv.view(-1)

    #收集一些统计信息(Metadata)方便我们后续用wandb或print监控
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算朴素的策略梯度损失
    """
    #形状对齐(广播准备)
    #如果传进来的是1D张量(batch_size,)，我们需要把它变成(batch_size, 1)
    #这样才能和(batch_size, sequence_length)的log_probs相乘
    if raw_rewards_or_advantages.dim() == 1:
        adv = raw_rewards_or_advantages.unsqueeze(1)
    else:
        adv = raw_rewards_or_advantages

    #计算Loss: -A_t * log p
    #adv会自动在sequence_length维度上广播，相当于这个回答里的每个Token
    #都乘以了相同的优势值
    loss = -adv * policy_log_probs

    return loss


def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算GRPO-Clip Loss。
    """
    #广播advantages，使其与log_probs的sequence_length维度匹配
    if advantages.dim() == 1:
        adv = advantages.unsqueeze(1)
    else:
        adv = advantages

    #计算概率比率
    #p_new/p_old = exp(log_p_new - log_p_old)
    ratio = torch.exp(policy_log_probs - old_log_probs)

    #计算未裁剪项和裁剪项
    unclipped_term = ratio * adv
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_term = clipped_ratio * adv

    #取两者中较小的一个，并取负号(为了梯度下降)
    loss = -torch.min(unclipped_term, clipped_term)

    #计算并记录Metadata
    #讲义建议记录是否被clip：即clipped是否严格小于左侧unclipped
    is_clipped = (clipped_term < unclipped_term).float()

    metadata = {
        #我们直接把这个布尔张量传出去，外面可以用masked_mean算真正的 clip fraction
        "is_clipped": is_clipped
    }

    return loss, metadata


def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
) -> torch.Tensor:
    """
    计算带有掩码的张量均值
    """
    mask_float=mask.to(tensor.dtype)
    masked_sum=(tensor*mask_float).sum(dim=dim)
    masked_count=mask_float.sum(dim=dim)
    return masked_sum / masked_count


def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    根据给定的loss_type，分发计算对应的策略梯度损失，并返回损失和相关的metadata。
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "no_baseline requires raw_rewards"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "reinforce_with_baseline requires advantages"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}

    elif loss_type == "grpo_clip":
        assert advantages is not None, "grpo_clip requires advantages"
        assert old_log_probs is not None, "grpo_clip requires old_log_probs"
        assert cliprange is not None, "grpo_clip requires cliprange"
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata

def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    执行单次GRPO Micro-batch的前向计算与反向传播。
    """
    #获取per-tokenloss
    #形状为(batch_size, sequence_length)
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    #聚合成per-example loss
    #沿着序列长度方向(dim=1)对有效的生成token求平均
    #形状变成(batch_size,)
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)

    #对当前整个Batch求标量Loss
    loss = per_example_loss.mean()

    #根据梯度累加步数缩放 Loss
    scaled_loss = loss / gradient_accumulation_steps

    scaled_loss.backward()

    #把未缩放的loss存入metadata供后续logging监控
    metadata["loss"] = loss.detach()

    #讲义要求返回调整后的loss和metadata
    return scaled_loss, metadata