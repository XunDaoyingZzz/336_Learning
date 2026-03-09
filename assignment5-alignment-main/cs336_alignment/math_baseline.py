import json
import os
from typing import List, Callable, Dict

from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        eval_sampling_params: SamplingParams,
        ground_truths: List[str],
        output_path: str
) -> None:
    """
    在给定prompt列表上评估语言模型，计算评价指标，并将结果序列化
    """
    print(f"开始使用vLLM 生成{len(prompts)}条回复...")
    #批量生成回复
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    format_reward_1_answer_1 = 0
    format_reward_1_answer_0 = 0
    format_reward_0_answer_0 = 0

    #遍历生成结果，计算Reward
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]

        #调用指定的reward_fn
        reward_dict = reward_fn(generated_text, ground_truth)

        format_reward = reward_dict.get("format_reward", 0.0)
        answer_reward = reward_dict.get("answer_reward", 0.0)

        #统计用于回答作业问题(b)的类别
        if format_reward == 1.0 and answer_reward == 1.0:
            format_reward_1_answer_1 += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            format_reward_1_answer_0 += 1
        elif format_reward == 0.0 and answer_reward == 0.0:
            format_reward_0_answer_0 += 1

        results.append({
            "prompt": prompts[i],
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "rewards": reward_dict
        })

    # 3. 序列化保存到磁盘
    with open(output_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"评估完成！结果已保存至:{output_path}")
    print(f"统计信息:")
    print(f"格式正确 且 答案正确 (Format 1, Answer 1):{format_reward_1_answer_1}")
    print(f"格式正确 但 答案错误 (Format 1, Answer 0):{format_reward_1_answer_0}")
    print(f"格式错误 且 答案错误 (Format 0, Answer 0):{format_reward_0_answer_0}")
    print(f"总体准确率 (Answer Reward = 1):{format_reward_1_answer_1 / len(prompts):.2%}")


def main():
    #路径配置
    model_path = "models/Qwen2.5-Math-1.5B"
    val_data_path = "data/gsm8k/test.jsonl"  #假设你将数据集放在了这里
    prompt_template_path = "cs336_alignment/prompts/r1_zero.prompt"
    output_path = "math_baseline_results.jsonl"
    #加载提示词模板
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    #加载验证集数据
    #根据实际jsonl的key来提取问题和答案。
    #这里假设key为"problem"和"solution"或"answer"
    questions = []
    ground_truths = []
    with open(val_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            #这里根据数据集调整
            questions.append(data.get("question", ""))  #去看数据集的结构

            raw_answer = data.get("answer", "")         #同上

            if "####" in raw_answer:                    #gsm8k数据集最终的答案在####后
                final_answer = raw_answer.split("####")[-1].strip()
            else:
                final_answer = raw_answer.strip()
            ground_truths.append(final_answer)

    #构造 prompts
    prompts = [prompt_template.replace("{question}", q) for q in questions]

    #根据讲义3.2要求配置SamplingParams
    sampling_params = SamplingParams(
        temperature=1.0,
    top_p = 1.0,
    max_tokens = 1024,
    stop = ["</answer>"],
    include_stop_str_in_output = True
    )

    #初始化 vLLM
    print("正在加载模型...")
    llm = LLM(model=model_path)

    #执行评估并保存
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        ground_truths=ground_truths,
        output_path=output_path
    )


if __name__ == "__main__":
    main()