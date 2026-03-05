import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def create_validation_bin():
    #放在data/paloma/目录下
    out_dir = "data/paloma"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tokenized_paloma_c4_100_domains_validation.bin")

    if os.path.exists(out_path):
        print(f"文件已存在: {out_path}")
        return

    print("加载GPT-2分词器")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("获取验证集数据")
    #尝试加载Paloma的C4_100_domains,huggingface可以断点续传
    print("尝试从allenai/paloma下载C4 100 domains...")
    dataset = load_dataset("allenai/paloma", "c4_100_domains", split="val")

    print("开始分词并转换为uint16")
    all_tokens = []
    for idx, item in enumerate(dataset):
        text = item["text"]
        #将纯文本转为token IDs
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

        #每篇文档末尾需要加EOS Token
        #GPT-2的EOS TokenID是50256
        all_tokens.append(tokenizer.eos_token_id)

        if idx % 1000 == 0 and idx > 0:
            print(f"已处理 {idx} 篇文档，当前累计 Token 数: {len(all_tokens)}...")

    print(f"分词完成！总计 {len(all_tokens)} 个 Tokens。")

    print("保存为二进制.bin文件")
    #保存为uint16，因为 GPT-2 的词表大小是 50257，刚好能塞进16无符号整数 (最大 65535)

    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(out_path)
    print(f"验证集已保存至: {out_path}")


if __name__ == "__main__":
    create_validation_bin()