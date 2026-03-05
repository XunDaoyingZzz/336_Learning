import os
import glob
import json
import numpy as np
from transformers import AutoTokenizer

#配置路径
INPUT_JSONL_PATTERN = "data/pipeline_out/03_final_minhash/*.jsonl"
OUTPUT_BIN_PATH = "data/train.bin"


def tokenize_training_data():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    files = glob.glob(INPUT_JSONL_PATTERN)

    if not files:
        print("没找到JSONL文件，检查路径")
        return

    print(f"开始Tokenize训练集 (共{len(files)}个文件)")
    all_tokens = []
    doc_count = 0

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue

                try:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                except json.JSONDecodeError:
                    text = line.strip()

                if not text: continue

                #编码并加上文档结束符<|endoftext|>
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
                all_tokens.append(tokenizer.eos_token_id)

                doc_count += 1
                if doc_count % 5000 == 0:
                    print(f"已处理 {doc_count} 篇训练文档，当前 Tokens 数量: {len(all_tokens)}...")

    print(f"\n训练集Tokenize完成，总文档数: {doc_count}，总 Token 数: {len(all_tokens)}")

    print("保存为 train.bin")
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(OUTPUT_BIN_PATH)
    print(f"数据集已生成: {OUTPUT_BIN_PATH}")


if __name__ == "__main__":
    tokenize_training_data()