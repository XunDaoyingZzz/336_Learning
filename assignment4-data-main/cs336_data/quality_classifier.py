import fasttext
import os
from typing import Tuple

import random
from fastwarc.warc import ArchiveIterator,WarcRecordType
from cs336_data.extract_text import extract_text_from_html_bytes

warc_path="data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
#获取训练好的模型路径
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(current_dir, "..", "data", "classifiers", "my_quality_classifier.bin"))

#全局加载模型
try:
    quality_model = fasttext.load_model(MODEL_PATH)
except Exception as e:
    print(f"加载质量模型失败，错误信息: {e}")
    quality_model = None


def classify_quality(text: str) -> Tuple[str, float]:
    """
    检测文本质量。
    返回 (标签, 置信度分数)
    """
    if not text or not text.strip() or quality_model is None:
        return "unknown", 0.0

    #预处理：替换换行符，适应 fastText
    text_clean = text.replace("\n", " ").replace("\r", " ")

    #进行预测
    predictions, probabilities = quality_model.predict(text_clean, k=1)

    #提取标签并去掉 __label__ 前缀(剩下 "hq" 或 "lq")
    label = predictions[0].replace("__label__", "")

    #标签映射
    #迎合测试脚本
    if label == "hq":
        label = "wiki"
    elif label == "lq":
        label = "cc"

    score = float(probabilities[0])
    return label, score


def inspect_quality_classifier():
    iterator = ArchiveIterator(open(warc_path, "rb"), record_types=WarcRecordType.response)

    sampled_texts = []
    for record in iterator:
        if random.random() < 0.05:
            try:
                text = extract_text_from_html_bytes(record.reader.read())
                if text.strip():
                    sampled_texts.append(text)
            except Exception:
                pass

        if len(sampled_texts) >= 20:
            break

    for i, text in enumerate(sampled_texts):
        # 用训练好的分类器
        label, score = classify_quality(text[:1000])  #截取前1000字符
        preview = text[:150].strip().replace('\n', ' ')

        print(f"样本 {i + 1}")
        print(f"预测标签: {label} (置信度: {score:.4f})")
        print(f"文本截取: {preview}...\n")

#简答题的测试
if __name__ == "__main__":
    inspect_quality_classifier()