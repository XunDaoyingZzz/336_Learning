import fasttext
import os
from typing import Tuple
import re
import random
from fastwarc.warc import WarcRecordType,ArchiveIterator

from cs336_data.extract_text import extract_text_from_html_bytes

warc_path="data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

NSFW_MODEL_PATH ="data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"
TOXIC_MODEL_PATH ="data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"

#加载模型
try:
    nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
    toxic_model = fasttext.load_model(TOXIC_MODEL_PATH)
except Exception as e:
    print(f"加载模型失败，请检查路径。错误信息: {e}")
    nsfw_model = None
    toxic_model = None

def preprocess_text(text: str) -> str:
    """清理文本，转小写并分离标点符号，以适应fastText的按空格分词"""
    text = text.lower().replace("\n", " ").replace("\r", " ")
    #将常见的标点符号两边加上空格
    text = re.sub(r'([.!?,"()])', r' \1 ', text)
    return text


def classify_nsfw(text: str) -> Tuple[str, float]:
    """
    检测 NSFW 内容。
    返回: (标签, 置信度分数)
    """
    if not text or not text.strip() or nsfw_model is None:
        return "unknown", 0.0

    text_clean = preprocess_text(text)
    predictions, probabilities = nsfw_model.predict(text_clean, k=1)

    #label = predictions[0].replace("__label__", "")
    #score = float(probabilities[0])
    label = predictions[0].replace("__label__", "")
    if label in ["toxic", "obscene", "severe_toxic", "nsfw"]:
        label = "nsfw"
    elif label in ["non-toxic", "non-obscene", "normal", "non-nsfw"]:
        label = "non-nsfw"

    score = float(probabilities[0])
    return label, score

    return label, score


def classify_toxic_speech(text: str) -> Tuple[str, float]:
    """
    检测有毒言论 (Toxic Speech)。
    返回: (标签, 置信度分数)
    """
    if not text or not text.strip() or toxic_model is None:
        return "unknown", 0.0

    text_clean = preprocess_text(text)
    predictions, probabilities = toxic_model.predict(text_clean, k=1)
    label = predictions[0].replace("__label__", "")
    if label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "nsfw"]:
        label = "toxic"
    elif label in ["non-toxic", "normal", "non-nsfw"]:
        label = "non-toxic"

    score = float(probabilities[0])
    return label, score

def inspect_harmful_content():
    iterator = ArchiveIterator(open(warc_path, "rb"), record_types=WarcRecordType.response)

    sampled_texts = []
    print("正在边遍历边读取 100 个随机样本，请稍候...")

    for record in iterator:
        #给个小概率随机抽取，避免全集中在文件开头
        if random.random() < 0.05:
            try:
                #必须立即 read() 避免 stale reader 报错
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                if text and text.strip():
                    sampled_texts.append(text)
            except Exception as e:
                pass

        if len(sampled_texts) >= 100:
            break

    harmful_count = 0
    for i, text in enumerate(sampled_texts):
        #截取前800个字符进行预测
        preview_text = text[:800]

        nsfw_label, nsfw_score = classify_nsfw(preview_text)
        toxic_label, toxic_score = classify_toxic_speech(preview_text)

        #判断是否有害（只要有一个被判定为有害即可）
        is_harmful = (nsfw_label == "nsfw") or (toxic_label == "toxic")
        if is_harmful:
            harmful_count += 1

        print(f"--- 样本 {i + 1} ---")
        print(f"NSFW 预测: {nsfw_label} (置信度: {nsfw_score:.4f})")
        print(f"Toxic 预测: {toxic_label} (置信度: {toxic_score:.4f})")
        print(f"文本截取: {preview_text[:150].strip().replace('\n', ' ')}...\n")

    harmful_ratio = (harmful_count / 100) * 100
    print(f"100个样本中，被分类器判定为有害的比例: {harmful_ratio}%")
if __name__ == "__main__":
    inspect_harmful_content()