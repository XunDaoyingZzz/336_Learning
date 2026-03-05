import os
import glob
import random
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extract_text import extract_text_from_html_bytes
import fasttext

#定义路径
WIKI_WARC_PATTERN = "data/wiki/*.warc.gz"

CC_WARC_PATH = "data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
TRAIN_FILE = "data/classifiers/quality_train.txt"
MODEL_SAVE_PATH = "data/classifiers/my_quality_classifier.bin"


def prepare_training_data(target_samples=2000):
    print("===开始准备训练数据===")
    dataset = []

    #提取自己抓取的正样本
    print("正在提取正样本...")
    wiki_files = glob.glob(WIKI_WARC_PATTERN)
    pos_count = 0
    for wf in wiki_files:
        if pos_count >= target_samples: break
        try:
            for record in ArchiveIterator(open(wf, "rb"), record_types=WarcRecordType.response):
                if pos_count >= target_samples: break
                try:
                    text = extract_text_from_html_bytes(record.reader.read())
                    if text and text.strip():
                        #fastText要求单行文本，替换掉换行符
                        clean_text = text.replace("\n", " ").replace("\r", " ")
                        dataset.append(f"__label__hq {clean_text}")
                        pos_count += 1
                except Exception:
                    continue
        except Exception:
            continue

    print(f"成功提取了{pos_count}个正样本。")

    #提取负样本 (低质量/普通)
    #保持正负样本数量平衡1:1，所以负样本数量也设为pos_count
    print("正在提取负样本...")
    neg_count = 0
    try:
        for record in ArchiveIterator(open(CC_WARC_PATH, "rb"), record_types=WarcRecordType.response):
            if neg_count >= pos_count:
                break
            try:
                text = extract_text_from_html_bytes(record.reader.read())
                if text and text.strip():
                    clean_text = text.replace("\n", " ").replace("\r", " ")
                    dataset.append(f"__label__lq {clean_text}")
                    neg_count += 1
            except Exception:
                continue
    except Exception as e:
        print(f"读取 CC WARC 失败: {e}")

    print(f"成功提取了{neg_count}个负样本。")

    #打乱并保存数据
    print("正在打乱并写入训练集...")
    random.shuffle(dataset)

    os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for line in dataset:
            f.write(line + "\n")

    print(f"训练数据已保存到 {TRAIN_FILE}，总计 {len(dataset)} 条记录。")


def train_classifier():
    print("\n===开始训练fastText分类器===")
    #训练模型，wordNgrams=2表示同时考虑双词组合
    model = fasttext.train_supervised(input=TRAIN_FILE, epoch=10, wordNgrams=2)
    model.save_model(MODEL_SAVE_PATH)
    print(f"模型训练完成！已保存至{MODEL_SAVE_PATH}")

    #在训练集上快速自测一下
    result = model.test(TRAIN_FILE)
    print(f"训练集指标自测 -> 样本数: {result[0]}, 准确率(Precision): {result[1]:.4f}, 召回率(Recall): {result[2]:.4f}")

if __name__ == "__main__":
    prepare_training_data(target_samples=2000)
    train_classifier()