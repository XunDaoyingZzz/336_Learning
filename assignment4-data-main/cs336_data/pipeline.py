import os
import glob
import json
import time
from pathlib import Path
import concurrent.futures
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filters import gopher_quality_filter
from cs336_data.quality_classifier import classify_quality
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.deduplication import exact_line_deduplication
from cs336_data.minhash_deduplication import minhash_deduplication

#配置路径
INPUT_WARC_PATTERN = "data/CC/*.warc.gz"
PHASE1_OUT_DIR = "data/pipeline_out/01_filtered"
PHASE2_OUT_DIR = "data/pipeline_out/02_exact_dedup"
PHASE3_OUT_DIR = "data/pipeline_out/03_final_minhash"

#过滤阈值配置
QUALITY_THRESHOLD = 0.65  #质量分类器阈值(如果是 wiki/cc 二分类，置信度大于 0.65 视为 wiki)
TOXIC_THRESHOLD = 0.50    #毒性分类器阈值


def process_single_warc(warc_path: str) -> dict:
    """
    处理单个WARC文件的Worker函数。
    """
    input_file = Path(warc_path)
    output_file = Path(PHASE1_OUT_DIR) / f"{input_file.name}.jsonl"

    stats = {
        "total": 0, "extract_fail": 0, "rule_drop": 0,
        "quality_drop": 0, "toxic_drop": 0, "kept": 0
    }

    #如果已经处理过，直接跳过,支持断点续传
    if output_file.exists():
        print(f"跳过已处理的文件: {input_file.name}")
        return stats

    with open(output_file, 'w', encoding='utf-8') as f_out:
        try:
            for record in ArchiveIterator(open(warc_path, "rb"), record_types=WarcRecordType.response):
                stats["total"] += 1

                #提取文本
                try:
                    text = extract_text_from_html_bytes(record.reader.read())
                    if not text or not text.strip():
                        stats["extract_fail"] += 1
                        continue
                except Exception:
                    stats["extract_fail"] += 1
                    continue

                #启发式质量规则
                if not gopher_quality_filter(text):
                    stats["rule_drop"] += 1
                    continue

                #截取一段文本用于分类器预测
                preview_text = text[:1500]

                #质量分类器 (过滤掉 CC，只保留 Wiki 风格的)
                q_label, q_score = classify_quality(preview_text)
                if q_label != "wiki" or q_score < QUALITY_THRESHOLD:
                    stats["quality_drop"] += 1
                    continue

                #毒性 & NSFW 过滤
                nsfw_label, _ = classify_nsfw(preview_text)
                toxic_label, _ = classify_toxic_speech(preview_text)
                if nsfw_label == "nsfw" or toxic_label == "toxic":
                    stats["toxic_drop"] += 1
                    continue

                #通过了上述的识别，存入 JSONL
                doc = {"text": text, "source": input_file.name, "id": stats["total"]}
                f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                stats["kept"] += 1

        except Exception as e:
            print(f"读取 {input_file.name} 发生错误: {e}")

    return stats


def run_phase1_filtering(max_workers=16):
    """多进程并发过滤"""
    print(f"\n多进程并发过滤(Workers: {max_workers})...")#按照自己的cpu而定
    os.makedirs(PHASE1_OUT_DIR, exist_ok=True)

    warc_files = glob.glob(INPUT_WARC_PATTERN)
    if not warc_files:
        print("未找到任何WARC文件，检查INPUT_WARC_PATTERN路径！")
        return []

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        #提交所有任务
        futures = {executor.submit(process_single_warc, wf): wf for wf in warc_files}

        #收集结果
        for future in concurrent.futures.as_completed(futures):
            wf = futures[future]
            try:
                stats = future.result()
                print(
                    f"完成{Path(wf).name}->保留:{stats['kept']}/{stats['total']}(耗时:{time.time() - start_time:.1f}s)")
            except Exception as e:
                print(f"处理{Path(wf).name}失败:{e}")

    print(f"收集数据完成，总耗时: {time.time() - start_time:.2f}秒")
    return glob.glob(f"{PHASE1_OUT_DIR}/*.jsonl")


def run_phase2_exact_dedup(input_files):
    """精确行去重"""
    print("\n启动精确行去重...")
    start_time = time.time()
    exact_line_deduplication(input_files, PHASE2_OUT_DIR)
    print(f"完成！总耗时:{time.time()-start_time:.2f}秒")
    return glob.glob(f"{PHASE2_OUT_DIR}/*.jsonl")


def run_phase3_minhash_dedup(input_files):
    """MinHash模糊去重"""
    print("\n启动MinHash模糊去重...")
    start_time = time.time()
    #使用工业级配置
    minhash_deduplication(
        input_files=input_files,
        output_directory=PHASE3_OUT_DIR,
        num_hashes=100,
        num_bands=20,  #100个哈希分20块，每块5个
        ngrams=5,
        jaccard_threshold=0.8
    )
    print(f"完成！总耗时:{time.time() - start_time:.2f}秒")
    return glob.glob(f"{PHASE3_OUT_DIR}/*.jsonl")


if __name__ == "__main__":
    print("数据清洗Pipeline启动")

    WORKERS = 16

    #过滤
    filtered_files = run_phase1_filtering(max_workers=WORKERS)

    #精确行去重
    if filtered_files:
        exact_dedup_files = run_phase2_exact_dedup(filtered_files)

        #模糊去重
        if exact_dedup_files:
            final_files = run_phase3_minhash_dedup(exact_dedup_files)
            print("\n全部流水线执行完毕！")
            print(f"最终数据已保存在: {PHASE3_OUT_DIR}")
        else:
            print("精确行去重中没有产出任何文件。")
    else:
        print("过滤阶段没有产出任何文件。")