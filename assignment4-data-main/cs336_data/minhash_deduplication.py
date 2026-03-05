import os
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

#config(测试里不用)
#NGRAM_SIZE = 5           #使用5-gram
#NUM_HASHES = 100         #签名长度（100个哈希函数）
#NUM_BANDS = 20           #分成20块
#ROWS_PER_BAND = 5        #每块5行(20*5 =100)
#JACCARD_THRESHOLD = 0.8  #相似度大于80%认为是重复

def get_ngrams(text: str, n: int) -> list:
    """将文本切分为N-gram集合"""
    words = text.split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def minhash_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike,num_hashes:int=100,num_bands:int=10,ngrams:int=5,jaccard_threshold:float=0.8):
    """模糊去重(MinHash+LSH)"""
    os.makedirs(output_directory, exist_ok=True)

    rows_per_band = num_hashes // num_bands
    # 生成100个固定的哈希函数参数(h(x) = (ax + b)%p)
    random.seed(42)
    P = 4294967311  # 一个大于2^32的质数
    A = [random.randint(1, P - 1) for _ in range(num_hashes)]
    B = [random.randint(0, P - 1) for _ in range(num_hashes)]

    def compute_signature(ngrams: list) -> list:
        """计算MinHash签名"""
        sig = [float('inf')] * num_hashes
        for ngram in ngrams:
            # 将ngram转换为一个32位的整数基准哈希
            x = int(hashlib.md5(ngram.encode('utf-8')).hexdigest()[:8], 16)
            for i in range(num_hashes):
                h = (A[i] * x + B[i]) % P
                if h < sig[i]:
                    sig[i] = h
        return sig

    #存储所有文档的签名，用于后续估算Jaccard
    #doc_id格式:"file_idx:line_idx"
    signatures = {}

    #LSH桶:buckets[band_idx][band_hash]=[doc_id1, doc_id2, ...]
    buckets = [defaultdict(list) for _ in range(num_bands)]

    #区分json，txt
    def get_documents(input_file, file_idx):
        path = Path(input_file)
        if path.name.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if not line.strip(): continue
                    try:
                        doc = json.loads(line)
                        text = doc.get("text", "")
                    except json.JSONDecodeError:
                        text = line.strip('\n')
                    yield f"{file_idx}:{line_idx}", text, line
        else:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                #整个文件作为一篇文档
                yield f"{file_idx}:0", content, content

    print("计算MinHash签名并分配LSH桶...")
    for file_idx, input_file in enumerate(input_files):
        for doc_id, text, _ in get_documents(input_file, file_idx):
            doc_ngrams = get_ngrams(text, ngrams)
            if not doc_ngrams: continue

            sig = compute_signature(doc_ngrams)
            signatures[doc_id] = sig

            for band_idx in range(num_bands):
                start = band_idx * rows_per_band
                end = start + rows_per_band
                band_tuple = tuple(sig[start:end])
                band_hash = hash(band_tuple)
                buckets[band_idx][band_hash].append(doc_id)

    print("查找碰撞并构建并查集...")
    #用来追踪哪些文档属于同一个重复组
    parent = {}

    def find(i):
        if parent.setdefault(i, i) != i: #setdefault(x,y)用于查找键x，如果没有x，那么加入x键并设定默认值y；我们找parent中的i键，如果没有则加一个{i:i}，并会返回i，这样整个函数会返回i；否则返回i对应的值（父亲），进而去找父亲的父亲
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            #始终保留id较小的文档（即先出现的文档）作为根节点
            #id 格式是 "file_idx:line_idx" 字符串，需要解析成数字比对
            idx_i = tuple(map(int, root_i.split(':')))
            idx_j = tuple(map(int, root_j.split(':')))
            if idx_i < idx_j:
                parent[root_j] = root_i
            else:
                parent[root_i] = root_j

    #遍历所有的桶，寻找相似候选对
    for band_idx in range(num_bands):
        for bucket in buckets[band_idx].values():
            if len(bucket) > 1:
                #桶内两两比对签名，估算真实Jaccard相似度
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        doc1 = bucket[i]
                        doc2 = bucket[j]
                        #如果已经在同一个集合里，跳过
                        if find(doc1) == find(doc2):
                            continue

                        sig1 = signatures[doc1]
                        sig2 = signatures[doc2]
                        #签名一致的数量占比即为估算的Jaccard相似度
                        match_count = sum(1 for a, b in zip(sig1, sig2) if a == b)
                        sim = match_count / num_hashes

                        if sim >= jaccard_threshold:
                            union(doc1, doc2)

    print("输出去重后的文件")
    for file_idx, input_file in enumerate(input_files):
        input_path = Path(input_file)
        output_path = Path(output_directory) / input_path.name

        docs_to_write = []
        for doc_id, text, raw_content in get_documents(input_file, file_idx):
            #如果文档太短没进signature，或者它是自己的父节点，才保留
            if doc_id not in signatures or find(doc_id) == doc_id:
                docs_to_write.append(raw_content)

        #只有当这篇文档/文件里有合法内容时，才创建文件
        if docs_to_write:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for content in docs_to_write:
                    f_out.write(content)