"""
此脚本用于采集数据
"""
import os
import random
import gzip
import subprocess

WIKI_URLS_LINK = "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz"
LOCAL_GZ_PATH = "data/wiki/enwiki-20240420-extracted_urls.txt.gz"
SUBSAMPLED_URLS_PATH = "data/wiki/subsampled_positive_urls.txt"

def prepare_directories():
    os.makedirs("data/wiki", exist_ok=True)

def download_wiki_urls():
    if not os.path.exists(LOCAL_GZ_PATH):
        print("正在下载维基百科URL列表(可能需要几分钟)...")
        subprocess.run(["wget", WIKI_URLS_LINK, "-O", LOCAL_GZ_PATH], check=True)
    else:
        print("维基百科 URL 列表已存在，跳过下载。")

def subsample_urls(sample_size=5000):
    print(f"正在从4350万个URL中随机抽取{sample_size}个...")
    #不用把整个几十MB的文件全读进内存
    sampled = []
    with gzip.open(LOCAL_GZ_PATH, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            url = line.strip()
            if not url: continue

            if i < sample_size:
                sampled.append(url)
            else:
                r = random.randint(0, i)
                if r < sample_size:
                    sampled[r] = url

    with open(SUBSAMPLED_URLS_PATH, 'w', encoding='utf-8') as out_f:
        for url in sampled:
            out_f.write(url + '\n')
    print(f"已将抽样的 URL 保存到 {SUBSAMPLED_URLS_PATH}")

def fetch_warc_from_urls():
    print("正在使用wget抓取网页并生成WARC文件...")
    #切换到data/wiki目录执行wget，这样生成的warc文件就存在那里
    os.chdir("data/wiki")
    #wget会给--warc-file的名字加上.warc.gz后缀
    cmd = [
        "wget",
        "--timeout=5",
        "--tries=1",  # 失败了就不重试，加快速度
        "-i", "subsampled_positive_urls.txt",
        "--warc-file=subsampled_positive_urls",
        "-O", "/dev/null"
    ]

    #我们忽略报错
    subprocess.run(cmd, stderr=subprocess.DEVNULL)
    print("\n抓取完成！生成的 WARC 文件在 data/wiki/ 目录下。")


if __name__ == "__main__":
    prepare_directories()
    download_wiki_urls()
    subsample_urls(sample_size=2000)  # 先抽 2000 个试试水，避免等太久
    fetch_warc_from_urls()