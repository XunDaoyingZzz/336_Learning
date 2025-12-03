from os.path import split
from typing import Iterable, Iterator, List, Dict, Tuple  # 导入类型提示，提高可读性
import os
import regex as re  # 导入regex库，用于复杂的正则表达式匹配
from collections import defaultdict, Counter  # 导入默认的字典和计数器，作为hash表的简单实现
import json  # JSON用于序列化和逆转
import time

# from anyio import value
from tqdm.contrib.concurrent import process_map

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# '(?:[sdmt]|ll|ve|re):匹配英语缩写's, 't, 'm, 'd, 'll, 've, 're
# | ?\p{L}+:匹配可选空格后跟一个或多个Unicode字母
# | ?\p{N}+:匹配可选空格后跟一个或多个Unicode数字
# | ?[^\s\p{L}\p{N}]+:匹配可选空格后跟一个或多个非空格、非字母、非数字的字符（如标点符号）
# |\s+(?!\S)：匹配后面不跟非空格字符的空格序列(如行末的空格)
# |\s+: 匹配其他空格序列

GPT2_RE = re.compile(GPT2_SPLIT_PATTERN)


def iter_pre_tokenize(text: str):
    """按正则逐个生成字节串，返回一个迭代器，每次产出字节串"""
    for m in GPT2_RE.finditer(text):
        yield m.group(0).encode("utf-8")


class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [x.encode("utf-8") for x in self.special_tokens]
        self.merges: list[tuple[bytes, bytes]] = []
        self.string_to_int: Dict[bytes, int] = {}
        self.int_to_string: Dict[int, bytes] = {}
        self.merges_rank: Dict[Tuple[bytes, bytes], int] = {}

        self.merges_set = set(self.merges)

        for i, token in enumerate(self.special_tokens_bytes):
            self.string_to_int[token] = i
            self.int_to_string[i] = token

        offset = len(self.special_tokens_bytes)

        for i in range(256):
            self.string_to_int[bytes([i])] = i + offset
            self.int_to_string[i + offset] = bytes([i])

        self.vocab = self.int_to_string.copy()

        self.pair2new = {(p1, p2): self.string_to_int[p1 + p2] for (p1, p2) in self.merges}

    def count_word(self, text: str):
        word_count = defaultdict(int)
        for m in re.finditer(GPT2_SPLIT_PATTERN, text):
            word = m.group(0)
            word_count[self.word2bytes(word)] += 1
        return word_count

    def word2bytes(self, word: str):
        a = list(word.encode('utf-8'))  # 先对传入的word:str进行utf的转换
        return tuple(bytes([i]) for i in a)

    def count_pair(self, word_cnt_dict: defaultdict[tuple,int]):
        pair_cnt = defaultdict(int)
        for word_bytes, cnt in word_cnt_dict.items():
            if len(word_bytes) < 2:
                continue
            for pair in zip(word_bytes[:-1], word_bytes[1:]):
                pair_cnt[pair] += cnt  # 就是说当前对所在的词条出现了多少次我们就统计多少次
        return pair_cnt

    def get_max_pair(self, pair_cnt: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes]:
        return max(pair_cnt.items(), key=lambda x: (x[1], x[0]))[0]  # 先按值，值相同就看键，键就直接比较bytes值，最后返回的是对应的键tuple

    def apply_merge(self, word_bytes: Tuple[bytes, bytes, ...], merge: Tuple[bytes, bytes]):
        merged_word = merge[0] + merge[1]
        i = 0
        new_word_bytes = []
        while i < len(word_bytes):
            if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i + 1] == merge[1]:
                new_word_bytes.append(merged_word)
                i += 2
            else:
                new_word_bytes.append(word_bytes[i])
                i += 1
        return tuple(new_word_bytes)

    def update_counts(self, word_count: Dict, pair_cnt: Dict, merge_pair: Tuple[bytes, bytes]):  # 传入的是
        new_word_count = defaultdict(int)
        new_pair_count = defaultdict(int, pair_cnt)

        for word_bytes, count in word_count.items():
            if len(word_bytes) < 2:  # 如果单词块长度小于2 则直接跳过
                continue

            old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

            if merge_pair not in old_pairs:  # 如果当前单词不包含合并的对，我们保留原单词，加入到new_word_count里面，并跳过当次循环
                new_word_count[word_bytes] += count
                continue

            new_word = self.apply_merge(word_bytes, merge_pair)  # 应用合并，在当前字节串下合并可合并的目标，使得new_word是含合并项的新字节串
            new_word_count[new_word] += count  # 新字节串的计数其实就是原字节串的计数，当然可能在当前步骤之前已经有其它的合并到了这一步，所以加上原来的计数即可

            # 减少原字节对的计数，因为合并之后这些对不再存在，注意这里我们的old_pairs一定是包含合并的对的
            for pair in old_pairs:  # 遍历原来字节串的对
                new_pair_count[pair] -= count  # 新的对计数器之中的pair需要自减count次
                if new_pair_count[pair] <= 0:  # 小于零了直接剔除计数，不小于零是因为我们会在循环节的后面重新添加对元
                    del new_pair_count[pair]

            new_pairs = list(zip(new_word[:-1], new_word[1:]))  # 新字节串的所有对组成的列表
            for p in new_pairs:  # 遍历这个列表
                new_pair_count[p] += count  # 对新对计数
        return new_word_count, new_pair_count

    def train(self, path: str | os.PathLike):
        assert self.vocab_size >= len(self.string_to_int)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if self.special_tokens:  # 非空特殊字符
            special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens)})"  # 构建特殊分词的正则表达式escape使得s可由join读取
            text_parts = re.split(special_pattern, text)  # 在text里面按special_tokens进行分割
            text_parts = [part for part in text_parts if part and part not in self.special_tokens]
        else:
            text_parts = [text]

        word_dicts = process_map(self.count_word, text_parts, chunksize=1000)

        merged_dict_word_count = defaultdict(int)  # 记录了各bytes词的出现次数
        for d in word_dicts:
            for k, v in d.items():
                merged_dict_word_count[k] += v

        pair_cnt = self.count_pair(merged_dict_word_count)

        num_merges_needed = self.vocab_size - len(self.string_to_int)  # 合并次数

        for i in range(num_merges_needed):
            if not pair_cnt:
                break

            max_pair = self.get_max_pair(pair_cnt)

            new_token = max_pair[0] + max_pair[1]
            new_token_id = len(self.string_to_int)

            self.string_to_int[new_token] = new_token_id
            self.int_to_string[new_token_id] = new_token

            self.merges.append(max_pair)

            merged_dict_word_count, pair_cnt = self.update_counts(merged_dict_word_count, pair_cnt, max_pair)

            if (i + 1) % 1000 == 0:
                print(f"合并训练{i + 1}/{num_merges_needed}次")
        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = self.int_to_string.copy()
        self.pair2new = {(p1, p2): self.string_to_int[p1 + p2] for (p1, p2) in self.merges}
        self.merges_set = set(self.merges)  # 更新merges_set

        print("BPE训练完成!")

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            text_to_process = [text]
        else:
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)  # 进行降序

            pattern = "|".join(re.escape(token) for token in special_tokens)
            pattern = f"({pattern})"
            pattern = re.compile(pattern)

            chunks = pattern.split(text)

            text_to_process = [c for c in chunks if c]

        tokens = []
        for chunk in text_to_process:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.string_to_int[chunk.encode("utf-8")])
            else:
                tokens.extend(self.encode_merged(chunk))
        return tokens

    def apply_merges(self, word_bytes):  # 注意是apply_merges不是merge
        word_bytes = list(word_bytes)

        while True:
            min_rank = float("inf")
            best_pair_pos_idx = -1
            merged = None

            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                if pair in self.merges_rank:
                    rank = self.merges_rank[pair]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair_pos_idx = i
                        merged = pair[0] + pair[1]

            if best_pair_pos_idx == -1:
                break

            word_bytes = (
                    word_bytes[:best_pair_pos_idx]
                    + [merged]
                    + word_bytes[best_pair_pos_idx + 2:]
            )
        return tuple(word_bytes)

    def encode_merged(self, text):
        if not text:  # 处理空字符串
            return []

        word_list = GPT2_RE.findall(text)

        tokens = []
        for word in word_list:
            word_bytes = self.word2bytes(word)
            merged_word_bytes = self.apply_merges(word_bytes)
            tokens.extend(self.string_to_int[i] for i in merged_word_bytes)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.int_to_string[t] for t in ids).decode('utf-8', errors='replace')

    def save(self, filepath: str):
        save_data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": [
                {
                    "pair": [pair[0].decode('latin1'), pair[1].decode('latin1')],
                    "merged": (pair[0] + pair[1]).decode('latin1'),
                }
                for pair in self.merges
            ],
            "string_to_int": {
                key.decode("latin1"): value
                for key, value in self.string_to_int.items()
            },
            "int_to_string": {
                key: value.decode("latin1")
                for key, value in self.int_to_string.items()
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"分词器已保存到{filepath}")

    @classmethod
    def from_files(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)

        tokenizer = cls(
            vocab_size=save_data["vocab_size"],
            special_tokens=save_data["special_tokens"]
        )
        tokenizer.merges = [
            (merge_info["pair"][0].encode('latin1'), merge_info["pair"][1].encode('latin1'))
            for merge_info in save_data["merges"]
        ]
        tokenizer.string_to_int = {
            key.encode('latin1'): value
            for key, value in save_data["string_to_int"].items()
        }
        tokenizer.int_to_string = {
            int(key): value.encode('latin1')
            for key, value in save_data['int_to_string'].items()
        }
        print("加载训练好的分词器成功！")
        return tokenizer



