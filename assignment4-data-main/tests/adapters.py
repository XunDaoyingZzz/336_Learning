from __future__ import annotations

import os
from typing import Any
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language
from cs336_data.mask_pii import mask_ips, mask_phone_numbers, mask_emails
from cs336_data.harmful_content import classify_nsfw,classify_toxic_speech
from cs336_data.gopher_quality_filters import gopher_quality_filter
from cs336_data.quality_classifier import classify_quality
from cs336_data.deduplication import exact_line_deduplication
from cs336_data.minhash_deduplication import minhash_deduplication

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)
    #raise NotImplementedError


def run_identify_language(text: str) -> tuple[Any, float]:
    #raise NotImplementedError
    return identify_language(text)

def run_mask_emails(text: str) -> tuple[str, int]:
    #raise NotImplementedError
    return mask_emails(text)

def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    #raise NotImplementedError
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    #raise NotImplementedError
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    #raise NotImplementedError
    return classify_nsfw(text)

def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    #raise NotImplementedError
    return classify_toxic_speech(text)

def run_classify_quality(text: str) -> tuple[Any, float]:
    return classify_quality(text)
    """
    讲一下这里的流程，需要先完成collect_wiki_urls.py的正样本采集（得挂梯子，而且最好是直接去网站下载而不要脚本下载）；然后跑通train_quality_classifier来保存模型，最后完成我们quality_classifier的测试
    """
    #raise NotImplementedError
    #return gopher_quality_filter(text)


def run_gopher_quality_filter(text: str) -> bool:
    #raise NotImplementedError
    return gopher_quality_filter(text)

def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return exact_line_deduplication(input_files, output_directory)
    #raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return minhash_deduplication(input_files, output_directory)
    #raise NotImplementedError
