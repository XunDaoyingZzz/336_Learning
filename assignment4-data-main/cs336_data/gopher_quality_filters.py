import random
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extract_text import extract_text_from_html_bytes

warc_path="data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

def gopher_quality_filter(text: str) -> bool:
    """
    基于一定规则过滤低质量文本。
    全部通过返回True，否则返回False。
    """
    if not text or not text.strip():
        return False

    words = text.split()
    num_words = len(words)

    #词数过滤：包含少于 50 或多于 100,000 个词
    if num_words < 50 or num_words > 100000:
        return False

    #平均词长过滤：平均词长在 3 到 10 个字符之外
    total_chars = sum(len(word) for word in words)
    mean_word_length = total_chars / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    #包含至少一个字母的词占比过滤：少于80%
    alpha_words_count = sum(1 for word in words if any(char.isalpha() for char in word))
    if (alpha_words_count / num_words) < 0.8:
        return False

    #省略号结尾行占比过滤：超30%的行以"..."结尾
    lines = text.splitlines()
    num_lines = len(lines)
    if num_lines > 0:
        # 注意要用 strip() 去除行尾可能存在的不可见空白字符
        ellipsis_lines_count = sum(1 for line in lines if line.strip().endswith("..."))
        if (ellipsis_lines_count / num_lines) > 0.3:
            return False

    return True


def inspect_gopher_rules():
    iterator = ArchiveIterator(open(warc_path, "rb"), record_types=WarcRecordType.response)

    sampled_texts = []
    print("正在抽取 20 个样本，请稍候...")

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

    passed_count = 0
    for i, text in enumerate(sampled_texts):
        is_high_quality = gopher_quality_filter(text)
        if is_high_quality:
            passed_count += 1

        # 截取前后两端，方便看出文档是不是太短，或者是不是满屏省略号
        preview = text[:200].strip().replace('\n', ' ')

        print(f"--- 样本 {i + 1} ---")
        print(f"质量判断: {'通过 (高质量)' if is_high_quality else '不通过(低质量)'}")
        print(f"总词数: {len(text.split())}")
        print(f"文本截取: {preview}...\n")

    print(f"\n20个样本中，通过质量规则过滤的比例: {(passed_count / 20) * 100}%")

if __name__ == "__main__":
    inspect_gopher_rules()