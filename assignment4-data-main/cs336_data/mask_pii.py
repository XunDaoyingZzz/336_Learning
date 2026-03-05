import re
from typing import Tuple

import random
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extract_text import extract_text_from_html_bytes

warc_path = "data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

def mask_emails(text: str) -> Tuple[str, int]:
    """
    屏蔽电子邮件地址。
    """
    # 匹配常见的邮箱格式
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    #[a-zA-Z0-9_.+-]+：匹配@之前的用户名部分。方括号里列出了允许的字符（大小写字母、数字，以及 _、.、+、-），末尾的+表示这些字符至少要出现一次。
    #[a-zA-Z0-9-] +：匹配域名（如gmail、yahoo）。同样是字母、数字和连字符，至少出现一次。
    #\.转义表示. 然后最后的[a-zA-Z0-9-.]+：匹配 顶级域名（如 com、org），所以里面包含了字母、数字、连字符和点。
    #re.subn直接返回(new_string, num_replacements)
    return re.subn(email_pattern, "|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str) -> Tuple[str, int]:
    """
    屏蔽电话号码。重点捕获常见的美国电话格式。
    例如: (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890
    """
    #这个正则兼容了美国常见的区号、分隔符(空格、点、横杠)以及可选的+1前缀
    phone_pattern = r'(?<!\d)\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)'
    #(?<!\d):确保电话号码前面没有紧挨着其他数字
    #\+?1?:匹配可选的+1或者1
    #[-.\s]?:可选的分隔符
    #\(?\d{3}\)?:3位数字，两边可能有括号
    #[-.\s]?:可选的分隔符
    #\d{3}:3位数字
    #[-.\s]?:可选的分隔符
    #\d{4}:4位数字
    return re.subn(phone_pattern, " |||PHONE_NUMBER|||", text)


def mask_ips(text: str) -> Tuple[str, int]:
    """
    屏蔽 IPv4 地址。
    4 个 0-255 之间的数字，用点分隔。
    """
    # 匹配 0-255 的数字段
    octet = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    ip_pattern = rf'\b{octet}\.{octet}\.{octet}\.{octet}\b'

    return re.subn(ip_pattern, "|||IP_ADDRESS|||", text)


def inspect_pii_masking():
    iterator = ArchiveIterator(open(warc_path, "rb"), record_types=WarcRecordType.response)

    samples_found = 0
    print("正在寻找触发替换的文本...")

    for record in iterator:
        if random.random() < 0.05:  # 稍微提高点概率
            try:
                text = extract_text_from_html_bytes(record.reader.read())

                #依次应用三个过滤器
                masked_email, c1 = mask_emails(text)
                masked_phone, c2 = mask_phone_numbers(masked_email)
                final_text, c3 = mask_ips(masked_phone)

                total_replacements = c1 + c2 + c3
                if total_replacements > 0:
                    samples_found += 1
                    print(f"---找到替换样本{samples_found}(替换了{total_replacements}处)---")
                    #只打印包含替换符号的附近文本
                    import re
                    matches = re.finditer(r'\|\|\|(EMAIL_ADDRESS|PHONE_NUMBER|IP_ADDRESS)\|\|\|', final_text)
                    for m in matches:
                        start = max(0, m.start() - 50)
                        end = min(len(final_text), m.end() + 50)
                        print(f"上下文: ...{final_text[start:end].replace('\n', ' ')}...")
                    print("\n")

                if samples_found >= 5:
                    break
            except Exception:
                pass
#对问题5的测试
if __name__ == "__main__":
    inspect_pii_masking()