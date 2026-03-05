from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import sys
from fastwarc.warc import ArchiveIterator,WarcRecordType


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    将HTML字节流转换为纯文本，具备自动编码检测功能。
    """
    try:
        #使用最常见的UTF-8 解码
        html_string = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        #如果失败，检测实际编码
        encoding = detect_encoding(html_bytes)
        #使用检测到的编码进行解码，若仍失败则忽略错误字符
        html_string = html_bytes.decode(encoding, errors="ignore")

    #执行文本提取
    return extract_plain_text(html_string)

#以下是对读取数据的测试

if __name__ == "__main__":
    warc_path="../data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    try:
        for record in ArchiveIterator(open(warc_path, "rb"),record_types=WarcRecordType.response): #利用ArchiveIterator迭代遍历WARC，record_types指定返回的内容
            url=record.headers.get("WARC-Target-URI") #获取url
            html_bytes = record.reader.read()         #读取原始byte流
            extracted_text = extract_text_from_html_bytes(html_bytes)
            print(f"url:{url}")
            print(extracted_text[:500]+"后面暂时略去")
            input("按回车查看下一个记录")
    except FileNotFoundError:
        print(f"错误：找不到文件 {warc_path}，请检查路径是否正确。")
    except Exception as e:
        print(f"运行出错: {e}")