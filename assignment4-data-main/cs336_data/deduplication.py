import os
import json
import hashlib
from pathlib import Path


def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    """
    精确行去重
    如果在整个数据集中某行出现了不止一次，则删除它的所有出现记录。
    """
    seen_once = set()
    seen_multiple = set()

    #扫描全集看哪些行出现了不止一次
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                #兼容JSONL和纯文本
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                except json.JSONDecodeError:
                    text = line.strip('\n')

                for text_line in text.split('\n'):
                    #不对纯空白行进行去重统计
                    if not text_line.strip():
                        continue

                    #用MD5哈希节省内存
                    h = hashlib.md5(text_line.encode('utf-8')).digest()
                    if h in seen_once:
                        seen_multiple.add(h)   #发现重复，放入黑名单
                    else:
                        seen_once.add(h)

    #重新遍历数据，根据黑名单进行过滤写入
    os.makedirs(output_directory, exist_ok=True)

    for input_file in input_files:
        input_path = Path(input_file)
        output_path = Path(output_directory) / input_path.name

        with open(input_path, 'r', encoding='utf-8') as f_in, \
                open(output_path, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                is_json = True
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                except json.JSONDecodeError:
                    is_json = False
                    text = line.strip('\n')

                kept_lines = []
                for text_line in text.split('\n'):
                    if not text_line.strip():
                        # 保留原有的空白行
                        kept_lines.append(text_line)
                    else:
                        h = hashlib.md5(text_line.encode('utf-8')).digest()
                        #只有不在黑名单里的行才保留下来
                        if h not in seen_multiple:
                            kept_lines.append(text_line)

                deduped_text = '\n'.join(kept_lines)

                if is_json:
                    doc["text"] = deduped_text
                    f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                else:
                    #纯文本模式：如果有内容留下，才写入。
                    #若一行因重复被删掉，这里就不会产生多余的换行。
                    if kept_lines:
                        f_out.write(deduped_text + '\n')