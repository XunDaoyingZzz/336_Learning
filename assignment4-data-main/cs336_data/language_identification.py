import fasttext
from typing import Tuple
from cs336_data.extract_text import extract_text_from_html_bytes
from fastwarc.warc import ArchiveIterator, WarcRecordType
import random

warc_path = "data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

#路径是 "data/classifiers/lid.176.bin"
MODEL_PATH = "data/classifiers/lid.176.bin"

#在全局加载模型，避免每次调用函数时重复加载，提高效率
try:
    lang_model = fasttext.load_model(MODEL_PATH)
except ValueError as e:
    print(f"无法加载 fasttext 模型，请检查路径: {MODEL_PATH}")
    lang_model = None


def identify_language(text: str) -> Tuple[str, float]:
    """
    识别给定字符串的主要语言。
    返回: (语言代码, 置信度分数)
    """
    if not text or not text.strip():
        #如果文本为空，返回unknown和0置信度
        return "unk", 0.0

    #fasttext预测时遇到换行符会截断，因此将其替换为空格
    text_clean = text.replace("\n", " ").replace("\r", " ")

    #k=1 表示只返回概率最高的一个预测结果
    predictions, probabilities = lang_model.predict(text_clean, k=1)

    #提取结果并去除'__label__'前缀
    lang_code = predictions[0].replace("__label__", "")
    score = float(probabilities[0])

    return lang_code, score


def inspect_language_id():
    iterator = ArchiveIterator(open(warc_path, "rb"), record_types=WarcRecordType.response)

    sampled_texts = []

    print("正在随机抽取并在内存中提取文本，请稍候...")
    #遍历读取
    for record in iterator:
        #比如我们给0.02的概率选中并读取这条记录
        if random.random() < 0.02:
            try:
                #必须在迭代器进入下一条记录前读取bytes
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                sampled_texts.append(text)
            except Exception as e:
                print(f"提取跳过一条错误记录: {e}")

        #凑满20个就停下
        if len(sampled_texts) >= 20:
            break

    english_count = 0
    for i, text in enumerate(sampled_texts):
        #只取前500个字符进行预测，提高速度
        preview_text = text[:500]
        lang_code, score = identify_language(preview_text)

        if lang_code == "en":
            english_count += 1

        print(f"--- 样本 {i + 1} ---")
        print(f"预测语言: {lang_code}, 置信度: {score:.4f}")
        print(f"文本截取: {preview_text[:150].strip().replace('\n', ' ')}...\n")

    print(f"20个样本中，分类器认为是英文的比例: {english_count / 20 * 100}%")
#对c问的尝试
if __name__ == "__main__":
    inspect_language_id()

"""
如下，样本2，8，9，12有很多混合的字符，导致识别不精
--- 样本 1 ---
预测语言: zh, 置信度: 0.9451
文本截取: 人妻久久久久久久久久久久久,国产午夜精品福利在线观看,色道久久综合亚洲精品蜜桃,国产欧美日韩另类精彩视频        •                       蘇州守真儀器設(shè)備有限公司                                 13382153105...

--- 样本 2 ---
预测语言: pl, 置信度: 0.2575
文本截取: WordPress Training Site    • Area 11 Schedules Committee   • Area 11 Schedules Committee Guidelines   • District 20   • Districts   • Gdzie Znaleźć Rz...

--- 样本 3 ---
预测语言: unk, 置信度: 0.0000
文本截取: ...

--- 样本 4 ---
预测语言: lv, 置信度: 0.9754
文本截取: Skip to main content Atjaunotne  Primary links    • Par mums   • Reliģija     • Dievs     • Kristietība     • Citas reliģijas     • Morāle   • Politik...

--- 样本 5 ---
预测语言: en, 置信度: 0.8024
文本截取: Boat Charter Valletta BoatCharterValletta.com Book Now  Select your language    • English (United Kingdom) EN   • Deutsch (Deutschland) DE   • French...

--- 样本 6 ---
预测语言: pl, 置信度: 0.9949
文本截取: Rozmiar czcionki Wysoki kontrast English Logo  Bądź na bieżąco  Zapisz się do naszego newslettera aby otrzymywać najciekawsze informacje i aktualności...

--- 样本 7 ---
预测语言: en, 置信度: 0.7880
文本截取: Home Shop shop now -peace love and [bump] -[bump]-tees -[bump]-tanks -[bump]-vees -[bump]-maroc -[bump]-yoga -[bump]-daddy -[bump]-dresses shopping ca...

--- 样本 8 ---
预测语言: fr, 置信度: 0.2414
文本截取: Skip navigation   • SMART LIBRARY   •   MENU   •    Login     • My DSpace     • Receive email       updates     • Edit Account details   • SMART LIBR...

--- 样本 9 ---
预测语言: en, 置信度: 0.1969
文本截取: %PDF-1.6 % 31 0 obj <> endobj 43 0 obj <>/Filter/FlateDecode/ID[<04D3C06FD1A7AD4F973F584DB00B4100>]/Index[31 28]/Info 30 0 R/Length 73/Prev 42366/Root...

--- 样本 10 ---
预测语言: en, 置信度: 0.6284
文本截取: Call nowAbout us   • About us   • Get in touch  Fingal Design  Meny Stäng    • About us   • Get in touch  Kategorier    • Accessories   • Children fur...

--- 样本 11 ---
预测语言: en, 置信度: 0.5511
文本截取: Skip to content Facebook Call Us Today! 732-943-0333|office@allstatesrestoration.com First Class Floor Cleaning Logo   • Home   • SERVICES     • Carp...

--- 样本 12 ---
预测语言: en, 置信度: 0.3332
文本截取: From: blakes7-d-request@lysator.liu.se Subject: blakes7-d Digest V00 #205 X-Loop: blakes7-d@lysator.liu.se X-Mailing-List: archive/volume00/205 Preced...

--- 样本 13 ---
预测语言: zh, 置信度: 0.9998
文本截取: • 网站首页   • 关于我们     • 公司简介     • 公司文化     • 公司宣传片     • 公司荣誉   • 产品中心     • 产品介绍     • 临床研究     • 产品资讯     • 下载中心   • 新闻中心     • 最新公告     • 公司新闻...

--- 样本 14 ---
预测语言: es, 置信度: 0.9422
文本截取: Saltar al contenido principal Gilitadas   • Inicio   • Blog   • DIY – Do It Yourself   • Modelado 3D   • Meteorología   • Contacto Gilitadas De todo...

--- 样本 15 ---
预测语言: zh, 置信度: 0.9885
文本截取: 1. 首頁   2. 討論&留言   3. 掉落查詢   4. 排行榜   5. 直播秀   6. 登入 ※請多多利用「搜尋主題」來尋找解答，若找不到也可以新發一篇方便做個紀錄※ ※搜尋可以針對標題關鍵字搜尋也可以輸入文章編號數字搜尋，目前共有 1 篇文章※  (No.7538) 佩琳  (討論...

--- 样本 16 ---
预测语言: zh, 置信度: 0.8927
文本截取: 黄色网无码在线国产强奸,午夜影视在线观看,天天曰天天射试看二分钟,无码av永久免费专区不卡,亚洲中文字幕无码一区日日添,男人扒开女人腿桶到爽免费,亚洲天天舔天天插超碰                                您好，泰州市恒達(dá)換熱設(shè)備制造有限公 司主營：冷凝器...

--- 样本 17 ---
预测语言: en, 置信度: 0.6419
文本截取: MLPS Maidstone Landscape & Property Services Ltd :  Just One Call for all your landscaping & property services work - 07786 070107    • Home   • Lands...

--- 样本 18 ---
预测语言: pt, 置信度: 0.8883
文本截取: Acessibilidade Mapa do Site Ir para o conteúdo   | Ir para o menu   | Ir para a busca   | Ir para o rodapé   • Cidade     • Calendário     • Gabinete...

--- 样本 19 ---
预测语言: es, 置信度: 0.9654
文本截取: Datos Participación Ciudadana Acceso a la Información Derecho a la Información Solicitudes de información Agencia de Acceso a la Información Pública D...

--- 样本 20 ---
预测语言: ru, 置信度: 0.5301
文本截取: ﻿ ОБОГАЋЕН ЈЕДНОСМЕНСКИ РАД (ШКОЛКА 2023/2024. ГОДИНА)    • ПОЧЕТНА   • О ШКОЛИ     • ОРГАНИ ШКОЛЕ       • ШКОЛСКИ ОДБОР       • САВЕТ РОДИТЕЉА     •...

20个样本中，分类器认为是英文的比例: 35.0%
"""