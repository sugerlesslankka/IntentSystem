import re

def process_intent_string(s: str) -> tuple:
    """
    处理输入字符串，返回两个处理后的部分：
      1. 中文意图：获取输入中"英文意图："之前那部分的最后一个"意图"后面的内容，
         如果该内容第一个字符是"："或者"是"，则去掉这个字符。
      2. 英文意图：获取"英文意图："之后部分中，"intention"之后的内容，
         如果第一个单词为"is"（不区分大小写）则去掉它。

    新增功能：去掉字符串中所有的“（”和“）”以及它们之间的内容。
    """
    
    # 去除所有中文括号及其中的内容
    s = re.sub(r'（[^）]*）', '', s)
    
    # 分割字符串为前部分和后部分（基于 "英文意图："）
    if "英文意图：" not in s:
        # 如果没有找到分隔符，则认为两部分均为空字符串
        return "", ""
    
    part_before, _, part_after = s.partition("英文意图：")
    
    # 处理中文意图部分：在 part_before 中寻找最后一个 "意图"
    last_index = part_before.rfind("意图")
    if last_index != -1:
        # 提取从最后一次"意图"之后的内容
        extracted_before = part_before[last_index + len("意图"):]
    else:
        # 如果没有找到 "意图"，则取全部内容
        extracted_before = part_before

    # 去掉首尾空白字符
    extracted_before = extracted_before.strip()
    
    # 如果内容第一个字符是 "：" 或 "是"，则去掉该字符
    if extracted_before and extracted_before[0] in "：是":
        extracted_before = extracted_before[1:].lstrip()

    if extracted_before and extracted_before[-1] in "。":
        extracted_before = extracted_before[:-1].rstrip()
    
    # 处理英文意图部分：在 part_after 中查找 "intention"
    pos = part_after.find("intention")
    if pos != -1:
        # 获取 "intention" 后面的部分
        extracted_after = part_after[pos + len("intention"):]
        extracted_after = extracted_after.strip()
        # 判断第一个单词是否为 "is"，不区分大小写
        words = extracted_after.split()
        if words and words[0].lower() == "is":
            # 去掉第一个单词后重新组合剩下的部分
            extracted_after = " ".join(words[1:])
    else:
        extracted_after = ""
    if extracted_after and extracted_after[-1] in ".":
        extracted_after = extracted_after[:-1].rstrip()

    return extracted_before, extracted_after

import json

# 示例测试
if __name__ == "__main__":
    with open('dataset_2_correct.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    
    # 遍历每个字典，如果存在 key "A" 且对应的值为字符串则处理该值
    for item in data:
        if "analysis" in item and isinstance(item["analysis"], str):
            chinese_intent, english_intent = process_intent_string(item["analysis"])
            # 可以将结果保存到新的键中，也可以覆盖原有值
            item["chinese_intent"] = chinese_intent
            item["english_intent"] = english_intent
    
    # 可选：将处理后的数据写入新文件
    with open("processed_dataset_2.json", 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
