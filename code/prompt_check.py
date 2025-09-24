import json

'''
coco cap 
'''
# 输入和输出文件路径
# input_file = "/data0/data_wk/playground/DataOptim/data/coco.json"  # 原始 JSON 文件路径
# output_file = "/data0/data_wk/playground/DataOptim/data/coco.json"  # 替换后保存的 JSON 文件路径
# input_file = "/data0/data_wk/playground/DataOptim/data/coco_original.json"  # 原始 JSON 文件路径
# output_file = "/data0/data_wk/playground/DataOptim/data/coco_original.json"  # 替换后保存的 JSON 文件路径
# 新的提示文本
# new_prompt = "Provide a one-sentence caption for the provided image."

'''
textcap 
'''
# input_file = "/data0/data_wk/playground/DataOptim/fixed_textcaps.json"  # 原始 JSON 文件路径
# output_file = "/data0/data_wk/playground/DataOptim/fixed_textcaps.json"  # 替换后保存的 JSON 文件路径
# input_file = "/data0/data_wk/playground/DataOptim/textcaps.json"  # 原始 JSON 文件路径
# output_file = "/data0/data_wk/playground/DataOptim/textcaps.json"  # 替换后保存的 JSON 文件路径
# # 新的提示文本
# new_prompt = "Provide a one-sentence caption for the provided image."
#


# 读取 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 遍历数据并替换提示
for item in data:
    if "conversations" in item:
        for conversation in item["conversations"]:
            if conversation.get("value", "").startswith("<image>"):
                conversation["value"] = "<image>\n" + new_prompt

# 保存修改后的 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Replaced prompts saved to {output_file}")