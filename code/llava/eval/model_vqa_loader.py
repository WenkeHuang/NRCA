import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import os

# # 设置工作目录为指定路径
# os.chdir('/home/huangwenke/Wenke_Project/LLaVAFast/')

# # 打印当前工作目录以确认更改
# print("Current working directory:", os.getcwd())

import sys
sys.path.append('/home/huangwenke/Wenke_Project/LLaVAFast')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def calculate_attention_contributions(all_attentions, pos):
    """
    计算每层 system / visual / textual attention 的总和（跨所有 attention head，指定 query token）。

    Args:
        all_attentions (Tensor): shape (num_layers, num_heads, seq_len_query, seq_len_key)
        pos (int): visual token 起始位置

    Returns:
        List[Dict]: 每层一个 dict，包含 system / visual / textual 的贡献值
    """
    num_layers = len(all_attentions)
    results = []

    for layer_idx in range(num_layers):
        attn = all_attentions[layer_idx][0, :, -1]  # shape: (num_heads, seq_len_key)

        system_token_att_map = attn[:, :pos].mean(dim=0).to(torch.float32).detach()
        visual_token_att_map = attn[:, pos:pos + 576].mean(dim=0).to(torch.float32).detach()
        textual_token_att_map = attn[:, pos + 576:].mean(dim=0).to(torch.float32).detach()

        system_contribution = torch.sum(system_token_att_map).item()
        visual_contribution = torch.sum(visual_token_att_map).item()
        textual_contribution = torch.sum(textual_token_att_map).item()

        results.append({
            'layer': layer_idx,
            'system': system_contribution,
            'visual': visual_contribution,
            'textual': textual_contribution
        })

    return results

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path) # 抓取模型名字
    sparse_method = args.sparse_method
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base,model_name,sparse_method=sparse_method)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    num_layers = 32
    system_sum = [0.0] * num_layers
    visual_sum = [0.0] * num_layers
    textual_sum = [0.0] * num_layers
    sample_count = 0

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        # 计算图像的均值和标准差
        # mean = image_tensor.mean(dim=(1, 2, 3), keepdim=True)  # 针对每个样本计算
        # std = image_tensor.std(dim=(1, 2, 3), keepdim=True)
        # 生成与图像相同形状的高斯噪声
        # noise = torch.randn_like(image_tensor) * std + mean
        # image_tensor = (image_tensor + noise)/2

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            if sparse_method == 'SparseVLM':
                output_ids = model.generate(
                    args.scale,
                    args.bias,
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p, # None
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
            elif sparse_method == 'Normal':
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature, # 0.0
                    top_p=args.top_p, # None
                    num_beams=args.num_beams, # 1
                    max_new_tokens=args.max_new_tokens, # 128
                    use_cache=True,
                    output_attentions=True,  # Add !!!!!
                    return_dict_in_generate=True  # Add !!!!
                )

        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip() # 将模型生成的输出标识符（output_ids）转化为可读的文本字符串，并去除掉特殊的标识符

        outputs = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)[0].strip()

        # ********************
        pos = input_ids[0].tolist().index(IMAGE_TOKEN_INDEX)
        all_attentions = output_ids['attentions'][0]  # shape: (num_layers, num_heads, seq_len_query, seq_len_key)

        # 假设你在循环中
        for item in calculate_attention_contributions(all_attentions, pos):
            idx = item['layer']
            system_sum[idx] += item['system']
            visual_sum[idx] += item['visual']
            textual_sum[idx] += item['textual']
        sample_count += 1

        # 打印平均
        for i in range(num_layers):
            print(f"Layer {i:2d} | Avg System: {system_sum[i] / sample_count:.4f} | "
                  f"Avg Visual: {visual_sum[i] / sample_count:.4f} | "
                  f"Avg Textual: {textual_sum[i] / sample_count:.4f}")

        # 假设已有这些总和（每层加起来的）
        total_system = sum(system_sum)
        total_visual = sum(visual_sum)
        total_textual = sum(textual_sum)

        # 总层数 × 样本数
        total_count = num_layers * sample_count

        # 平均贡献
        avg_system = total_system / total_count
        avg_visual = total_visual / total_count
        avg_textual = total_textual / total_count
        avg_total = (avg_system + avg_visual + avg_textual)

        # 打印
        print("=== Overall Average Attention Contribution (All Layers + All Samples) ===")
        print(f"System : {avg_system:.4f}")
        print(f"Visual : {avg_visual:.4f}")
        print(f"Textual: {avg_textual:.4f}")
        print(f"Total  : {avg_total:.4f}")

        # ********************

        ans_id = shortuuid.uuid()

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

import os
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if __name__ == "__main__":
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    parser = argparse.ArgumentParser()
    '''
    原本参数设定
    '''
    # parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="")
    # parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    # parser.add_argument("--answers-file", type=str, default="answer.jsonl")

    # parser.add_argument("--model-path", type=str, default='/data0/data_wk/playground/checkpoints-Plus/4-30-2-3/mixture_flickr30k_scienceqa/llava-v1.5-7b-Normal')
    # parser.add_argument("--model-path", type=str, default='/data0/data_wk/playground/checkpoints-Plus/4-30-2-3/mixture_flickr30k_scienceqa/llava-v1.5-7b-RatioGapTwo')
    # parser.add_argument("--model-path", type=str, default='/data0/data_wk/playground/checkpoints-Plus/4-30-2-3/mixture_flickr30k_scienceqa/llava-v1.5-7b-RandomMask')

    # parser.add_argument("--model-path", type=str, default='/data0/data_wk/playground/checkpoints-Plus/4-30-2-3/mixture_flickr30k_scienceqa/llava-v1.5-7b-LossOneReg')
    # parser.add_argument("--model-path", type=str, default='/data0/data_wk/playground/checkpoints-Plus/4-30-2-3/mixture_flickr30k_scienceqa/llava-v1.5-7b-LossTwoReg')
    parser.add_argument("--model-path", type=str, default='/data0/data_wk/playground/checkpoints-Plus/4-30-2-3/mixture_flickr30k_scienceqa/llava-v1.5-7b-MagnitudeMask')

    # parser.add_argument("--model-path", type=str, default='/data0/data_wk/vlm_zoom/llava-v1.5-7b')

    parser.add_argument("--model-base", type=str, default='/data0/data_wk/vlm_zoom/llava-v1.5-7b')

    # parser.add_argument("--image-folder", type=str, default="/data0/data_wk/playground/images/coco_2014/val2014")
    # parser.add_argument("--question-file", type=str, default="/data0/data_wk/playground/DataEval/okvqa/okvqa_val.jsonl")

    # parser.add_argument("--image-folder", type=str, default="/data0/data_wk/playground/images/textvqa/train_images")
    # parser.add_argument("--question-file", type=str, default="/data0/data_wk/playground/DataEval/textvqa/llava_textvqa_val_v051_ocr.jsonl")


    parser.add_argument("--image-folder", type=str, default="/data0/data_wk/playground/images/")
    parser.add_argument("--question-file", type=str, default="/data0/data_wk/playground/DataEval/flickr30k/fixed_flickr30k_test_div.jsonl")

    parser.add_argument("--answers-file", type=str, default="./answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--sparse_method",type=str,default="Normal") # SparseVLM Normal
    # parser.add_argument("--sparse", action="store_true", help="sparse flag")
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--bias", type=float, default=0.0)
    args = parser.parse_args()

    eval_model(args)