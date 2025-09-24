#!/bin/bash

# cd /Wenke_Project/LLaVAFast
# bash ./scripts/v1_5/all_eval/eval_all_zs.sh
##### 做eval的脚本 #####
# Pretrain Dataset
EVAL_ON_OKVQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_okvqa.sh
EVAL_ON_VQAV2=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_vqav2.sh
EVAL_ON_GQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_gqa.sh
EVAL_ON_OCRVQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_ocrvqa.sh
EVAL_ON_TEXTVQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_textvqa.sh

# Unseen Dataset
#EVAL_ON_VIZWIZ=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_vizwiz.sh
EVAL_ON_SCIENCEQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_sqa.sh
#EVAL_ON_POPE=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_pope.sh

EVAL_ON_FLICKR30K=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_flickr30k.sh
EVAL_ON_ICONQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_iconqa.sh
EVAL_ON_COCOCAP=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_cococap.sh

MODEL_ARCHITECTURE=llava-v1.5-7b

# 模型和结果的前缀
MODEL_E1=/data0/data_wk/vlm_zoom/llava-v1.5-7b/
RESULT_DIR_E1=/home/huangwenke/Wenke_Project/LLaVAFast/results/$MODEL_ARCHITECTURE/

mkdir -p $RESULT_DIR_E1
##### 汇总结果的输出路径 #####
SUMMARY_OUTPUT_DIR=$RESULT_DIR_E1/SummaryResult.txt

##### 将当前时间写入文件 #####
current_time=$(date '+%Y-%m-%d %H:%M:%S')
summary_file=$SUMMARY_OUTPUT_DIR
#> "$summary_file"    # Clear out the output file if it exists.
echo "Current Time: $current_time" >> "$summary_file"
echo "" >> "$summary_file"

#CUDA_VISIBLE_DEVICES=0,1,2,6 bash $EVAL_ON_OKVQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=0,1,2,6 bash $EVAL_ON_TEXTVQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=0,1,2,6 bash $EVAL_ON_GQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=0,1,2,6 bash $EVAL_ON_OCRVQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=0,1,2,6 bash $EVAL_ON_FLICKR30K $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=0,1,2,6 bash $EVAL_ON_COCOCAP $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR

CUDA_VISIBLE_DEVICES=1,2,3,7 bash $EVAL_ON_ICONQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR
#
#CUDA_VISIBLE_DEVICES=0,1,2,6 bash $EVAL_ON_SCIENCEQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR
