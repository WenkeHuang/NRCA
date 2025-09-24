#!/bin/bash
# bash ./scripts/v1_5/all_eval/eval_all.sh

##### 做eval的脚本 #####
# Pretrain Dataset
EVAL_ON_OKVQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_okvqa.sh
EVAL_ON_VQAV2=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_vqav2.sh
EVAL_ON_GQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_gqa.sh
EVAL_ON_OCRVQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_ocrvqa.sh

# Unseen Dataset
EVAL_ON_VIZWIZ=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_vizwiz.sh
EVAL_ON_SCIENCEQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_sqa.sh
EVAL_ON_TEXTVQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_textvqa.sh
EVAL_ON_POPE=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_pope.sh

EVAL_ON_FLICKR30K=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_flickr30k.sh
EVAL_ON_ICONQA=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_iconqa.sh
EVAL_ON_COCOCAP=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/all_eval/eval_cococap.sh

export CUDA_VISIBLE_DEVICES=4,5,6,7


TRAINING_BATCH=4
TUNE_DECODER_INDEX=30
TUNE_DECODER_LAYER=2
MODEL_ARCHITECTURE=llava-v1.5-7b

for TRAINING_EPOCH in 3  # 1 5 10
do
  echo "TRAINING_EPOCH = $TRAINING_EPOCH"
  for DATASET_TRAIN in   mixture_flickr30k_scienceqa  # mixture_coco_scienceqa  # mixture_coco_iconqa mixture_flickr30k_iconqa
  do
    echo "DATASET_TRAIN = $DATASET_TRAIN"
    for TRAINER_NAME in  LossOneReg #  RandomMask Normal RatioGapTwo LossTwoReg LossOneReg # Normal RatioGapFive RatioGapOne RatioGapTwo
    do
      echo "TRAINER_NAME = $TRAINER_NAME"
      # 模型和结果的前缀
      MODEL_E1=/data0/data_wk/playground/checkpoints-Plus/$TRAINING_BATCH-$TUNE_DECODER_INDEX-$TUNE_DECODER_LAYER-$TRAINING_EPOCH/$DATASET_TRAIN/$MODEL_ARCHITECTURE-$TRAINER_NAME
      RESULT_DIR_E1=/home/huangwenke/Wenke_Project/LLaVAFast/results/$TRAINING_BATCH-$TUNE_DECODER_INDEX-$TUNE_DECODER_LAYER-$TRAINING_EPOCH/$DATASET_TRAIN/$MODEL_ARCHITECTURE-$TRAINER_NAME

      mkdir -p $RESULT_DIR_E1
      ##### 汇总结果的输出路径 #####
      SUMMARY_OUTPUT_DIR=$RESULT_DIR_E1/SummaryResult.txt

      ##### 将当前时间写入文件 #####
      current_time=$(date '+%Y-%m-%d %H:%M:%S')
      summary_file=$SUMMARY_OUTPUT_DIR
      #> "$summary_file"    # Clear out the output file if it exists.
      echo "Current Time: $current_time" >> "$summary_file"
      echo "" >> "$summary_file"
      # 根据 DATASET_TRAIN 的值决定执行哪一条指令
      if [[ "$DATASET_TRAIN" =~ flickr30k ]]; then
          # 执行与 flickr30k 相关的指令
          bash $EVAL_ON_FLICKR30K $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR
      fi
#      if [[ "$DATASET_TRAIN" =~ iconqa ]]; then
#          # 执行与 iconqa 相关的指令
#          bash $EVAL_ON_ICONQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR
#      fi
#      if [[ "$DATASET_TRAIN" =~ scienceqa ]]; then
#          bash $EVAL_ON_SCIENCEQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR
#      fi
#      if [[ "$DATASET_TRAIN" =~ coco ]]; then
#          bash $EVAL_ON_COCOCAP $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR
#      fi


#      if
#          echo "Unknown DATASET_TRAIN: $DATASET_TRAIN"
#      fi  # <- Make sure you have this to close the if-else block
#      ##### 开始eval #####
#      # 在下游任务上测试（OKVQA）
#      # OKVQA FINISH
#  #    bash $EVAL_ON_OKVQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR
#  #
#  #    # TEXTVQA FINISH
#  #    bash $EVAL_ON_TEXTVQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR
#  #
#  #    # gqa FINISH
#  #    bash $EVAL_ON_GQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR
#  #
#  #    # OCRVQA FINISH
#  #    bash $EVAL_ON_OCRVQA $MODEL_E1 $RESULT_DIR_E1 $SUMMARY_OUTPUT_DIR
    done
  done
done