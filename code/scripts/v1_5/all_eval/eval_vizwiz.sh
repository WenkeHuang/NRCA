#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


MODEL_PATH=""
MODEL_BASE="/data1/lj_data/llava/model/llava-v1.5-7b-ft"
CKPT="llava-v1.5-7b"
SPLIT="vizwiz"
RESULT_DIR=""


if [ ! -n "$1" ] ;then
    MODEL_PATH=$MODEL_PATH
else
    MODEL_PATH=$1
fi

if [ ! -n "$2" ] ;then
    RESULT_DIR=$RESULT_DIR
else
    RESULT_DIR=$2
fi

mkdir -p $RESULT_DIR


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --model-base $MODEL_BASE \
        --question-file /data0/data_wk/playground/DataEval/vizwiz/llava_test.jsonl \
        --image-folder /data0/data_wk/playground/DataEval/vizwiz/test \
        --answers-file $RESULT_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done



# ##### 注释掉了model-base参数for zero_shot eval #####
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path $MODEL_PATH \
#         --question-file /data1/lj_data/llava/data/eval/vizwiz/llava_test.jsonl \
#         --image-folder /data1/lj_data/llava/data/eval/vizwiz/test \
#         --answers-file $RESULT_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

#wait
#
#output_file=$RESULT_DIR/$SPLIT/$CKPT/merge-vizwiz.jsonl
#
## Clear out the output file if it exists.
#> "$output_file"
#
## Loop through the indices and concatenate each file.
#for IDX in $(seq 0 $((CHUNKS-1))); do
#    cat $RESULT_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
#done


python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /data0/data_wk/playground/DataEval/vizwiz/llava_test.jsonl \
    --result-file $output_file \
    --result-upload-file $RESULT_DIR/$SPLIT/$CKPT/llava-v1.5-7b-vizwiz_uploaded.json
