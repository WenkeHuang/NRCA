#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH=""
MODEL_BASE="/data1/lj_data/llava/model/llava-v1.5-7b-ft"
CKPT="llava-v1.5-7b"
SPLIT="vqav2"
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
        --question-file /data0/data_wk/playground/DataEval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
        --image-folder /data0/data_wk/playground/DataEval/vqav2/test2015 \
        --answers-file $RESULT_DIR/$SPLIT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done


wait

output_file=$RESULT_DIR/$SPLIT/merge-vqav2.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python /home/huangwenke/Wenke_Project/LLaVA/scripts/convert_vqav2_for_submission.py \
    --dir $output_file\
    --test_split /data0/data_wk/playground/DataEval/vqav2/llava_vqav2_mscoco_test2015.jsonl \
    --dst $RESULT_DIR/$SPLIT/$CKPT/merge-vqav2_answers_upload.jsonl

