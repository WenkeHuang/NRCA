#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


MODEL_PATH=""
MODEL_BASE="/data0/data_wk/vlm_zoom/llava-v1.5-7b"
SPLIT="flickr30k"
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

if [ ! -n "$3" ] ;then
    SUMMARY_OUTPUT_DIR="None"
else
    SUMMARY_OUTPUT_DIR=$3
fi

mkdir -p $RESULT_DIR

 for IDX in $(seq 0 $((CHUNKS-1))); do
     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
         --model-path $MODEL_PATH \
         --model-base $MODEL_BASE \
         --question-file /data0/data_wk/playground/DataEval/flickr30k/fixed_flickr30k_test_div.jsonl \
         --image-folder /data0/data_wk/playground/images/ \
         --answers-file $RESULT_DIR/$SPLIT/${CHUNKS}_${IDX}.jsonl \
         --num-chunks $CHUNKS \
         --chunk-idx $IDX \
         --temperature 0 \
         --conv-mode vicuna_v1 &
 done

wait

output_file=$RESULT_DIR/$SPLIT/$CKPT/merge-flickr30k.jsonl


# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
   cat $RESULT_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python -m llava.eval.eval_flickr30k \
    --annotation-file /data0/data_wk/playground/DataEval/flickr30k/flickr_annotations_30k.csv \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$SPLIT/$CKPT \
    --summary-output-dir $SUMMARY_OUTPUT_DIR