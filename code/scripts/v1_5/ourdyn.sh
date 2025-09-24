#!/bin/bash
# bash ./scripts/v1_5/ourdyn.sh
# 已修改 --gradient_accumulation_steps 1 ！

export NCCL_P2P_LEVEL=NVL

DATASET_TRAIN=flickr30k
TRAINER_NAME=Normal
MODEL_ARCHITECTURE=llava-v1.5-7b
TRAINING_BATCH=16
TUNE_DECODER_INDEX=30
TUNE_DECODER_LAYER=2
TRAINING_EPOCH=5

if [ ! -n "$1" ] ;then
    DATASET_TRAIN=$DATASET_TRAIN
else
    DATASET_TRAIN=$1
fi

if [ ! -n "$2" ] ;then
    TRAINER_NAME=TRAINER_NAME
else
    TRAINER_NAME=$2
fi

if [ ! -n "$3" ] ;then
    MODEL_ARCHITECTURE=MODEL_ARCHITECTURE
else
    MODEL_ARCHITECTURE=$3
fi

if [ ! -n "$4" ] ;then
    TRAINING_BATCH=TRAINING_BATCH
else
    TRAINING_BATCH=$4
fi

if [ ! -n "$5" ] ;then
    TUNE_DECODER_INDEX=TUNE_DECODER_INDEX
else
    TUNE_DECODER_INDEX=$5
fi

if [ ! -n "$6" ] ;then
    TUNE_DECODER_LAYER=TUNE_DECODER_LAYER
else
    TUNE_DECODER_LAYER=$6
fi

if [ ! -n "$7" ] ;then
    TRAINING_EPOCH=TRAINING_EPOCH
else
    TRAINING_EPOCH=$7
fi


deepspeed --include="localhost:4,5,6,7" --master_port 29120 train_eval.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data0/data_wk/vlm_zoom/$MODEL_ARCHITECTURE \
    --version v1 \
    --data_path /data0/data_wk/playground/DataOptim/data/$DATASET_TRAIN.json \
    --image_folder /data0/data_wk/playground/images \
    --vision_tower ./openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data0/data_wk/playground/checkpoints-Plus/$TRAINING_BATCH-$TUNE_DECODER_INDEX-$TUNE_DECODER_LAYER-$TRAINING_EPOCH/$DATASET_TRAIN/$MODEL_ARCHITECTURE-$TRAINER_NAME \
    --num_train_epochs $TRAINING_EPOCH \
    --per_device_train_batch_size $TRAINING_BATCH \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --tune_decoder_index $TUNE_DECODER_INDEX \
    --tune_decoder_layer $TUNE_DECODER_LAYER \
    --mm_projector_lr 2e-5 \
    --trainer_name $TRAINER_NAME
