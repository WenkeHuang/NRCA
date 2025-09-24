#!/bin/bash
# bash ./scripts/v1_5/training_all.sh

TRAIN_ON_OurDYN=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/ourdyn.sh

MODEL_ARCHITECTURE=llava-v1.5-7b

'''
4-2-5
'''
TRAINING_BATCH=8
TUNE_DECODER_INDEX=30
TUNE_DECODER_LAYER=2

for TRAINING_EPOCH in 3   #
do
  for DATASET_TRAIN in mixture_flickr30k_scienceqa # mixture_coco_scienceqa # mixture_flickr30k_iconqa  mixture_coco_iconqa
  do
    echo "TRAINING_EPOCH=$TRAINING_EPOCH"
    echo "DATASET_TRAIN=$DATASET_TRAIN"
      for TRAINER_NAME in MagnitudeMask RandomMask  Normal NRCA
      do
        echo "TRAINER_NAME=$TRAINER_NAME"
        bash $TRAIN_ON_OurDYN $DATASET_TRAIN $TRAINER_NAME $MODEL_ARCHITECTURE $TRAINING_BATCH $TUNE_DECODER_INDEX $TUNE_DECODER_LAYER $TRAINING_EPOCH
      done
  done
done
