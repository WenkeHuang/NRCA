#!/bin/bash
# bash ./scripts/v1_5/single_training_all.sh

TRAIN_ON_OurDYN=/home/huangwenke/Wenke_Project/LLaVAFast/scripts/v1_5/ourdyn.sh

MODEL_ARCHITECTURE=llava-v1.5-7b

'''
4-2-5
'''
TRAINING_BATCH=4
TUNE_DECODER_INDEX=30
TUNE_DECODER_LAYER=2

for TRAINING_EPOCH in 5   #
do
  for DATASET_TRAIN in scienceqa
  do
    echo "TRAINING_EPOCH=$TRAINING_EPOCH"
    echo "DATASET_TRAIN=$DATASET_TRAIN"
      for TRAINER_NAME in Normal
      # LossTwoReg RatioGapFive RatioGapTwo Normal RatioAbaltionOne RatioGapThree MagnitudeMask RandomMask
      do
        echo "TRAINER_NAME=$TRAINER_NAME"
        bash $TRAIN_ON_OurDYN $DATASET_TRAIN $TRAINER_NAME $MODEL_ARCHITECTURE $TRAINING_BATCH $TUNE_DECODER_INDEX $TUNE_DECODER_LAYER $TRAINING_EPOCH
      done
  done
done
