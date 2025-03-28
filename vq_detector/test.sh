#!/bin/bash

set -x

RUN_NAME=$1
OUTPUT_DIR=./exp/$RUN_NAME

# test
python train_vq.py --data_config $OUTPUT_DIR/data_config.yaml --model_config $OUTPUT_DIR/model_config.yaml --ckpt_dir $OUTPUT_DIR --test 2>&1 |tee $OUTPUT_DIR/testvq_webtext.log
