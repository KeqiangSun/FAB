#!/usr/bin/env bash

python ./src/train_FAB.py  \
--structure_predictor_train_dir ./data/checkpoints/structure_predictor_train_dir/ \
--video_deblur_train_dir ./data/checkpoints/video_deblur_train_dir/ \
--resnet_train_dir ./data/checkpoints/resnet_train_dir/ \
--end_2_end_train_dir ./data/checkpoints/end_2_end_train_dir/ \
--end_2_end_valid_dir ./data/checkpoints/end_2_end_valid_dir/ \
--max_steps 2000000 \
--resume_structure_predictor False \
--resume_video_deblur False \
--resume_resnet False \
--data_dir ./data/300VW/Images/ \
--img_list ./data/300VW/labels_68pt_256_train_sorted.txt \
--data_dir_valid None \
--img_list_valid None \
--training_period train  &
