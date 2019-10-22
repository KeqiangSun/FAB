#!/usr/bin/env bash

python ./src/test_FAB.py \
--structure_predictor_train_dir ./data/checkpoints/structure_predictor_train_dir/ \
--voxel_flow_train_dir ./data/checkpoints/voxel_flow_train_dir/ \
--resnet_train_dir ./data/checkpoints/resnet_train_dir/ \
--resume_structure_predictor True \
--resume_video_devlur True \
--resume_resnet True \
--resume_all False \
--data_dir ./data/300VW/Images/ \
--img_list ./data/300VW/labels_68pt_256_train_sorted.txt \
--end_2_end_test_dir ../data/test_results/ &
