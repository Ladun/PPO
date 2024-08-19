#!/bin/bash

# python main.py --train --config=configs/BulletHuman.yaml 

# Gail
# python main.py --train --config=configs/Ant-v4-gail.yaml --exp_name=gail

python main.py --eval \
               --experiment_path=checkpoints/BulletHuman/exp20240819124758 \
               --load_postfix=best \
               --video_path=videos/BulletHuman
# python main.py --eval \
#                --experiment_path=checkpoints/Ant/gail \
#                --load_postfix=best \
#                --video_path=videos/Ant/gail

# Make gail expert dataset
# python make_expert_dataset.py --experiment_path=checkpoints/CartPole/exp20240711224529 \
#                               --load_postfix=last \
#                               --minimum_score=4500 \
#                               --n_episode=30 \
#                               --expert_dir=experts/Ant/learnable_std