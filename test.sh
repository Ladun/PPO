#!/bin/bash

# python main.py --train \
#                --config=configs/Ant-v4.yaml \


python main.py --eval \
               --experiment_path=/workspaces/PPO/checkpoints/exp20240321113555 \
               --load_postfix=timesteps15022080
