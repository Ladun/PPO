# PPO
Minimal implementation of Proximal Policy Optimization (PPO) in PyTorch
- support discrete and continuous action space 
    - In continuous action space, we use the constance std for sampling.
- utils to plot learning graphs in tensorboard

## Update
- 2023-09-09 
    - Update "Generative Adversarial Imitation Learning(GAIL)"

# Train
Find or make a config file and run the following command.
```
python main.py --config=configs/Ant-v4.yaml 
               --exp_name=test
               --train
```

## Make expert dataset for gail

```
python make_expert_dataset.py --experiment_path=checkpoints/Ant/test
                              --load_postfix=last
                              --minimum_score=5000
                              --n_episode=30
```

# How to play
```
python main.py --experiment_path=checkpoints/Ant/test
               --eval
               --eval_n_episode=50
               --load_postfix=last
               --video_path=videos/Ant
```
- load_path: pretrained model prefix(ex/ number of episode, 'best' or 'last') to play

# Result

| Mujoco Ant-v4 | Mujoco Ant-v4 |
| :-------------------------:|:-------------------------: |
| <video src="https://github.com/Ladun/PPO/blob/master/plots/ant.mp4" width="320" height="240" controls></video>|  ![](https://github.com/Ladun/PPO/blob/master/plots/ant.png) |

| Mujoco Reacher-v4 | Mujoco Reacher-v4 |
| :-------------------------:|:-------------------------: |
| <video src="https://github.com/Ladun/PPO/blob/master/plots/reacher.mp4" width="320" height="240" controls></video> |  ![](https://github.com/Ladun/PPO/blob/master/plots/reacher.png) |

| Mujoco HalfCheetah-v4 | Mujoco HalfCheetah-v4 |
| :-------------------------:|:-------------------------: |
| <video src="https://github.com/Ladun/PPO/blob/master/plots/cheetah.mp4" width="320" height="240" controls></video>|  ![](https://github.com/Ladun/PPO/blob/master/plots/cheetah.png) |



# Reference
- IMPLEMENTATION MATTERS IN DEEP POLICY GRADIENTS: A CASE STUDY ON PPO AND TRPO
- https://github.com/junkwhinger/PPO_PyTorch
- https://github.com/nikhilbarhate99/PPO-PyTorch