# PPO
Minimal implementation of Proximal Policy Optimization (PPO) in PyTorch
- support discrete and continuous action space 
    - In continuous action space, we use the constance std for sampling.
- utils to plot learning graphs in tensorboard

# Train
Find or make a config file and run the following command.
```
python main.py --config=configs/Ant-v4.yaml --train
```

# How to play
```
python main.py --config=configs/Ant-v4.yaml --load_path=<num_of_episode>
```
- load_path: pretrained model prefix(ex/ number of episode, 'best' or 'last') to play

# Result

| Mujoco Ant-v4 | Mujoco Ant-v4 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/Ladun/PPO/plots/ant.git) |  ![](https://github.com/Ladun/PPO/plots/ant.png) |

| Mujoco Reacher-v4 | Mujoco Reacher-v4 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/Ladun/PPO/plots/reacher.git) |  ![](https://github.com/Ladun/PPO/plots/reacher.png) |
| Mujoco HalfCheetah-v4 | Mujoco HalfCheetah-v4 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/Ladun/PPO/plots/cheetah.git) |  ![](https://github.com/Ladun/PPO/plots/cheetah.png) |


# Reference
- IMPLEMENTATION MATTERS IN DEEP POLICY GRADIENTS: A CASE STUDY ON PPO AND TRPO
- https://github.com/junkwhinger/PPO_PyTorch
- https://github.com/nikhilbarhate99/PPO-PyTorch