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

## Ant-v4
 ![](https://github.com/Ladun/PPO/blob/master/plots/ant.png) 
 
https://github.com/Ladun/PPO/assets/47883234/243abecb-16a4-4ffc-920f-70b644675660

## Reacher-v4
![](https://github.com/Ladun/PPO/blob/master/plots/reacher.png) 


https://github.com/Ladun/PPO/assets/47883234/ff5705e9-b544-41af-a477-10459508f9ac

## HalfCheetah-v4 
 ![](https://github.com/Ladun/PPO/blob/master/plots/cheetah.png) 

https://github.com/Ladun/PPO/assets/47883234/5c1aabc2-f472-4bd1-b4bd-b44cf6cf39dd


# Reference
- IMPLEMENTATION MATTERS IN DEEP POLICY GRADIENT


S: A CASE STUDY ON PPO AND TRPO
- https://github.com/junkwhinger/PPO_PyTorch
- https://github.com/nikhilbarhate99/PPO-PyTorch
